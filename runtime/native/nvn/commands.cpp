#include <algorithm>
#include <unordered_set>
// NVN command buffers: recording into host command lists, and the executor.
//
// A command buffer records host Command entries; EndRecording returns a handle, which (as on the
// console) is an address inside the control memory the application gave the command buffer, so
// handles are unique while their control memory is live and get reused when the app recycles it.

#include <pthread.h>
#include <condition_variable>
#include <deque>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <thread>
#include <chrono>
#include <mutex>
#include <unordered_map>

#include "program.h"
#include "latency.h"
#include "../../native.h"
#include "commands.h"
#include "batch.h"
#include "present.h"
#include "texture.h"
#include "memory.h"

using namespace rt;
using namespace nvn;

namespace nvn {
namespace {
// Values events will hold once the GPU signals recorded so far land (see WaitEvent).
std::mutex event_mutex;
}
u64 current_queue = 0;
namespace {
std::unordered_map<u32*, u32> pending_events;
std::mutex cpu_events_mutex;
std::unordered_set<u32*> cpu_events; // events the CPU signals (nvnEventSignal)
}
void NoteCpuEvent(u32* value) { std::lock_guard lock{cpu_events_mutex}; cpu_events.insert(value); }
bool IsCpuEvent(u32* value) { std::lock_guard lock{cpu_events_mutex}; return cpu_events.count(value) != 0; }
namespace {

struct CommandBufferState {
    CommandList* list = nullptr;
    bool recording = false;
    u64 control = 0, control_size = 0, control_used = 0;
    u64 command_pool = 0, command_offset = 0, command_size = 0;
    u64 callback = 0, callback_data = 0;
};

CommandBufferState*& HostCommandBuffer(u64 object) { return *reinterpret_cast<CommandBufferState**>(object); }

std::mutex lists_mutex;
std::unordered_map<u64, std::shared_ptr<CommandList>> lists;

constexpr u64 HandleBytes = 16;

// Takes HandleBytes of control memory for a new handle, asking the application for more memory
// through its callback when the current chunk is exhausted (NVN_COMMAND_BUFFER_MEMORY_EVENT_
// OUT_OF_CONTROL_MEMORY = 1).
u64 TakeControl(Cpu* c, u64 object, CommandBufferState& cb) {
    if (cb.control_used + HandleBytes > cb.control_size) {
        if (!cb.callback) Fatal("command buffer out of control memory and no callback");
        CallGuest(c, cb.callback, {object, 1, HandleBytes, cb.callback_data});
        if (cb.control_used + HandleBytes > cb.control_size) Fatal("control memory callback added no memory");
    }
    const u64 handle = cb.control + cb.control_used;
    cb.control_used += HandleBytes;
    return handle;
}

} // namespace

std::shared_ptr<const CommandList> LookupCommandList(u64 handle) {
    std::lock_guard lock{lists_mutex};
    auto it = lists.find(handle);
    return it == lists.end() ? nullptr : it->second;
}

// The command list being recorded by a guest command buffer (for command natives).
CommandList& Recording(u64 object) {
    CommandBufferState* cb = HostCommandBuffer(object);
    if (!cb || !cb->recording) Fatal("NVN command recorded outside BeginRecording/EndRecording");
    return *cb->list;
}

// ---- executor ------------------------------------------------------------------------------------
// Each queue runs its submissions in order with its own renderer state. A WaitEvent whose value
// is not there yet parks the rest of its queue's work (the hardware queue would stall) until a
// signal recorded on another queue, or set by the CPU, satisfies it: the async queues wait for the
// main queue's per-frame event, and the main queue for events the CPU sets once the frame's data is
// written. Running the waiting work at once would read the previous frame's results.
struct EventWait { u32* value = nullptr; u32 wanted = 0; bool equal = true; };
enum class Step { Done, Call, Wait };
Step ExecuteFrom(const CommandList& list, size_t& index, size_t end, std::vector<u64>& children, EventWait& wait);
namespace { std::mutex sync_mutex; std::condition_variable sync_cv; }
void NotifySyncSignalled() { { std::lock_guard lock{sync_mutex}; } sync_cv.notify_all(); }
bool WaitSyncSignalled(const u32* state, std::chrono::nanoseconds timeout) {
    std::unique_lock lock{sync_mutex};
    sync_cv.wait_for(lock, timeout, [state] { return __atomic_load_n(state, __ATOMIC_ACQUIRE) != 0; });
    return __atomic_load_n(state, __ATOMIC_ACQUIRE) != 0;
}
bool BackgroundGpuThreads() {
    static const bool on = !std::getenv("OOT_SYNC_EXECUTOR") && !std::getenv("OOT_NO_RETIRE_THREAD") && !std::getenv("OOT_GPU_PROFILE") && !std::getenv("OOT_BUSY_SYNC_WAIT");
    return on;
}
static bool IsDrawOp(u32 op) { switch ((Op)op) { case Op::DrawArrays: case Op::DrawElements: case Op::DrawArraysInstanced: case Op::DrawElementsInstanced: case Op::DrawElementsBaseVertex: return true; default: return false; } }
bool TimelineOn() { static const bool on = std::getenv("OOT_TIMELINE") != nullptr; return on; }
double TimelineMs() { static const auto t0 = std::chrono::steady_clock::now(); return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
namespace {
struct Work { std::shared_ptr<const CommandList> list; size_t next = 0, end = ~size_t{0}; };
struct QueueContext {
    std::deque<Work> work;
    bool parked = false; EventWait wait; std::chrono::steady_clock::time_point parked_at;
    std::shared_ptr<void> state; // renderer state while another queue runs
};
std::recursive_mutex execution_mutex;
std::unordered_map<u64, QueueContext> queues;
u64 active_queue = ~0ull;
bool EventSatisfied(const EventWait& w);
// The work from `from` with every CallCommands resolved now (a parked queue's command buffers may
// be re-recorded before it resumes): runs of the (immutable) lists, without copying commands.
void Flatten(const std::shared_ptr<const CommandList>& list, size_t from, size_t end, std::deque<Work>& out) {
    end = std::min(end, list->commands.size());
    size_t run = from;
    // Only the CallCommands entries matter: recorded lists know where they are.
    std::vector<u32> scanned;
    if (!list->indexed) for (u32 i = 0; i < list->commands.size(); ++i) if ((Op)list->commands[i].op == Op::CallCommands) scanned.push_back(i);
    const auto& calls = list->indexed ? list->calls : scanned;
    for (const u32 call : calls) {
        const size_t i = call;
        if (i < from || i >= end) continue;
        const auto& command = list->commands[i];
        if (run < i) out.push_back({list, run, i});
        for (size_t k = 0; k < command.data.size() / 8; ++k) {
            auto child = LookupCommandList(reinterpret_cast<const u64*>(command.data.data())[k]);
            if (!child) Fatal("call of unknown command handle");
            Flatten(child, 0, ~size_t{0}, out);
        }
        run = i + 1;
    }
    if (run < end) out.push_back({list, run, end});
}
void SwitchTo(u64 queue) {
    if (active_queue == queue) return;
    if (active_queue != ~0ull) queues[active_queue].state = SaveRenderState();
    auto& c = queues[queue];
    if (c.state) RestoreRenderState(c.state); // (a new queue starts from the current state)
    active_queue = queue;
}
// The GPU starts on recorded work while the rest is recorded: a submission every ~1 ms of recording
// (OoT releases a whole frame at once, ~9 ms to record in busy scenes; submitted at the end, the GPU
// waited for all of it). OOT_SUBMIT_EVERY_US=<us> (0: only at the end, as before).
void SubmitWhileRecording() {
    // (every 4 ms where memory is imported: there a submission cost ~1 ms of the executor's time on the Steam Deck, the
    // driver checking each import. With the pools as GPU memory in place a submission is 0.12 ms, and the Deck's GPU
    // stood idle for almost half of a frame waiting for the next piece: every 1 ms there too, Castle Town 21.8 -> 23.6
    // fps, 2026-11-17.)
    static const u64 every = std::getenv("OOT_SUBMIT_EVERY_US") ? std::strtoull(std::getenv("OOT_SUBMIT_EVERY_US"), nullptr, 10) * 1000 : Vk().properties.vendorID == 0x10DE || nvn::PoolsInGpuMemory() ? 1'000'000 : 4'000'000;
    if (!every) return;
    static auto last = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    static u64 flushes_seen = 0;
    if (nvn::FlushCount() != flushes_seen) { flushes_seen = nvn::FlushCount(); last = now; return; } // (submitted meanwhile)
    if ((u64)std::chrono::duration_cast<std::chrono::nanoseconds>(now - last).count() < every) return;
    nvn::Submit();
    flushes_seen = nvn::FlushCount(); last = now;
}
// OOT_LIST_REPEAT=1 (diagnostic): how much of a frame's draws sit in command-list segments that are the same as a
// segment of one of the three frames before: exactly (A), but for uniform buffer addresses and inline uniform data
// (B), but for every buffer address (C), and in their sequence of operations alone (D).
void ListRepeatStats(const Work& w) {
    static const bool on = std::getenv("OOT_LIST_REPEAT") != nullptr;
    if (!on) return;
    static std::array<std::unordered_set<u64>, 4> seen[4]; // [variant][frames ago]
    static u64 draws_total = 0, draws_hit[4]{}, segments = 0, frames = 0;
    const size_t e = std::min(w.end, w.list->commands.size());
    u64 h[4]{0x1234, 0x1234, 0x1234, 0x1234}; u32 draws = 0; bool present = false;
    const auto mix = [](u64& hash, u64 v) { hash = (hash ^ v) * 0x9E3779B97F4A7C15ull; hash ^= hash >> 31; };
    for (size_t k = w.next; k < e; ++k) {
        const Command& c = w.list->commands[k]; const Op op = (Op)c.op;
        draws += IsDrawOp(c.op) || op == Op::DrawMeshTasks || op == Op::DrawMeshTasksIndirect; present |= op == Op::QueuePresent;
        const bool uniform = op == Op::BindUniformBuffer || op == Op::BindUniformBuffers || op == Op::BindUniformBuffersDeferred || op == Op::UpdateUniformBuffer;
        const bool buffer = uniform || op == Op::BindVertexBuffer || op == Op::BindVertexBuffers || op == Op::BindStorageBuffer || op == Op::BindStorageBuffers || op == Op::BindStorageBuffersDeferred || IsDrawOp(c.op);
        u64 full = c.op; for (u64 v : c.x) mix(full, v); for (float f : c.f) { u32 b; std::memcpy(&b, &f, 4); mix(full, b); }
        for (size_t i = 0; i + 8 <= c.data.size(); i += 8) { u64 v; std::memcpy(&v, c.data.data() + i, 8); mix(full, v); }
        mix(h[0], full);
        mix(h[1], uniform ? (u64)c.op << 32 | (u32)c.x[0] << 8 | (u32)c.x[1] : full);
        mix(h[2], buffer ? (u64)c.op << 32 | (IsDrawOp(c.op) ? (u32)c.x[2] : (u32)c.x[0] << 8 | (u32)c.x[1]) : full);
        mix(h[3], c.op);
    }
    if (draws >= 4) {
        draws_total += draws; ++segments;
        for (int v = 0; v < 4; ++v) { bool hit = false; for (int a = 1; a < 4; ++a) hit |= seen[v][a].count(h[v]) != 0; if (hit) draws_hit[v] += draws; seen[v][0].insert(h[v]); }
    }
    if (present) {
        for (int v = 0; v < 4; ++v) { seen[v][3] = std::move(seen[v][2]); seen[v][2] = std::move(seen[v][1]); seen[v][1] = std::move(seen[v][0]); seen[v][0].clear(); }
        if (++frames % 300 == 0 && draws_total) {
            RT_LOG("list repeat: %.0f draws a frame in %.1f segments; in segments seen in the last 3 frames: exact %.0f%%, but uniform addresses %.0f%%, but all buffer addresses %.0f%%, same operations %.0f%%",
                   draws_total / 300.0, segments / 300.0, 100.0 * draws_hit[0] / draws_total, 100.0 * draws_hit[1] / draws_total, 100.0 * draws_hit[2] / draws_total, 100.0 * draws_hit[3] / draws_total);
            draws_total = segments = 0; for (auto& d : draws_hit) d = 0;
        }
    }
}
// The end of the run of commands from `begin` that ExecuteSegmentsParallel takes: state changes, draws and counter
// commands (`end`, or the first command that is none of these). Its draws are counted, and the addresses its reports
// write appended.
size_t ParallelRunEnd(const CommandList& list, size_t begin, size_t end, u32* draws, std::vector<u64>* reports) {
    for (size_t i = begin; i < end; ++i) {
        const Command& command = list.commands[i];
        switch ((Op)command.op) {
        case Op::DrawArrays: case Op::DrawElements: case Op::DrawArraysInstanced: case Op::DrawElementsInstanced: case Op::DrawElementsBaseVertex: ++*draws; break;
        case Op::ReportCounter: if (reports) reports->push_back(command.x[1]); break;
        case Op::ResetCounter: case Op::Barrier: break;
        // (run by the executor in their place, from the state they are met in: render.cpp, IsSerialItem)
        case Op::ClearColor: case Op::ClearColori: case Op::ClearColorui: case Op::ClearDepthStencil: case Op::DrawMeshTasks: case Op::DrawMeshTasksIndirect: {
            // OOT_PARALLEL_END_AT=<bits> (bisection): 1 mesh draws, 2 depth clears, 4 colour clears end a run (the executor runs them the usual way)
            static const int end_at = std::getenv("OOT_PARALLEL_END_AT") ? std::atoi(std::getenv("OOT_PARALLEL_END_AT")) : 0;
            const Op o = (Op)command.op; const bool mesh = o == Op::DrawMeshTasks || o == Op::DrawMeshTasksIndirect;
            if ((mesh && (end_at & 1)) || (o == Op::ClearDepthStencil && (end_at & 2)) || (!mesh && o != Op::ClearDepthStencil && (end_at & 4))) return i;
            break; }
        case Op::BindProgram: case Op::BindUniformBuffer: case Op::BindUniformBuffers: case Op::BindStorageBuffer: case Op::BindStorageBuffers:
        case Op::BindVertexBuffer: case Op::BindVertexBuffers: case Op::BindVertexAttribState: case Op::BindVertexStreamState:
        case Op::BindTexture: case Op::BindTextures: case Op::BindImage: case Op::BindImages: case Op::SetTexturePool: case Op::SetSamplerPool:
        case Op::SetRenderTargets: case Op::SetViewport: case Op::SetViewports: case Op::SetScissor: case Op::SetScissors: case Op::SetDepthRange: case Op::SetDepthRanges:
        case Op::SetBlendColor: case Op::SetStencilMask: case Op::SetStencilValueMask: case Op::SetStencilRef: case Op::SetDepthBias: case Op::SetPointSize: case Op::SetLineWidth:
        case Op::BindBlendState: case Op::BindChannelMaskState: case Op::BindColorState: case Op::BindDepthStencilState: case Op::BindMultisampleState: case Op::BindPolygonState:
        case Op::SetPrimitiveRestart: case Op::SetRenderEnable: case Op::SetRenderEnableConditional: case Op::SetDepthClamp: case Op::SetAlphaRef: case Op::SetSampleMask:
        case Op::SetRasterizerDiscard: case Op::SetDepthBounds: case Op::SetTiledCacheAction: case Op::SetTiledCacheTileSize: case Op::SetGeometryProcessingMode:
        case Op::SetConservativeRasterEnable:
        case Op::PushGraphicsConstants: case Op::PushComputeConstants: case Op::PushGraphicsConstantsDeferred: case Op::PushComputeConstantsDeferred:
        case Op::BindTexturesDeferred: case Op::BindImagesDeferred: case Op::BindUniformBuffersDeferred: case Op::BindStorageBuffersDeferred: break;
        default: return i;
        }
    }
    return end;
}
constexpr u32 ParallelRunDraws = 12;   // a run of fewer draws is not worth a group
constexpr u32 ParallelChunkDraws = 16; // draws a thread resolves in one piece (a run is cut wherever that many are reached)
// A run's commands as pieces of about ParallelChunkDraws draws (each starts from the state the commands before leave).
void AddChunks(std::vector<ParallelSegment>& group, const CommandList& list, size_t begin, size_t end, u32 draws) {
    const u32 pieces = std::max(1u, (draws + ParallelChunkDraws / 2) / ParallelChunkDraws), each = std::max(1u, draws / pieces);
    size_t from = begin; u32 in_piece = 0, made = 1;
    for (size_t i = begin; i < end; ++i) {
        switch ((Op)list.commands[i].op) {
        case Op::DrawArrays: case Op::DrawElements: case Op::DrawArraysInstanced: case Op::DrawElementsInstanced: case Op::DrawElementsBaseVertex:
            if (++in_piece >= each && made < pieces) { group.push_back({&list, from, i + 1}); from = i + 1; in_piece = 0; ++made; }
            break;
        default: break;
        }
    }
    if (from < end) group.push_back({&list, from, end});
}
// With `w` first: the run of such commands at the front of the queue's work, across its segments, executed together.
// False: `w` is to run the usual way, up to w.end (cut where the next run worth a group starts: the rest of it is
// then the queue's next work).
bool RunParallelGroup(QueueContext& c, Work& w) {
    if (!ParallelResolveThreads() || !ParallelDrawsUsable()) return false;
    static std::vector<ParallelSegment> group; static std::vector<u64> reports; static std::vector<Work> held;
    group.clear(); reports.clear(); held.clear();
    const size_t end = std::min(w.end, w.list->commands.size());
    u32 draws = 0;
    const size_t stop = ParallelRunEnd(*w.list, w.next, end, &draws, &reports);
    if (draws < ParallelRunDraws && stop < end) {
        // Not here: the usual way up to the next run that is.
        size_t at = stop + 1, cut = end;
        while (at < end) {
            u32 d = 0;
            const size_t next_stop = ParallelRunEnd(*w.list, at, end, &d, nullptr);
            if (d >= ParallelRunDraws) { cut = at; break; }
            at = next_stop + 1;
        }
        if (cut < end) { c.work.push_front({w.list, cut, w.end}); w.end = cut; }
        return false;
    }
    u32 total = draws;
    AddChunks(group, *w.list, w.next, stop, draws);
    if (stop < end) c.work.push_front({w.list, stop, w.end}); // (what follows the run: the queue's next work)
    else while (!c.work.empty() && group.size() < 192) {     // (the run goes on in the segments that follow)
        Work& next = c.work.front();
        const size_t next_end = std::min(next.end, next.list->commands.size());
        u32 d = 0;
        const size_t next_stop = ParallelRunEnd(*next.list, next.next, next_end, &d, &reports);
        if (next_stop > next.next) { AddChunks(group, *next.list, next.next, next_stop, d); total += d; }
        if (next_stop < next_end) { next.next = next_stop; break; }
        held.push_back(std::move(next)); c.work.pop_front(); // (the lists stay alive while their segments run)
    }
    if (group.size() < 2 || total < 2 * ParallelRunDraws) { // (too little for a group: the usual way, in order)
        for (const auto& segment : group) ExecuteSegmentSerial(*segment.list, segment.begin, segment.end);
        return true;
    }
    {   // OOT_PARALLEL_GROUP_MASK=<hex> (bisection): only the frame's groups whose number (0.., 31 for the rest) is set run in parallel
        static const char* mask_text = std::getenv("OOT_PARALLEL_GROUP_MASK");
        if (mask_text) { static const u64 mask = std::strtoull(mask_text, nullptr, 16); static u64 frame = ~0ull; static u32 ordinal = 0;
            if (const u64 now = CurrentFrameNumber(); now != frame) { frame = now; ordinal = 0; }
            static const bool trace = std::getenv("OOT_PARALLEL_GROUP_TRACE") != nullptr; static u32 shown = 0;
            if (trace && shown++ < 60) RT_LOG("parallel group %u of frame %llu: %zu segments, %u draws", ordinal, (unsigned long long)frame, group.size(), total);
            const u32 bit = std::min(ordinal++, 31u);
            if (!((mask >> bit) & 1)) { for (const auto& segment : group) ExecuteSegmentSerial(*segment.list, segment.begin, segment.end); return true; } } }
    std::sort(reports.begin(), reports.end());
    ExecuteSegmentsParallel(group, reports);
    return true;
}
void Run(u64 queue) {
    SwitchTo(queue);
    auto& c = queues[queue];
    while (!c.work.empty()) {
        SubmitWhileRecording();
        Work w = std::move(c.work.front()); c.work.pop_front();
        if (RunParallelGroup(c, w)) continue;
        size_t i = w.next; std::vector<u64> children; EventWait wait;
        { static const bool shape = std::getenv("OOT_FRAME_SHAPE") != nullptr; // (diagnostic: the segments of ~3 frames, a minute in)
          if (shape) { static const auto t0 = std::chrono::steady_clock::now(); static u32 shown = 0; const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (s > 75 && shown < 700) { ++shown; const size_t e = std::min(w.end, w.list->commands.size()); u32 draws = 0, rts = 0, disp = 0, presents = 0, waits = 0;
                for (size_t k = w.next; k < e; ++k) { const Op op = (Op)w.list->commands[k].op; draws += IsDrawOp((u32)op) || op == Op::DrawMeshTasks || op == Op::DrawMeshTasksIndirect; rts += op == Op::SetRenderTargets; disp += op == Op::DispatchCompute || op == Op::DispatchComputeIndirect; presents += op == Op::QueuePresent; waits += op == Op::WaitEvent || op == Op::WaitSync || op == Op::FenceSync || op == Op::SignalEvent; }
                RT_LOG("shape q %#llx list %p [%zu,%zu) draws %u targets %u dispatches %u sync %u present %u", (unsigned long long)queue, (void*)w.list.get(), w.next, e, draws, rts, disp, waits, presents); } } }
        ListRepeatStats(w);
        const Step step = ExecuteFrom(*w.list, i, w.end, children, wait);
        if (step == Step::Done) continue;
        if (step == Step::Call) {
            c.work.push_front({w.list, i, w.end});
            for (size_t k = children.size(); k-- > 0;) {
                auto child = LookupCommandList(children[k]);
                if (!child) Fatal("call of unknown command handle %#llx", (unsigned long long)children[k]);
                c.work.push_front({std::move(child), 0});
            }
            continue;
        }
        // Park: snapshot the rest of the queue's work.
        std::deque<Work> rest;
        Flatten(w.list, i, w.end, rest);
        for (const auto& more : c.work) Flatten(more.list, more.next, more.end, rest);
        c.work = std::move(rest);
        c.parked = true; c.wait = wait; c.parked_at = std::chrono::steady_clock::now();
        if (TimelineOn()) RT_LOG("T %.2f park queue %#llx for %p = %u", TimelineMs(), (unsigned long long)queue, (void*)wait.value, wait.wanted);
        return;
    }
}
void CaptureForResumedWork();
void ResumeQueues() {
    for (bool progressed = true; progressed;) {
        progressed = false;
        for (auto& [queue, c] : queues) {
            if (!c.parked) continue;
            const bool satisfied = EventSatisfied(c.wait);
            const bool late = !satisfied && std::chrono::steady_clock::now() - c.parked_at > std::chrono::milliseconds(100);
            if (!satisfied && !late) continue;
            if (late) { static u32 shown = 0; if (shown++ < 10) RT_LOG("queue %#llx resumed without its event (%u wanted, %u now)", (unsigned long long)queue, c.wait.wanted, __atomic_load_n(c.wait.value, __ATOMIC_ACQUIRE)); }
            c.parked = false;
            if (TimelineOn()) RT_LOG("T %.2f resume queue %#llx (%zu segments)", TimelineMs(), (unsigned long long)queue, c.work.size());
            CaptureForResumedWork();
            Run(queue);
            if (TimelineOn()) RT_LOG("T %.2f resumed queue %#llx done", TimelineMs(), (unsigned long long)queue);
            progressed = true;
            break; // (Run may have changed the map)
        }
    }
}
} // namespace

// Submissions are executed (translated to Vulkan) by one executor thread, in arrival order: the
// submitting game thread continues at once, as the console's GPU front end works on its own.
// Command lists are immutable once looked up, so the arrival snapshots them (calls resolved).
std::atomic<u64> executor_idle_ns{0}, executor_parked_ns{0}, executor_busy_ns{0}; // (slow-frame diagnosis, render.cpp)
namespace {
std::mutex incoming_mutex;
std::condition_variable incoming_cv, drained_cv;
struct Arrival { u64 first; std::deque<Work> second; void* capture = nullptr; }; // (capture: the CPU's pool writes as they were at the submit, memory.h)
std::deque<Arrival> incoming;
bool wake = false;
u64 arrived = 0, executed = 0;
std::deque<Arrival> taken;         // (the executor's: arrivals it took out of `incoming` and has not started yet)
std::mutex submit_capture_mutex;   // a capture and its place in the order of arrivals are one step
std::atomic<u64> resume_captures{0};
// A parked queue's commands were submitted before the event they waited for. What the game wrote in between - it parks
// the frame's last passes and then writes their constants, the exposure among them - is in no capture made at a submit
// before the work runs: it ran with the constants of a frame or two earlier, in about two frames of three (the whole
// picture brighter or darker for a frame). So work that resumes gets a capture of its own, after the ones made before
// it (they would otherwise put older bytes over it). OOT_NO_RESUME_CAPTURE=1: not.
void CaptureForResumedWork() {
    static const bool off = std::getenv("OOT_NO_RESUME_CAPTURE") != nullptr;
    if (off) return;
    std::vector<void*> earlier; void* now = nullptr;
    {
        std::lock_guard order{submit_capture_mutex};
        for (auto& a : taken) if (a.capture) { earlier.push_back(a.capture); a.capture = nullptr; }
        { std::lock_guard lock{incoming_mutex}; for (auto& a : incoming) if (a.capture) { earlier.push_back(a.capture); a.capture = nullptr; } }
        now = CapturePoolWrites();
    }
    if (earlier.empty() && !now) return;
    nvn::Submit(); // (the work recorded so far keeps the data it was recorded with)
    for (void* c : earlier) ApplyPoolCapture(c);
    ApplyPoolCapture(now);
    resume_captures.fetch_add(1, std::memory_order_relaxed);
}
bool SynchronousExecutor() { static const bool on = std::getenv("OOT_SYNC_EXECUTOR") != nullptr; return on; }
thread_local u32 executing = 0; // (re-entry from AfterFlush actions run during a flush)
void Execute(u64 queue, std::deque<Work>&& work) { // (execution_mutex held)
    struct Depth { Depth() { ++executing; } ~Depth() { --executing; } } depth;
    auto& c = queues[queue];
    for (auto& w : work) c.work.push_back(std::move(w));
    if (!c.parked) Run(queue);
    ResumeQueues();
}
void ExecutorLoop() {
    rt::SetThreadName("nvn-exec");
    for (;;) {
        std::deque<Arrival> items;
        {
            std::unique_lock lock{incoming_mutex};
            // (Also every 20 ms: parked queues whose event never comes resume after a timeout.)
            const auto idle_from = std::chrono::steady_clock::now();
            incoming_cv.wait_for(lock, std::chrono::milliseconds(20), [] { return wake || !incoming.empty(); });
            const u64 idle = (u64)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - idle_from).count();
            executor_idle_ns += idle;
            bool any_parked = false; for (auto& [q, c] : queues) any_parked |= c.parked;
            if (any_parked) executor_parked_ns += idle;
            wake = false;
            items.swap(incoming);
        }
        {
            const auto busy_from = std::chrono::steady_clock::now();
            std::lock_guard lock{execution_mutex};
            taken = std::move(items); items.clear();
            size_t ran = 0;
            while (!taken.empty()) {
                Arrival a = std::move(taken.front()); taken.pop_front(); ++ran;
                if (a.capture) { nvn::Submit(); ApplyPoolCapture(a.capture); } // (the work recorded so far keeps the data it was submitted with)
                current_queue = a.first; Execute(a.first, std::move(a.second)); }
            items.resize(ran); // (only their number is used below)
            ResumeQueues();
            nvn::Submit(); // the work starts on the GPU now
            executor_busy_ns += (u64)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - busy_from).count();
        }
        if (!items.empty()) { std::lock_guard lock{incoming_mutex}; executed += items.size(); drained_cv.notify_all(); }
    }
}
void StartExecutor() { static std::once_flag once; std::call_once(once, [] { std::thread(ExecutorLoop).detach(); }); }
} // namespace

void CommandExecutor::Submit(u64 queue, std::shared_ptr<const CommandList> list) {
    static const bool serial = std::getenv("OOT_SERIAL_QUEUES") != nullptr; // previous behavior
    if (serial) queue = 0;
    std::deque<Work> work;
    Flatten(list, 0, ~size_t{0}, work);
    { static const bool trace = std::getenv("OOT_FLAG_TRACE") != nullptr;
      if (trace) { static std::atomic<u64> lists{0}, with_flags{0}, again{0};
        { std::vector<const CommandList*> met; for (const auto& w : work) { if (std::find(met.begin(), met.end(), w.list.get()) != met.end()) continue; met.push_back(w.list.get()); ++lists; if (!w.list->has_flags) continue; ++with_flags; if (w.list->submissions.fetch_add(1) > 0) ++again; } }
        static std::atomic<u64> n{0}; if (++n % 3000 == 0) RT_LOG("flags: %llu list segments submitted, %llu with flag decisions, %llu of those from a list submitted before", (unsigned long long)lists.load(), (unsigned long long)with_flags.load(), (unsigned long long)again.load()); } }
    if (SynchronousExecutor()) {
        void* const capture = CapturePoolWrites();
        std::lock_guard lock{execution_mutex};
        if (capture) { nvn::Submit(); ApplyPoolCapture(capture); }
        current_queue = queue;
        Execute(queue, std::move(work));
        return;
    }
    StartExecutor();
    { std::lock_guard order{submit_capture_mutex}; // (see CaptureForResumedWork)
      void* const capture = CapturePoolWrites();
      std::lock_guard lock{incoming_mutex}; incoming.push_back({queue, std::move(work), capture}); ++arrived; }
    incoming_cv.notify_one();
}
void CommandExecutor::Submit(u64 queue, u64 handle) {
    auto list = LookupCommandList(handle);
    if (!list) Fatal("submit of unknown command handle %#llx", (unsigned long long)handle);
    Submit(queue, std::move(list));
}
void CommandExecutor::QueueOperation(u64 queue, Op op, u64 argument, u64 argument2) {
    auto list = std::make_shared<CommandList>();
    Command command; command.op = (u32)op; command.x[0] = argument; command.x[1] = argument2;
    list->commands.push_back(std::move(command));
    Submit(queue, std::move(list));
}
// Parked queues whose events may now be satisfied: wake the executor.
void CommandExecutor::Resume() {
    if (SynchronousExecutor()) { if (executing) return; std::lock_guard lock{execution_mutex}; ResumeQueues(); nvn::Submit(); return; }
    StartExecutor();
    { std::lock_guard lock{incoming_mutex}; wake = true; }
    incoming_cv.notify_one();
}
// Everything submitted so far has been translated (parked queues excepted).
void CommandExecutor::Drain() {
    if (SynchronousExecutor()) return;
    std::unique_lock lock{incoming_mutex};
    const u64 target = arrived;
    rt::CoreRelease release;
    drained_cv.wait(lock, [target] { return executed >= target; });
}
void CommandExecutor::Flush() { nvn::Submit(); }
void CommandExecutor::Finish() { Drain(); nvn::Flush(); }

void ExecuteSegmentSerial(const CommandList& list, size_t begin, size_t end) {
    std::vector<u64> children; EventWait wait; size_t index = begin;
    if (ExecuteFrom(list, index, end, children, wait) != Step::Done) Fatal("a command-list segment of state changes and draws did not run to its end");
}
CommandExecutor& Executor() {
    static CommandExecutor executor;
    return executor;
}

} // namespace nvn

// ---- command buffer management --------------------------------------------------------------------
NATIVE("nvnCommandBufferInitialize") {
    HostCommandBuffer(c->x[0]) = new CommandBufferState{};
    Ret(c, 1);
}
NATIVE("nvnCommandBufferFinalize") { delete HostCommandBuffer(c->x[0]); }
NATIVE("nvnCommandBufferSetDebugLabel") {}
NATIVE("nvnCommandBufferSetMemoryCallback") { HostCommandBuffer(c->x[0])->callback = c->x[1]; }
NATIVE("nvnCommandBufferGetMemoryCallback") { Ret(c, HostCommandBuffer(c->x[0])->callback); }
NATIVE("nvnCommandBufferSetMemoryCallbackData") { HostCommandBuffer(c->x[0])->callback_data = c->x[1]; }
NATIVE("nvnCommandBufferGetMemoryCallbackData") { Ret(c, HostCommandBuffer(c->x[0])->callback_data); }
NATIVE("nvnCommandBufferSetCommandMemoryCallbackEnabled") {}
NATIVE("nvnCommandBufferSetControlMemoryCallbackEnabled") {}
NATIVE("nvnCommandBufferSetRecordingMode") {}
NATIVE("nvnCommandBufferSetShaderScratchMemory") {}
NATIVE("nvnCommandBufferAddCommandMemory") {
    auto* cb = HostCommandBuffer(c->x[0]);
    cb->command_pool = c->x[1];
    cb->command_offset = c->x[2];
    cb->command_size = c->x[3];
}
NATIVE("nvnCommandBufferAddControlMemory") {
    auto* cb = HostCommandBuffer(c->x[0]);
    cb->control = c->x[1];
    cb->control_size = c->x[2];
    cb->control_used = 0;
}
NATIVE("nvnCommandBufferGetCommandMemorySize") { Ret(c, HostCommandBuffer(c->x[0])->command_size); }
NATIVE("nvnCommandBufferGetCommandMemoryUsed") { Ret(c, 0); }
NATIVE("nvnCommandBufferGetCommandMemoryFree") { Ret(c, HostCommandBuffer(c->x[0])->command_size); }
NATIVE("nvnCommandBufferGetControlMemorySize") { Ret(c, HostCommandBuffer(c->x[0])->control_size); }
NATIVE("nvnCommandBufferGetControlMemoryUsed") { Ret(c, HostCommandBuffer(c->x[0])->control_used); }
NATIVE("nvnCommandBufferGetControlMemoryFree") {
    auto* cb = HostCommandBuffer(c->x[0]);
    Ret(c, cb->control_size - cb->control_used);
}
NATIVE("nvnCommandBufferIsRecording") { Ret(c, HostCommandBuffer(c->x[0])->recording); }
NATIVE("nvnCommandBufferBeginRecording") {
    auto* cb = HostCommandBuffer(c->x[0]);
    cb->list = new CommandList{};
    cb->recording = true;
}
NATIVE("nvnCommandBufferEndRecording") {
    auto* cb = HostCommandBuffer(c->x[0]);
    const u64 handle = TakeControl(c, c->x[0], *cb);
    cb->recording = false;
    for (u32 i = 0; i < cb->list->commands.size(); ++i) if ((Op)cb->list->commands[i].op == Op::CallCommands) cb->list->calls.push_back(i);
    for (const auto& command : cb->list->commands) if (((Op)command.op == Op::SetRenderEnableConditional || (Op)command.op == Op::ReportCounter) && command.x[7]) { cb->list->has_flags = true; break; }
    cb->list->indexed = true;
    {
        std::lock_guard lock{lists_mutex};
        lists[handle].reset(cb->list);
    }
    cb->list = nullptr;
    Ret(c, handle);
}
NATIVE("nvnDeviceFinalizeCommandHandle") {
    std::lock_guard lock{lists_mutex}; lists.erase(c->x[1]);
}

namespace nvn {

void PrepareWindowTexture(u64 window, s32 index, std::shared_ptr<Texture>& frame) {
    const auto& textures=**reinterpret_cast<std::vector<u64>**>(window+0x78);
    if(index<0||(size_t)index>=textures.size())Fatal("invalid window texture index %d",index);
    auto& texture=*HostTexture(textures[index]);
    if(!frame||frame->width!=texture.width||frame->height!=texture.height||frame->format!=texture.format){
        frame=std::shared_ptr<Texture>(new Texture,RetireTexture);
        frame->width=texture.width;frame->height=texture.height;
        frame->format=texture.format;frame->info=texture.info;
        frame->storage=reinterpret_cast<u64>(frame.get()); // private, never aliases a guest texture
        EnsureImage(*frame);
    }
    // Guest images retire with the graphics batch. Presentation has its own
    // fence, so copy inside that batch into an image owned by the presenter.
    Immediate([&](VkCommandBuffer cmd){
        Transition(cmd,texture,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        Transition(cmd,*frame,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageCopy copy{};
        copy.srcSubresource=copy.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};
        copy.extent={texture.width,texture.height,1};
        vkCmdCopyImage(cmd,texture.image,texture.layout,frame->image,frame->layout,1,&copy);
        Transition(cmd,texture,VK_IMAGE_LAYOUT_GENERAL);
        Transition(cmd,*frame,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    });
}

namespace {
bool EventSatisfied(const EventWait& w) {
    const auto satisfied = [&](u32 current) { return w.equal ? current == w.wanted : (s32)(current - w.wanted) >= 0; };
    if (satisfied(__atomic_load_n(w.value, __ATOMIC_ACQUIRE))) return true;
    std::lock_guard lock{event_mutex};
    auto it = pending_events.find(w.value);
    return it != pending_events.end() && satisfied(it->second);
}
}
Step ExecuteFrom(const CommandList& list, size_t& index, size_t end, std::vector<u64>& children, EventWait& wait) {
    std::lock_guard batch_lock{BatchMutex()};
    static u64 row_stride = 0, image_stride = 0;
    // OOT_WATCH_ADDR=<hex>: in a logged frame, report which command changes that word.
    static const u64 watch = std::getenv("OOT_WATCH_ADDR") ? std::strtoull(std::getenv("OOT_WATCH_ADDR"), nullptr, 16) : 0;
    static const bool always = std::getenv("OOT_WATCH_ALWAYS") != nullptr;
    const bool watching = watch && (always || FrameLogged());
    u32 watched = watching ? *reinterpret_cast<const u32*>(watch) : 0;
    u32 previous_op = ~0u; u64 previous_x[4]{};
    end = std::min(end, list.commands.size());
    // Long command streams go to the GPU in pieces, so it works while the rest is translated.
    static const u32 submit_every = std::getenv("OOT_SUBMIT_EVERY") ? std::atoi(std::getenv("OOT_SUBMIT_EVERY")) : Vk().properties.vendorID == 0x10DE ? 1500 : 0;
    static u32 since_submit = 0;
    static const bool op_stats = std::getenv("OOT_OP_STATS") != nullptr;
    for (; index < end; ++index) {
        const auto& command = list.commands[index];
        NoteFastDrawBreaker(command);
        if (FrameLogged() && command.op > 7 && command.op != 55 && command.op != 56 && command.op != 77 && command.op != 79) RT_LOG("cmd op %u x %#llx %#llx %#llx", command.op, (unsigned long long)command.x[0], (unsigned long long)command.x[1], (unsigned long long)command.x[2]);
        if (op_stats) { static u64 counts[128]; static u64 total = 0; ++counts[std::min<u32>(command.op, 127)];
            if (++total % 1000000 == 0) { std::string t; for (u32 i = 0; i < 128; ++i) if (counts[i] > 10000) { char b[32]; std::snprintf(b, sizeof b, " %u:%llu", i, (unsigned long long)(counts[i] / 1000)); t += b; } RT_LOG("ops (thousands per million):%s", t.c_str()); std::memset(counts, 0, sizeof counts); } }
        if (submit_every && ++since_submit >= submit_every && !SamplesRunOpen()) { since_submit = 0; nvn::Submit(); } // (not inside an occlusion test: its count would be cut in two)
        BatchMutex().Yield(); // (the game's thread waits for the batch lock: its readback or submit goes first)
        static const bool slow_ops = std::getenv("OOT_SLOW_OP") != nullptr;
        struct OpTimer { bool on; u32 op; std::chrono::steady_clock::time_point t0; u64 x0;
            ~OpTimer() { if (!on) return; const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (ms > 30) RT_LOG("slow command: op %u took %.1f ms (x0 %#llx)", op, ms, (unsigned long long)x0); } }
            op_timer{slow_ops, command.op, slow_ops ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}, command.x[0]};
        if (watching) {
            Flush();
            const u32 now = *reinterpret_cast<const u32*>(watch);
            static int reports = 0;
            if (now != watched && (!always || reports++ < 40)) { RT_LOG("watch %#llx: %#x -> %#x by op %u (args %#llx %#llx %#llx %#llx)", (unsigned long long)watch, watched, now, previous_op, (unsigned long long)previous_x[0], (unsigned long long)previous_x[1], (unsigned long long)previous_x[2], (unsigned long long)previous_x[3]); watched = now; }
            previous_op = command.op; std::memcpy(previous_x, command.x, sizeof previous_x);
        }
        const auto* x = command.x;
        switch ((Op)command.op) {
        case Op::SetCopyRowStride: row_stride = x[0]; break;
        case Op::SetCopyImageStride: image_stride = x[0]; break;
        case Op::Barrier: break; // submissions below finish before returning
        // Work executes in submission order, so GPU-side waits are already satisfied.
        case Op::WaitSync: {
            EventWait w{reinterpret_cast<u32*>(x[0]), 1, true};
            if (EventSatisfied(w)) break;
            wait = w; ++index;
            return Step::Wait;
        }
        case Op::WaitEvent: {
            // WaitEvent(event, mode (0 equal, 1 greater-or-equal with wrap), value)
            const u64* e = reinterpret_cast<const u64*>(x[0]);
            EventWait w{static_cast<u32*>(ResolveAddress(PoolOf(e[0])->address + e[1], 4)), (u32)x[2], x[1] == 0};
            if (EventSatisfied(w)) break;
            static const bool prerecord = std::getenv("OOT_PRERECORD") != nullptr; // (experiment: CPU-event gates ignored)
            if (prerecord && IsCpuEvent(w.value)) break;
            { static u32 shown = 0; if (std::getenv("OOT_EVENT_TRACE") && shown++ < 30) RT_LOG("queue %#llx parks for event %#llx = %u (now %u)", (unsigned long long)current_queue, (unsigned long long)x[0], w.wanted, __atomic_load_n(w.value, __ATOMIC_ACQUIRE)); }
            wait = w; ++index;
            return Step::Wait;
        }
        case Op::QueuePresent: PresentWindowTexture(x[0], (s32)x[1]); UploadTexturesAhead(); latency::FrameDone(); break; // (the driver's low latency pacing: no lock held here)
        case Op::ResetCounter: if (FrameLogged()) RT_LOG("counter reset %llu", (unsigned long long)x[0]); ResetRenderCounter((u32)x[0]); break;
        // CPU-visible signals land once the GPU work recorded before them has completed.
        case Op::FenceSync: {
            // The sync is unsignalled until the work before it completes; queues waiting for it
            // (WaitSync) may run once it is recorded (queue order).
            u32* sync = reinterpret_cast<u32*>(x[0]);
            // When the game sees a fence signalled. The game waits for frame N's fence before it releases frame N+1,
            // so with "done" (the GPU finished the work) recording never overlapped the GPU's tail of the frame before:
            // Castle Town 46 fps. "submit": once the work before the fence is submitted, which is when the CPU-written
            // pool data it reads has been copied to the device (RecordPoolTransfers), so the game may write its next
            // frame's data; GPU results are still waited for where they are read (WaitForWrites, page faults): 54 fps.
            // OOT_FENCE=done|submit|record (record: as soon as recorded, an experiment: later CPU writes can reach
            // the frame's draws). OOT_EARLY_FENCES=1 is record, as before.
            static const int fence_mode = [] { const char* e = std::getenv("OOT_FENCE");
                if (std::getenv("OOT_EARLY_FENCES") || (e && !std::strcmp(e, "record"))) return 2;
                return e && !std::strcmp(e, "done") ? 0 : 1; }();
            if (fence_mode == 2) { __atomic_store_n(sync, 1u, __ATOMIC_RELEASE); break; }
            if (fence_mode == 1) {
                __atomic_store_n(sync, 0u, __ATOMIC_RELEASE);
                nvn::Submit();
                __atomic_store_n(sync, 1u, __ATOMIC_RELEASE);
                if (TimelineOn()) RT_LOG("T %.2f fence %p signalled at submit", TimelineMs(), (void*)sync);
                NotifySyncSignalled();
                break;
            }
            __atomic_store_n(sync, 0u, __ATOMIC_RELEASE);
            { std::lock_guard lock{event_mutex}; pending_events[sync] = 1; }
            if (TimelineOn()) RT_LOG("T %.2f fence %p recorded", TimelineMs(), (void*)sync);
            AfterFlush([sync] { __atomic_store_n(sync, 1u, __ATOMIC_RELEASE);
                if (TimelineOn()) RT_LOG("T %.2f fence %p signalled", TimelineMs(), (void*)sync);
                { std::lock_guard lock{event_mutex}; pending_events.erase(sync); }
                NotifySyncSignalled(); });
            break;
        }
        case Op::SignalEvent: {
            // SignalEvent(event, mode (0 write, 1 add), location, flags, value)
            const u64* e = reinterpret_cast<const u64*>(x[0]);
            u32* v = static_cast<u32*>(ResolveAddress(PoolOf(e[0])->address + e[1], 4));
            const bool add = x[1] == 1; const u32 value = (u32)x[4];
            { static u32 shown = 0; if (std::getenv("OOT_EVENT_TRACE") && shown++ < 30) RT_LOG("queue %#llx signals event %#llx %s %u", (unsigned long long)current_queue, (unsigned long long)x[0], add ? "+=" : "=", value); }
            u32 expected;
            {
                std::lock_guard lock{event_mutex};
                auto it = pending_events.find(v);
                const u32 base = it != pending_events.end() ? it->second : __atomic_load_n(v, __ATOMIC_ACQUIRE);
                expected = add ? base + value : value;
                pending_events[v] = expected;
            }
            AfterFlush([v, add, value, expected] {
                if (add) __atomic_fetch_add(v, value, __ATOMIC_ACQ_REL); else __atomic_store_n(v, value, __ATOMIC_RELEASE);
                { std::lock_guard lock{event_mutex};
                  if (auto it = pending_events.find(v); it != pending_events.end() && it->second == expected) pending_events.erase(it); }
                Executor().Resume(); // (wakes the resume worker for queues waiting on it)
            });
            break;
        }
        case Op::ReportCounter: {
            if (FrameLogged()) RT_LOG("counter report %llu @%#llx", (unsigned long long)x[0], (unsigned long long)x[1]);
            ReportRenderCounter((u32)x[0],x[1],x[7]!=0);
            break;
        }
        case Op::ReportValue: {
            u64* out=static_cast<u64*>(ResolveAddress(x[1],16));const u32 value=(u32)x[0];
            NotePendingWrite(x[1],16);
            if (!RecordReportTimestamp(x[1], &value)) AfterFlush([out,value]{out[0]=value;out[1]=GuestTimestampNs();});
            break;
        }
        case Op::CopyBufferToBuffer: {
            // CopyBufferToBuffer(source, destination, size, flags)
            const auto src = GpuLookupAddress(x[0]), dst = GpuLookupAddress(x[1]);
            if (!src.pool || !dst.pool || x[2] > src.available || x[2] > dst.available) Fatal("buffer copy outside mapped memory");
            UniformUpdatesBefore(src.cpu_address, x[2]); UniformUpdatesBefore(dst.cpu_address, x[2]);
            const bool overlap = src.buffer == dst.buffer && src.offset < dst.offset + x[2] && dst.offset < src.offset + x[2];
            if (overlap) { Flush(); std::memmove(reinterpret_cast<void*>(dst.cpu_address), reinterpret_cast<const void*>(src.cpu_address), x[2]); }
            else if (x[2]) NotePendingWrite(x[1], x[2]), Immediate([&](VkCommandBuffer cmd) { VkBufferCopy region{src.offset, dst.offset, x[2]}; vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &region); });
            break;
        }
        case Op::CopyBufferToTexture: {
            const auto& copy = *reinterpret_cast<const TextureCopy*>(command.data.data());
            auto* t = HostTexture(x[1]);
            const u64 bytes = (u64)copy.region[3] * copy.region[4] * copy.region[5] * t->info.bytes;
            const auto range = LookupAddress(x[0]);
            if (!range.pool || bytes > range.available) Fatal("buffer-to-texture source range unavailable");
            // Reconstruct the compact NVN view for the CPU transfer helper.
            u64 view[2]{};
            view[0] = (u64)copy.view.levels_base << 32 | 1ULL << 40 | 1ULL << 56 |
                (u64)copy.view.layers_base | (u64)copy.view.layers_count << 16 | 1ULL << 57;
            Cpu cpu{}; cpu.x[0] = x[1]; cpu.x[1] = (u64)view; cpu.x[2] = (u64)copy.region;
            cpu.x[3] = range.cpu_address; cpu.x[4] = row_stride; cpu.x[5] = image_stride;
            FindNative("nvnTextureWriteTexelsStrided")(&cpu);
            break;
        }
        case Op::CopyTextureToBuffer: {
            const auto& copy = *reinterpret_cast<const TextureCopy*>(command.data.data());
            CopyTextureToGuest(*HostTexture(x[0]), copy, x[3], row_stride, image_stride);
            static const bool trace = std::getenv("OOT_READBACK_TRACE") != nullptr;
            if (trace) { auto* t = HostTexture(x[0]); const float* f = reinterpret_cast<const float*>(ResolveAddress(x[3], 16));
                RT_LOG("readback texture %ux%u fmt %u region %d,%d %dx%d -> %#llx: %g %g %g %g", t->width, t->height, t->format,
                       copy.region[0], copy.region[1], copy.region[3], copy.region[4], (unsigned long long)x[3], f[0], f[1], f[2], f[3]); }
            break;
        }
        case Op::CopyTextureToTexture: {
            const auto* regions=reinterpret_cast<const TextureCopy*>(command.data.data());
            CopyTexture(*HostTexture(x[0]),regions[0],*HostTexture(x[3]),regions[1],(u32)x[6]);
            break;
        }
        case Op::CallCommands:
            for (size_t i = 0; i < command.data.size() / 8; ++i) children.push_back(reinterpret_cast<const u64*>(command.data.data())[i]);
            ++index;
            return Step::Call;
        default: { const bool allowed_before = fast_draw_allowed; const bool ok = ExecuteRenderCommand(command);
            if (allowed_before && !fast_draw_allowed && command.op < 128 && !IsDrawOp(command.op)) ++slow_draw_why[command.op];
            if (!ok) Fatal("NVN command executor pending: operation %u", command.op); }
        }
    }
    return Step::Done;
}

} // namespace nvn

namespace {
Command& Record(Cpu* c, Op op) {
    Command command;
    command.op = (u32)op;
    for (int i = 0; i < 7; ++i) command.x[i] = c->x[i + 1];
    for (int i = 0; i < 8; ++i) std::memcpy(&command.f[i], &c->v[i], 4);
    auto& commands = Recording(c->x[0]).commands;
    commands.push_back(std::move(command));
    return commands.back();
}
void Capture(Command& command, u64 pointer, size_t bytes) {
    command.data.resize(bytes);
    if (bytes) std::memcpy(command.data.data(), reinterpret_cast<void*>(pointer), bytes);
}
}
#define NVN_COMMAND(name) NATIVE("nvnCommandBuffer" #name) { Record(c, Op::name); }
NVN_COMMAND(DrawArrays)
NVN_COMMAND(DrawElements)
NVN_COMMAND(DrawArraysInstanced)
NVN_COMMAND(DrawElementsInstanced)
NVN_COMMAND(DrawArraysIndirect)
NVN_COMMAND(DrawElementsIndirect)
NVN_COMMAND(DispatchCompute)
NVN_COMMAND(DispatchComputeIndirect)
// (the host's program is taken now, on the game's thread: the executor runs this later, when the game may have ended the
// program object or made another in its place - a map change ends hundreds. program.cpp keeps ended programs.)
NATIVE("nvnCommandBufferBindProgram") {
    const u64 object = c->x[1];
    c->x[1] = object ? reinterpret_cast<u64>(HostProgram(object)) : 0;
    Record(c, Op::BindProgram);
    c->x[1] = object;
}
NVN_COMMAND(BindUniformBuffer)
NVN_COMMAND(BindStorageBuffer)
NVN_COMMAND(BindVertexBuffer)
NVN_COMMAND(BindTexture)
NVN_COMMAND(BindImage)
NVN_COMMAND(SetTexturePool)
NVN_COMMAND(SetSamplerPool)
NVN_COMMAND(SetViewport)
NVN_COMMAND(SetScissor)
NVN_COMMAND(SetDepthRange)
NVN_COMMAND(SetStencilMask)
NVN_COMMAND(SetDepthBias)
NVN_COMMAND(SetPointSize)
NVN_COMMAND(SetLineWidth)
NVN_COMMAND(ClearDepthStencil)
NVN_COMMAND(CopyBufferToBuffer)
NVN_COMMAND(Barrier)
NVN_COMMAND(FenceSync)
NVN_COMMAND(WaitSync)
NVN_COMMAND(SignalEvent)
NVN_COMMAND(WaitEvent)
// The game's one-shot flags. Its CPU writes 1 into a report's counter word to ask for something drawn under
// SetRenderEnableConditional(not equal) - doors, vines, the ornaments on a shield, button prompts - and a report of
// another counter after that draw puts the word back to 0. On the console the GPU is done with a frame before the game
// writes the next frame's flags. Here the executor is behind the game's threads: its 0, written when it reached the
// report, landed on the 1 the game had already written for the next frame (that object's shading was skipped for a
// frame: its depth-only pre-pass left a hole that bloom blew up, a one-frame glow), and a condition it read could
// already hold the next frame's 1 (a prompt drawn a frame early from stale data, anywhere on the screen).
// Both now happen in the game's own order, while it records: a report of such a counter clears its word at once,
// and a condition on such a word is decided at once and travels with the command. OOT_FLAGS_LATE=1: as before.
namespace {
std::mutex flag_words_mutex;
std::unordered_set<u64> flag_words; // addresses reports of other counters than samples were made to, or where a condition met the game's request (1 against 0)
std::unordered_set<u64> sample_words; // addresses samples reports were made to: never flags
bool FlagsAtRecord() { static const bool on = std::getenv("OOT_FLAGS_LATE") == nullptr && std::getenv("OOT_FLAG_REPORTS_LATE") == nullptr && std::getenv("OOT_DYNAMIC_RESOLUTION") == nullptr; return on; }
}
NATIVE("nvnCommandBufferReportCounter") {
    Command& command = Record(c, Op::ReportCounter);
    if (!FlagsAtRecord()) return;
    const u64 counter = command.x[0], address = command.x[1];
    { static std::atomic<u64> n{0}; if (++n % 50000 == 0 && std::getenv("OOT_FLAG_TRACE")) RT_LOG("flags: report %llu: counter %llu @%#llx", (unsigned long long)n.load(), (unsigned long long)counter, (unsigned long long)address); }
    { std::lock_guard lock{flag_words_mutex}; if (counter == 1) { flag_words.erase(address); sample_words.insert(address); } else { flag_words.insert(address); sample_words.erase(address); } }
    if (counter == 1) {
        // A samples report's slot reads "visible" from the moment the game records it. The slots are a ring, a new one
        // for every object every frame, and the game's lists run on several queues: a condition on a slot was at times
        // reached before the report that fills it had been executed, and read what the slot held before - 0 - so an
        // object counted at 8,000 pixels a frame earlier lost its shading for a frame (the one-frame glows: its
        // depth-only pre-pass left a hole). The real count is written over it when the GPU has it (Longshot does the
        // same for work held back, f3fe1763). OOT_FLAGS_LATE=1: not.
        if (auto* word = static_cast<u64*>(ResolveAddress(address, 16))) *word = 4096;
        return;
    }
    if (auto* word = static_cast<u64*>(ResolveAddress(address, 16))) { *word = 0; command.x[7] = 1; }
}
NVN_COMMAND(ResetCounter)
NVN_COMMAND(SetTiledCacheAction)
NVN_COMMAND(SetTiledCacheTileSize)
NVN_COMMAND(SetConservativeRasterEnable)
NVN_COMMAND(SetPrimitiveRestart)
NVN_COMMAND(SetRenderEnable)
NVN_COMMAND(SetCopyRowStride)
NVN_COMMAND(SetCopyImageStride)
NVN_COMMAND(SetStencilValueMask)
NVN_COMMAND(SetStencilRef)
#undef NVN_COMMAND
NATIVE("nvnCommandBufferSetDepthClamp") { Record(c, Op::SetDepthClamp); }
NATIVE("nvnCommandBufferSetAlphaRef") { Record(c, Op::SetAlphaRef); }
NATIVE("nvnCommandBufferSetSampleMask") { Record(c, Op::SetSampleMask); }
NATIVE("nvnCommandBufferSetRasterizerDiscard") { Record(c, Op::SetRasterizerDiscard); }
NATIVE("nvnCommandBufferSetDepthBounds") { Record(c, Op::SetDepthBounds); }
NATIVE("nvnCommandBufferDrawElementsBaseVertex") { Record(c, Op::DrawElementsBaseVertex); }
// Push{Graphics,Compute}Constants(cmd, root table, byte offset, size, data): inline writes of
// the hardware root tables (NVN2 push constants; shaders read table 0 as constant bank 17).
// The Deferred forms take a GPU address read when the command executes.
NATIVE("nvnCommandBufferPushGraphicsConstants") { auto& op = Record(c, Op::PushGraphicsConstants); Capture(op, c->x[4], std::min<u64>(c->x[3], 0x100)); }
NATIVE("nvnCommandBufferPushComputeConstants") { auto& op = Record(c, Op::PushComputeConstants); Capture(op, c->x[4], std::min<u64>(c->x[3], 0x100)); }
NATIVE("nvnCommandBufferPushGraphicsConstantsDeferred") { Record(c, Op::PushGraphicsConstantsDeferred); }
NATIVE("nvnCommandBufferPushComputeConstantsDeferred") { Record(c, Op::PushComputeConstantsDeferred); }
// DrawMeshTasks(first task workgroup, task workgroup count)
NATIVE("nvnCommandBufferDrawMeshTasks") { Record(c, Op::DrawMeshTasks); }
NATIVE("nvnCommandBufferDrawMeshTasksIndirect") { Record(c, Op::DrawMeshTasksIndirect); }
NATIVE("nvnCommandBufferClearBuffer") { Record(c, Op::ClearBuffer); }
NATIVE("nvnCommandBufferSetRenderEnableConditional") {
    Command& command = Record(c, Op::SetRenderEnableConditional);
    if (!FlagsAtRecord()) return;
    const u64 address = command.x[1];
    { static std::atomic<u64> all{0}; if (++all % 20000 == 0 && std::getenv("OOT_FLAG_TRACE")) { std::lock_guard lock{flag_words_mutex}; RT_LOG("flags: %llu conditions recorded, %zu flag words known, this one %#llx mode %llu", (unsigned long long)all.load(), flag_words.size(), (unsigned long long)address, (unsigned long long)command.x[0]); } }
    const auto* words = static_cast<const u64*>(ResolveAddress(address, 24));
    if (!words) return;
    const u64 x = words[0], y = words[2];
    {   // (a flag word: one a report of another counter resets, or - as Longshot tells them, nvn_native.c - one where the
        // game's request stands, 1 against 0, that no samples report was made to. The game gates clears with them too.)
        std::lock_guard lock{flag_words_mutex};
        if (!flag_words.count(address)) { if (x == 1 && y == 0 && !sample_words.count(address)) flag_words.insert(address); else return; }
    }
    command.x[7] = ((x == y) == (command.x[0] == 0)) ? 1 : 2; // (decided: 1 drawn, 2 skipped)
    { static std::atomic<u64> n{0}, hits{0}; if (command.x[7]) ++hits; if (++n % 20000 == 0 && std::getenv("OOT_FLAG_TRACE")) RT_LOG("flags: %llu of %llu conditions on flag words decided at record", (unsigned long long)hits.load(), (unsigned long long)n.load()); }
}
NATIVE("nvnCommandBufferReportValue") { Record(c, Op::ReportValue); }
// Deferred bindings retain the client array until submission, instead of
// capturing its current contents as the ordinary binding entry points do.
NATIVE("nvnCommandBufferBindTexturesDeferred") { Record(c,Op::BindTexturesDeferred); }
NATIVE("nvnCommandBufferBindImagesDeferred") { Record(c,Op::BindImagesDeferred); }
NATIVE("nvnCommandBufferBindUniformBuffersDeferred") { Record(c,Op::BindUniformBuffersDeferred); }
NATIVE("nvnCommandBufferBindStorageBuffersDeferred") { Record(c,Op::BindStorageBuffersDeferred); }
NATIVE("nvnCommandBufferSetGeometryProcessingMode") { Record(c,Op::SetGeometryProcessingMode); }
// BindUniformBuffers/BindStorageBuffers(cmd, stage, first, count, NVNbufferRange* {address, size})
NATIVE("nvnCommandBufferBindUniformBuffers") { auto& op = Record(c, Op::BindUniformBuffers); Capture(op, c->x[4], c->x[3] * 16); }
NATIVE("nvnCommandBufferBindStorageBuffers") { auto& op = Record(c, Op::BindStorageBuffers); Capture(op, c->x[4], c->x[3] * 16); }
// UpdateUniformBuffer(cmd, buffer, size, offset, data size, data): an inline write ordered with
// the other commands, recorded as data and applied at execution.
NATIVE("nvnCommandBufferUpdateUniformBuffer") { auto& op = Record(c, Op::UpdateUniformBuffer); Capture(op, c->x[5], c->x[4]); }
// ClearTexture(cmd, texture, view, copy region, colour, mask)
static void RecordClearTexture(Cpu* c, Op o) {
    auto& op = Record(c, o);
    nvn::TextureCopy region{};
    if (c->x[2]) region.view = ReadView(c->x[2]);
    if (c->x[3]) std::memcpy(region.region, Ptr(c, 3), sizeof region.region);
    op.data.resize(sizeof region + 16);
    std::memcpy(op.data.data(), &region, sizeof region);
    std::memcpy(op.data.data() + sizeof region, Ptr(c, 4), 16);
}
NATIVE("nvnCommandBufferClearTexture") { RecordClearTexture(c, Op::ClearTexture); }
NATIVE("nvnCommandBufferClearTexturei") { RecordClearTexture(c, Op::ClearTexturei); }
NATIVE("nvnCommandBufferClearTextureui") { RecordClearTexture(c, Op::ClearTextureui); }
// Console-specific hints and debugging aids with no effect on the host GPU's output.
#define NVN_IGNORED(name) NATIVE("nvnCommandBuffer" #name) {}
NVN_IGNORED(InsertDebugMarker)
// Variable-rate shading is an optimization; everything shades at full rate.
NVN_IGNORED(BindShadingRateImage)
NVN_IGNORED(SetShadingRatePaletteMode)
NVN_IGNORED(ShadingRateImagePalette)
NVN_IGNORED(InsertDebugMarkerGraphics)
NVN_IGNORED(InsertDebugMarkerCompute)
NVN_IGNORED(PushDebugGroup)
NVN_IGNORED(PushDebugGroupGraphics)
NVN_IGNORED(PushDebugGroupCompute)
NVN_IGNORED(PopDebugGroup)
NVN_IGNORED(PopDebugGroupGraphics)
NVN_IGNORED(PopDebugGroupCompute)
NVN_IGNORED(InsertCheckpointDynamic)
NVN_IGNORED(InsertCheckpointStatic)
NVN_IGNORED(DiscardColor)
NVN_IGNORED(DiscardDepthStencil)
NVN_IGNORED(SaveZCullData)
NVN_IGNORED(RestoreZCullData)
NVN_IGNORED(SetZCullZF32CompressionEnable)
NVN_IGNORED(SetStencilCullCriteria)
NVN_IGNORED(SetGraphicsWorkloadPriority)
NVN_IGNORED(SetComputeWorkloadPriority)
NVN_IGNORED(SetComputeOverlapMode)
NVN_IGNORED(SetResourceOccupancy)
NVN_IGNORED(SetSubpixelPrecisionBias)
NVN_IGNORED(SetColorReductionEnable)
NVN_IGNORED(SetColorReductionThresholds)
NVN_IGNORED(SetFloatBlendClamp)
NVN_IGNORED(SetRepresentativeFragmentTest)
NVN_IGNORED(OverrideSubtileSize)
NVN_IGNORED(ResolveDepthBuffer)
NVN_IGNORED(SetConservativeRasterDilate)
NVN_IGNORED(SetConservativeRasterMode)
#undef NVN_IGNORED
NATIVE("nvnCommandBufferSetPolygonOffsetClamp") { Record(c, Op::SetDepthBias); }
NATIVE("nvnCommandBufferClearColor") { auto& op = Record(c, Op::ClearColor); Capture(op, c->x[2], 16); }
NATIVE("nvnCommandBufferClearColori") { auto& op = Record(c, Op::ClearColori); Capture(op, c->x[2], 16); }
NATIVE("nvnCommandBufferClearColorui") { auto& op = Record(c, Op::ClearColorui); Capture(op, c->x[2], 16); }
NATIVE("nvnCommandBufferSetBlendColor") { auto& op = Record(c, Op::SetBlendColor); Capture(op, c->x[1], 16); }
NATIVE("nvnCommandBufferBindBlendState") { auto& op = Record(c, Op::BindBlendState); Capture(op, c->x[1], 8); }
NATIVE("nvnCommandBufferBindChannelMaskState") { auto& op = Record(c, Op::BindChannelMaskState); Capture(op, c->x[1], 4); }
NATIVE("nvnCommandBufferBindColorState") { auto& op = Record(c, Op::BindColorState); Capture(op, c->x[1], 4); }
NATIVE("nvnCommandBufferBindDepthStencilState") { auto& op = Record(c, Op::BindDepthStencilState); Capture(op, c->x[1], 8); }
NATIVE("nvnCommandBufferBindMultisampleState") { auto& op = Record(c, Op::BindMultisampleState); Capture(op, c->x[1], 24); }
NATIVE("nvnCommandBufferBindPolygonState") { auto& op = Record(c, Op::BindPolygonState); Capture(op, c->x[1], 4); }
NATIVE("nvnCommandBufferCallCommands") { auto& op = Record(c, Op::CallCommands); Capture(op, c->x[2], c->x[1] * 8); }
NATIVE("nvnCommandBufferCopyCommands") {
    auto& destination=Recording(c->x[0]).commands;
    std::lock_guard lock{lists_mutex};
    for (u64 i=0;i<c->x[1];++i) {
        const u64 handle=Ptr<const u64>(c,2)[i];
        auto it=lists.find(handle);
        if(it==lists.end())Fatal("copy of unknown command handle %#llx",(unsigned long long)handle);
        // Copy at recording time: the source handle may be finalized or reused
        // before this destination list is submitted.
        const auto& source=it->second->commands;
        destination.insert(destination.end(),source.begin(),source.end());
    }
}
NATIVE("nvnCommandBufferCopyTextureToTexture") {
    auto& op=Record(c,Op::CopyTextureToTexture);
    TextureCopy regions[2]{};
    if(c->x[2])regions[0].view=ReadView(c->x[2]);
    if(c->x[5])regions[1].view=ReadView(c->x[5]);
    std::memcpy(regions[0].region,Ptr(c,3),sizeof regions[0].region);
    std::memcpy(regions[1].region,Ptr(c,6),sizeof regions[1].region);
    Capture(op,(u64)regions,sizeof regions);
}
// CopyTextureToBuffer(cmd, texture, view, region, buffer address, flags)
NATIVE("nvnCommandBufferCopyTextureToBuffer") {
    auto& op = Record(c, Op::CopyTextureToBuffer);
    nvn::TextureCopy copy{};
    if (c->x[2]) copy.view = ReadView(c->x[2]);
    std::memcpy(copy.region, Ptr(c, 3), sizeof copy.region);
    op.data.resize(sizeof copy); std::memcpy(op.data.data(), &copy, sizeof copy);
}
NATIVE("nvnCommandBufferCopyBufferToTexture") {
    auto& op = Record(c, Op::CopyBufferToTexture);
    nvn::TextureCopy copy{};
    if (c->x[3]) copy.view = ReadView(c->x[3]);
    std::memcpy(copy.region, Ptr(c, 4), sizeof copy.region);
    op.data.resize(sizeof copy); std::memcpy(op.data.data(), &copy, sizeof copy);
}
NATIVE("nvnCommandBufferBindVertexAttribState") { auto& op = Record(c, Op::BindVertexAttribState); Capture(op, c->x[2], c->x[1] * 4); }
NATIVE("nvnCommandBufferBindVertexStreamState") { auto& op = Record(c, Op::BindVertexStreamState); Capture(op, c->x[2], c->x[1] * 8); }
NATIVE("nvnCommandBufferBindVertexBuffers") { auto& op = Record(c, Op::BindVertexBuffers); Capture(op, c->x[3], c->x[2] * 16); }
NATIVE("nvnCommandBufferBindTextures") { auto& op = Record(c, Op::BindTextures); Capture(op, c->x[4], c->x[3] * 8); }
NATIVE("nvnCommandBufferBindImages") { auto& op = Record(c, Op::BindImages); Capture(op, c->x[4], c->x[3] * 8); }
NATIVE("nvnCommandBufferSetViewports") { auto& op = Record(c, Op::SetViewports); Capture(op, c->x[3], c->x[2] * 16); }
NATIVE("nvnCommandBufferSetScissors") { auto& op = Record(c, Op::SetScissors); Capture(op, c->x[3], c->x[2] * 16); }
NATIVE("nvnCommandBufferSetDepthRanges") { auto& op = Record(c, Op::SetDepthRanges); Capture(op, c->x[3], c->x[2] * 8); }
NATIVE("nvnCommandBufferSetRenderTargets") {
    auto& op = Record(c, Op::SetRenderTargets);
    if (c->x[1] > 8) Fatal("too many render targets");
    nvn::RenderTargets targets;
    for (u32 i = 0; i < c->x[1]; ++i) {
        targets.colors[i] = Ptr<u64>(c, 2)[i];
        if (c->x[3] && Ptr<u64>(c, 3)[i]) targets.color_views[i] = ReadView(Ptr<u64>(c, 3)[i]);
    }
    targets.depth = c->x[4];
    if (c->x[5]) targets.depth_view = ReadView(c->x[5]);
    op.data.resize(sizeof targets); std::memcpy(op.data.data(), &targets, sizeof targets);
}
