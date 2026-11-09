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
    // (NVIDIA only by default: on the Steam Deck a submission costs ~1 ms of the executor's time)
    static const u64 every = std::getenv("OOT_SUBMIT_EVERY_US") ? std::strtoull(std::getenv("OOT_SUBMIT_EVERY_US"), nullptr, 10) * 1000 : Vk().properties.vendorID == 0x10DE ? 1'000'000 : 4'000'000;
    if (!every) return;
    static auto last = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    static u64 flushes_seen = 0;
    if (nvn::FlushCount() != flushes_seen) { flushes_seen = nvn::FlushCount(); last = now; return; } // (submitted meanwhile)
    if ((u64)std::chrono::duration_cast<std::chrono::nanoseconds>(now - last).count() < every) return;
    nvn::Submit();
    flushes_seen = nvn::FlushCount(); last = now;
}
void Run(u64 queue) {
    SwitchTo(queue);
    auto& c = queues[queue];
    while (!c.work.empty()) {
        SubmitWhileRecording();
        Work w = std::move(c.work.front()); c.work.pop_front();
        size_t i = w.next; std::vector<u64> children; EventWait wait;
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
std::deque<std::pair<u64, std::deque<Work>>> incoming;
bool wake = false;
u64 arrived = 0, executed = 0;
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
        std::deque<std::pair<u64, std::deque<Work>>> items;
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
            for (auto& [queue, work] : items) { current_queue = queue; Execute(queue, std::move(work)); }
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
    if (SynchronousExecutor()) {
        std::lock_guard lock{execution_mutex};
        current_queue = queue;
        Execute(queue, std::move(work));
        return;
    }
    StartExecutor();
    { std::lock_guard lock{incoming_mutex}; incoming.emplace_back(queue, std::move(work)); ++arrived; }
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

VkImage WindowTextureImage(u64 window, s32 index, VkExtent2D* extent, VkCommandBuffer cmd) {
    const auto& textures=**reinterpret_cast<std::vector<u64>**>(window+0x78);
    if(index<0||(size_t)index>=textures.size())Fatal("invalid window texture index %d",index);
    auto& texture=*HostTexture(textures[index]);
    Transition(cmd,texture,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    *extent={texture.width,texture.height};return texture.image;
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
        // Draw fast path (render.cpp): only these commands keep the previous draw's setup valid.
        switch ((Op)command.op) {
        case Op::DrawArrays: case Op::DrawElements: case Op::DrawArraysInstanced: case Op::DrawElementsInstanced:
        case Op::DrawElementsBaseVertex: case Op::BindUniformBuffer: case Op::BindUniformBuffers:
        case Op::BindVertexBuffer: case Op::BindVertexBuffers: case Op::BindVertexAttribState: case Op::BindVertexStreamState:
        case Op::SetViewport: case Op::SetViewports: case Op::SetScissor: case Op::SetScissors: case Op::SetDepthRange: case Op::SetDepthRanges:
        case Op::ReportCounter: case Op::ResetCounter: break;
        // These invalidate only when they change a value (their handlers in render.cpp check).
        case Op::BindProgram: case Op::BindTexture: case Op::SetRenderTargets: case Op::BindBlendState: case Op::BindColorState:
        case Op::BindPolygonState: case Op::SetLineWidth: case Op::BindStorageBuffer: case Op::BindChannelMaskState:
        case Op::SetPrimitiveRestart: case Op::SetPointSize: break;
        case Op::SetBlendColor: case Op::SetDepthBias: break; // (dynamic state, set per draw)
        // No effect on what a draw resolves: render enable is evaluated per draw; barriers are
        // already implied by submission order here.
        case Op::SetRenderEnable: case Op::SetRenderEnableConditional: case Op::Barrier: break;
        default: if (fast_draw_allowed) ++fast_draw_breakers[std::min<u32>(command.op, 127)]; fast_draw_allowed = false;
        }
        if (FrameLogged() && command.op > 7 && command.op != 55 && command.op != 56 && command.op != 77 && command.op != 79) RT_LOG("cmd op %u x %#llx %#llx %#llx", command.op, (unsigned long long)command.x[0], (unsigned long long)command.x[1], (unsigned long long)command.x[2]);
        if (op_stats) { static u64 counts[128]; static u64 total = 0; ++counts[std::min<u32>(command.op, 127)];
            if (++total % 1000000 == 0) { std::string t; for (u32 i = 0; i < 128; ++i) if (counts[i] > 10000) { char b[32]; std::snprintf(b, sizeof b, " %u:%llu", i, (unsigned long long)(counts[i] / 1000)); t += b; } RT_LOG("ops (thousands per million):%s", t.c_str()); std::memset(counts, 0, sizeof counts); } }
        if (submit_every && ++since_submit >= submit_every) { since_submit = 0; nvn::Submit(); }
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
        case Op::QueuePresent: PresentWindowTexture(x[0], (s32)x[1]); break;
        case Op::ResetCounter: if (FrameLogged()) RT_LOG("counter reset %llu", (unsigned long long)x[0]); ResetRenderCounter((u32)x[0]); break;
        // CPU-visible signals land once the GPU work recorded before them has completed.
        case Op::FenceSync: {
            // The sync is unsignalled until the work before it completes; queues waiting for it
            // (WaitSync) may run once it is recorded (queue order).
            u32* sync = reinterpret_cast<u32*>(x[0]);
            static const bool early = std::getenv("OOT_EARLY_FENCES") != nullptr; // experiment: signalled once recorded
            if (early) { __atomic_store_n(sync, 1u, __ATOMIC_RELEASE); break; }
            __atomic_store_n(sync, 0u, __ATOMIC_RELEASE);
            { std::lock_guard lock{event_mutex}; pending_events[sync] = 1; }
            if (TimelineOn()) RT_LOG("T %.2f fence %p recorded", TimelineMs(), (void*)sync);
            AfterFlush([sync] { __atomic_store_n(sync, 1u, __ATOMIC_RELEASE);
                if (TimelineOn()) RT_LOG("T %.2f fence %p signalled", TimelineMs(), (void*)sync);
                std::lock_guard lock{event_mutex}; pending_events.erase(sync); });
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
            if (FrameLogged()) RT_LOG("counter report %llu", (unsigned long long)x[0]);
            ReportRenderCounter((u32)x[0],x[1]);
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
        default: if (!ExecuteRenderCommand(command)) Fatal("NVN command executor pending: operation %u", command.op);
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
NVN_COMMAND(BindProgram)
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
NVN_COMMAND(ReportCounter)
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
NATIVE("nvnCommandBufferSetRenderEnableConditional") { Record(c, Op::SetRenderEnableConditional); }
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
