#include <shared_mutex>
#include "../graphics_settings.h"
#include <set>
// NVN state and synchronous Vulkan execution. Keep submissions ordered while the
// first renderer is brought up; caches avoid rebuilding unchanged pipelines.
#include <unordered_set>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <functional>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <thread>
#include <mutex>
#include <map>
#include <unordered_map>
#include <list>
#include <memory>
#include <string>
#include "../../native.h"
#include "occlusion_spv.h"
#include "indirect_clamp_spv.h"
#include "mesh_advance_spv.h"
#include "timestamp_scale_spv.h"
#include "commands.h"
#include "memory.h"
#include "program.h"
#include "texture.h"
#include "shader_compiler.h"
#include "batch.h"
#include "upscale.h"
#include "longshot_upscale.h"
#include "vk_temporal.h" // (third_party/longshot-upscale)
#include "vk_fsr3.h"
#include "vk_fsr3_inputs.h"
#include "dlss_actor.h"
#include "pipeline_list.h"
using namespace rt;
namespace nvn {
bool LsReadSubmitted(u64 address, void* dst, u64 size);
void LsNoteDraw(Texture* colour, Texture* depth, VkExtent2D extent, u64 camera, u64 camera_size); // (the vendored upscalers, below)
std::atomic<u64> skin_reject[12];
bool LsSkinned(const Command& command, mkwvk::SkinnedDraw& k, u64 program);
void LsNoteSkinned(mkwvk::SkinnedDraw& k, Texture* depth, VkExtent2D extent);
void PreSubmit(); // (before each submission: report processing and timestamps)
inline void ReportBindingRate(){static u64 last=0;const u64 now=binding_generation.load();RT_LOG("texture/sampler registrations: %.1f per frame",(now-last)/300.0);last=now;}
void EnsureCounterRings(); // (defined with the GPU counters below)

namespace {
struct Buffer { u64 address = 0, size = 0; };
struct StageState { std::array<Buffer, 18> uniforms; std::array<Buffer, 16> storage; std::array<u64, 32> textures{}, images{}; const ShaderCode* shader = nullptr; };
struct State {
    std::array<StageState, 9> stages; // NVN2 stages: 0-4 graphics, 5 compute, 6-7 task/mesh, 8 (driver-special)
    std::array<Buffer, 16> vertices;
    std::array<u32, 32> attributes{};
    std::array<std::array<u32, 2>, 16> streams{};
    TexturePool* textures = nullptr;
    SamplerPool* samplers = nullptr;
    RenderTargets targets{};
    u32 color_count = 0;
    VkViewport viewport{0,0,1920,1080,0,1};
    VkRect2D scissor{{0,0},{1920,1080}};
    u64 blends[8]{0x01020102110c0000ULL,0x01020102110c0000ULL,0x01020102110c0000ULL,0x01020102110c0000ULL,
        0x01020102110c0000ULL,0x01020102110c0000ULL,0x01020102110c0000ULL,0x01020102110c0000ULL};
    u32 color = 0x08030000, channel = ~0u, polygon = 0x14;
    u64 depth = 0x1118111800000022ULL;
    float blend_color[4]{}, depth_bias[3]{}, line_width = 1;
    u32 stencil_ref[2]{}, stencil_compare[2]{0xff,0xff}, stencil_write[2]{0xff,0xff};
    bool restart = false, render_enable = true, depth_clamp = false, discard = false;
    u64 render_condition = 0;
    bool render_if_equal = true;
    u8 render_decided = 0; // a condition on one of the game's flag words, decided when it was recorded: 1 drawn, 2 skipped (commands.cpp)
    u32 sample_mask = ~0u;
    bool depth_bounds = false;
    float depth_bounds_range[2]{0, 1};
    u32 primitive = 4;
    float point_size = 1;
    // Root tables (NVN2 push constants): [0] graphics, [1] compute; 8 tables of 256 bytes.
    u32 root[2][8][64]{};
} main_state;
// The state the commands of this thread apply to: the executor's, or a copy a helper thread resolves one command
// list's draws from (ExecuteSegmentsParallel).
thread_local State* tls_state=&main_state;
// GPU sample counters and predicates (occlusion.comp). Reports accumulate copied query results
// into a running total on the GPU and write it to guest memory; a conditional draw whose operands
// are still being produced gets its predicate computed on the GPU and runs under
// VK_EXT_conditional_rendering. The CPU never waits for occlusion results.
// Conditional draws by how their predicate was decided (since the last fps report).
u64 frame_clears_skipped=0,frame_clears_predicated=0,frame_net=0,frame_net_hash=0,frame_dispatches=0,frame_dispatch_hash=0;
u64 frame_bails=0,frame_fallbacks=0,frame_groups=0,frame_cond_drawn=0,frame_cond_skipped=0,frame_cond_gpu=0; // (OOT_FRAME_EVENTS)
// What ran between a samples reset and its report (the executor's order). Only a run of the game's depth-only bounding
// boxes gives a count to trust: see ReportRenderCounter.
// OOT_OCC_QUERIES=1: the game's samples reports get the GPU's occlusion counts (it then leaves out what reads 0).
// Off by default: every report reads "visible" (4096), as Hookshot's and Longshot's replays did before cd906265 and still
// do wherever a count is not sure. With the counts, one frame in ten in the woods had a wrong 0 (a run cut in two by a
// submission, a slot read before its report ran, and more not found yet): that object's shading skipped for the frame,
// its depth pre-pass left as a hole that bloom blew up - Link, vines, trees glowing for single frames.
// (the settings window's Occlusion Tests, read at start; OOT_OCC_QUERIES=0|1 overrides)
bool OcclusionQueries(){static const bool on=[]{const char* e=std::getenv("OOT_OCC_QUERIES");return e?std::atoi(e)!=0||!*e:gfx::Current().occlusion!=0;}();return on;}
struct RunNote { u32 draws = 0, mesh = 0, colour = 0, not_run = 0; } run_note;
bool samples_run_open = false; u64 samples_run_batch = 0; // a samples reset whose report has not come; the submission it began in
u64 run_split=0;
u64 run_classes[5]{}; // (statistics: counted, with a mesh draw, with a colour draw, with a draw that did not run, empty)
struct PredicateStats { u64 cpu = 0, flushed = 0, bracketed = 0, dispatched = 0, processing = 0, processed_reports = 0, depth_tested = 0, waited = 0, wait_ns = 0; } predicate_stats;
struct GpuCounters {
    bool ready=false;
    VkBuffer results=VK_NULL_HANDLE,total=VK_NULL_HANDLE,predicates=VK_NULL_HANDLE,entries=VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout=VK_NULL_HANDLE;VkPipelineLayout layout=VK_NULL_HANDLE;VkPipeline pipeline=VK_NULL_HANDLE;
    PFN_vkCmdBeginConditionalRenderingEXT begin=nullptr;PFN_vkCmdEndConditionalRenderingEXT end=nullptr;
    // Reports recorded but not yet processed (one dispatch processes them all, before the first
    // consumer or the submit). Layout matches occlusion.comp's Entry.
    struct Entry { u64 address; u32 end, slot, kind, pad; };
    static constexpr u32 EntryCapacity=1u<<18;
    std::vector<Entry> pending;
    u32 entries_used=0;          // entries consumed from the mapped list in this batch
    Entry* entry_map=nullptr;
    u32 processed=0;             // query results already added to the GPU total in this batch
    u32 segment_query=~0u;       // occlusion query open in the current render pass instance
    u64 last_report=0;           // address of the previous samples report (0 after a reset)
    struct ReportInfo { u32 slot; bool follows_previous, after_reset; };
    std::unordered_map<u64,ReportInfo> reports; // this batch's samples reports
    u32 next_predicate=0;
    u32* predicate_values=nullptr; // mapped predicates (OOT_COUNTER_CHECK)
    static constexpr u32 PredicateCapacity=1u<<20;
} gpu_counters;
void CounterDispatch(u32 mode,u32 first,u32 count,u64 guest,u32 copy_queries);
void ProcessReports();
void PredicatedClear(u32 predicate,bool inverted,Texture* t,const TextureView& view,VkImageAspectFlags aspect,const u8* color,float depth_value,u32 stencil_value);
u32 draw_predicate=~0u; // predicate slot of the draw being recorded (~0: unconditional)
bool draw_inverted=false; // the draw runs when that slot is zero
bool LogThisFrame(); // (below)
bool PredicateOperandsFinal(){
    // Reports and GPU writes earlier in the open batch can feed the predicate.
    static const bool sync_predicates=std::getenv("OOT_SYNC_PREDICATES")!=nullptr;
    // OOT_ASYNC_OCCLUSION=1: samples reports still in flight are not waited for; the predicate
    // uses the last completed result at that address (one or two frames old).
    static const bool async_occlusion=std::getenv("OOT_ASYNC_OCCLUSION")!=nullptr;
    if(HasPendingWrite((*tls_state).render_condition,24))return false;
    return !HasPendingReport((*tls_state).render_condition,24)||sync_predicates||async_occlusion;
}
// Values of reports whose value is known when recorded (counters other than samples passed
// report 0): predicates over them are decided on the CPU, even while the write is in flight.
std::unordered_map<u64,u64> known_reports;
// What the map holds for the addresses written last (executor only): a report that reads the same as the frame before
// - nearly all of them, thousands a frame - is then one comparison instead of a walk through the map (0.9 ms a
// frame in a heavy view). OOT_REPORT_MEMO=1 turns it on.
struct KnownMemo{u64 address=~0ull,value=0;};
KnownMemo known_memo[8192];
inline void SetKnownReport(u64 address,u64 value){
    static const bool memo=std::getenv("OOT_REPORT_MEMO")!=nullptr; // (off unless asked for: nothing measured)
    auto& m=known_memo[(address>>3)&8191];
    if(memo&&m.address==address&&m.value==value)return;
    known_reports[address]=value;m.address=address;m.value=value;
}
inline void EraseKnownReport(u64 address){
    auto& m=known_memo[(address>>3)&8191];if(m.address==address)m.address=~0ull;
    known_reports.erase(address);
}
bool EvaluatePredicateOnCpu(){
    const auto* operands=static_cast<const u8*>(ResolveAddress((*tls_state).render_condition,24));
    u64 initial,current;
    std::memcpy(&initial,operands,8);std::memcpy(&current,operands+16,8);
    if(const auto k=known_reports.find((*tls_state).render_condition);k!=known_reports.end())initial=k->second;
    if(const auto k=known_reports.find((*tls_state).render_condition+16);k!=known_reports.end())current=k->second;
    return (initial==current)==(*tls_state).render_if_equal;
}
// Clears and other non-draw work: exact, waiting for the operands when needed.
bool RenderEnabled() {
    if (!(*tls_state).render_enable) return false;
    static const bool ignore_conditions=std::getenv("OOT_NO_RENDER_CONDITION")!=nullptr; // (diagnostic)
    if (!(*tls_state).render_condition||ignore_conditions) return true;
    if(!PredicateOperandsFinal())Flush();
    return EvaluatePredicateOnCpu();
}
// Draws: operands still in flight are evaluated on the GPU (sets draw_predicate).
bool DrawEnabled() {
    draw_predicate=~0u;draw_inverted=false;
    if (!(*tls_state).render_enable) return false;
    if (!(*tls_state).render_condition) return true;
    if ((*tls_state).render_decided) { ++(((*tls_state).render_decided == 1) ? frame_cond_drawn : frame_cond_skipped); return (*tls_state).render_decided == 1; }
    {static const bool ignore_conditions=std::getenv("OOT_NO_RENDER_CONDITION")!=nullptr;if(ignore_conditions)return true;} // (diagnostic)
    if(PredicateOperandsFinal()){++predicate_stats.cpu;const bool drawn=EvaluatePredicateOnCpu();++(drawn?frame_cond_drawn:frame_cond_skipped);
        {static const bool cl=std::getenv("OOT_CONDITION_LOG")!=nullptr;static u64 lines=0; // (diagnostic: every CPU decision, 40,000 of them)
         if(cl&&lines<40000&&CurrentFrameNumber()>3300){++lines;const u64 at=(*tls_state).render_condition;const auto* o=static_cast<const u8*>(ResolveAddress(at,24));u64 a,b,c2;std::memcpy(&a,o,8);std::memcpy(&b,o+8,8);std::memcpy(&c2,o+16,8);
            const auto k0=known_reports.find(at),k1=known_reports.find(at+16);
            RT_LOG("cd %llu %llx %llu %llu %llu k %lld %lld %s %d",(unsigned long long)CurrentFrameNumber(),(unsigned long long)at,(unsigned long long)a,(unsigned long long)b,(unsigned long long)c2,
                k0==known_reports.end()?-1ll:(long long)k0->second,k1==known_reports.end()?-1ll:(long long)k1->second,drawn?"D":"S",(int)(*tls_state).render_if_equal);}}
        {static const bool verify=std::getenv("OOT_CONDITION_VERIFY")!=nullptr; // (diagnostic: the same question again once everything recorded so far has run)
         if(verify){const u64 at=(*tls_state).render_condition;const bool if_equal=(*tls_state).render_if_equal;
            const auto* o=static_cast<const u8*>(ResolveAddress(at,24));u64 a0,c0;std::memcpy(&a0,o,8);std::memcpy(&c0,o+16,8);
            const bool k0=known_reports.count(at)!=0,k1=known_reports.count(at+16)!=0;
            AfterFlush([at,if_equal,drawn,a0,c0,k0,k1]{const auto* p=static_cast<const u8*>(ResolveAddress(at,24));u64 a,c;std::memcpy(&a,p,8);std::memcpy(&c,p+16,8);
                const bool later=(a==c)==if_equal;static u32 n=0;static u64 total=0,differ=0;++total;
                if(later!=drawn){++differ;if(n++<40)RT_LOG("condition @%#llx decided %s from %llu/%llu (known %d/%d); after the batch memory holds %llu/%llu -> %s",(unsigned long long)at,drawn?"drawn":"SKIPPED",
                    (unsigned long long)a0,(unsigned long long)c0,(int)k0,(int)k1,(unsigned long long)a,(unsigned long long)c,later?"drawn":"skipped");}
                if(total%20000==0)RT_LOG("condition verify: %llu of %llu decisions differ from memory after their batch",(unsigned long long)differ,(unsigned long long)total);});}}
        {static const bool cs=std::getenv("OOT_CONDITION_STATS")!=nullptr; // (diagnostic: conditions that rarely differ from their usual answer, and the counts they had then)
         if(cs){struct S{u64 drawn=0,skipped=0;};static std::map<std::pair<u64,bool>,S> seen;static u64 n=0;static u32 shown=0;
            auto& e=seen[{(*tls_state).render_condition,(*tls_state).render_if_equal}];const u64 total=e.drawn+e.skipped;
            const bool rare=total>200&&(drawn?e.drawn*50<total:e.skipped*50<total);
            if(rare&&shown++<300){const auto* o=static_cast<const u8*>(ResolveAddress((*tls_state).render_condition,24));u64 a,b,c;std::memcpy(&a,o,8);std::memcpy(&b,o+8,8);std::memcpy(&c,o+16,8);
                RT_LOG("rare condition @%#llx (draw if %s): %llu %llu %llu -> %s (usually: %llu drawn, %llu skipped) depth %#x",(unsigned long long)(*tls_state).render_condition,(*tls_state).render_if_equal?"equal":"not equal",
                    (unsigned long long)a,(unsigned long long)b,(unsigned long long)c,drawn?"drawn":"skipped",(unsigned long long)e.drawn,(unsigned long long)e.skipped,(u32)(*tls_state).depth);}
            ++(drawn?e.drawn:e.skipped);++n;}}
        if(LogThisFrame()){const auto* o=static_cast<const u8*>(ResolveAddress((*tls_state).render_condition,24));u64 a,b,c;std::memcpy(&a,o,8);std::memcpy(&b,o+8,8);std::memcpy(&c,o+16,8);
            RT_LOG("condition @%#llx (draw if %s): %llu %llu %llu known %d/%d -> %s",(unsigned long long)(*tls_state).render_condition,(*tls_state).render_if_equal?"equal":"not equal",(unsigned long long)a,(unsigned long long)b,(unsigned long long)c,
                (int)known_reports.count((*tls_state).render_condition),(int)known_reports.count((*tls_state).render_condition+16),drawn?"drawn":"SKIPPED");}
        return drawn;}
    // OOT_CONDITION_WAIT=1: a condition whose reports are in work already submitted waits for that work and is decided
    // from its results (exact, but the executor then stands still for the GPU: 1.5-3.4 ms a frame). Off: such conditions
    // are decided on the GPU below, in order with the reports.
    {static const bool wait=std::getenv("OOT_CONDITION_WAIT")!=nullptr;
     if(wait&&!resolve_worker){
        const auto t0=std::chrono::steady_clock::now();
        const bool open=WaitForSubmittedReports((*tls_state).render_condition,24);
        predicate_stats.wait_ns+=(u64)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-t0).count();++predicate_stats.waited;
        if(!open&&PredicateOperandsFinal()){++predicate_stats.cpu;return EvaluatePredicateOnCpu();}
        if(open){++predicate_stats.flushed;Flush();return EvaluatePredicateOnCpu();}}}
    if(LogThisFrame())RT_LOG("condition @%#llx: operands still on their way (pending report %d, pending write %d)",(unsigned long long)(*tls_state).render_condition,(int)HasPendingReport((*tls_state).render_condition,24),(int)HasPendingWrite((*tls_state).render_condition,24));
    // (OOT_OCCLUSION_SHORTCUT=1, as it was: a "draw if visible" draw that is depth tested is drawn without its result,
    // since the depth test rejects all of it when it is in fact occluded. That holds only when the report counted the
    // draw's own object. The game also conditions draws on other things' reports - the white light in a house's
    // doorway, drawn over the door; Link's hat the same - and with draws resolved ahead the executor reaches the
    // conditions before the GPU has the results, which it never did alone: those draws flashed for single frames.
    // Now the condition is decided on the GPU, in order with the reports; no flash in 4,800 frames.)
    static const bool exact_occlusion=std::getenv("OOT_OCCLUSION_SHORTCUT")==nullptr;
    const u32 depth_func=((*tls_state).depth>>4)&15;
    if(!exact_occlusion&&!(*tls_state).render_if_equal&&((*tls_state).depth&1)&&(*tls_state).targets.depth&&(depth_func==2||depth_func==3||depth_func==4)&&HasPendingReport((*tls_state).render_condition,24)){
        ++predicate_stats.depth_tested;return true;}
    if(!gpu_counters.ready||HasPendingWrite((*tls_state).render_condition,24)){++predicate_stats.flushed;Flush();return EvaluatePredicateOnCpu();}
    EnsureCounterRings();
    if(PredicateOperandsFinal())return EvaluatePredicateOnCpu(); // (a ring flush completed the reports)
    // The usual pattern: two adjacent reports bracket an occlusion test, and the draw depends on
    // whether samples passed between them, which processing stores in the second report's slot.
    const auto pair=gpu_counters.reports.find((*tls_state).render_condition+16);
    // Or (this game): the counter is reset before each occlusion test, and the report at +0 is
    // compared with a final zero at +16; samples since the reset are then the report's value.
    const auto single=gpu_counters.reports.find((*tls_state).render_condition);
    u64 second=1;
    if(single!=gpu_counters.reports.end()&&single->second.after_reset&&!HasPendingReport((*tls_state).render_condition+16,8)&&!HasPendingWrite((*tls_state).render_condition+16,8)){
        std::memcpy(&second,ResolveAddress((*tls_state).render_condition+16,8),8);
        if(const auto k=known_reports.find((*tls_state).render_condition+16);k!=known_reports.end())second=k->second;
    }
    const bool after_reset=single!=gpu_counters.reports.end()&&single->second.after_reset&&second==0;
    const bool bracketed=after_reset||(pair!=gpu_counters.reports.end()&&pair->second.follows_previous);
    if(!gpu_counters.pending.empty())ProcessReports();
    {static u32 shown2=0;if(std::getenv("OOT_PREDICATE_TRACE")&&shown2++<20){u64 a=0,b=0;const bool w=PendingWriteRange((*tls_state).render_condition,24,&a,&b);
        RT_LOG("predicate @%#llx: pending report %d, pending write %d (%#llx-%#llx), known %d/%d",(unsigned long long)(*tls_state).render_condition,(int)HasPendingReport((*tls_state).render_condition,24),(int)w,(unsigned long long)a,(unsigned long long)b,
            (int)known_reports.count((*tls_state).render_condition),(int)known_reports.count((*tls_state).render_condition+16));}}
    {static u32 shown=0;if(std::getenv("OOT_PREDICATE_TRACE")&&shown++<30){
        const auto a=gpu_counters.reports.find((*tls_state).render_condition);
        RT_LOG("predicate @%#llx eq %d: report@+0 %s (slot %u follows %d), report@+16 %s (follows %d), last report %#llx",(unsigned long long)(*tls_state).render_condition,(int)(*tls_state).render_if_equal,
            a!=gpu_counters.reports.end()?"yes":"no",a!=gpu_counters.reports.end()?a->second.slot:0,a!=gpu_counters.reports.end()?(int)a->second.follows_previous:-1,
            pair!=gpu_counters.reports.end()?"yes":"no",pair!=gpu_counters.reports.end()?(int)pair->second.follows_previous:-1,(unsigned long long)gpu_counters.last_report);}}
    ++frame_cond_gpu;
    if(bracketed){++predicate_stats.bracketed;draw_predicate=after_reset?single->second.slot:pair->second.slot;draw_inverted=(*tls_state).render_if_equal;}
    else{
        ++predicate_stats.dispatched;
        draw_predicate=gpu_counters.next_predicate++;
        CounterDispatch(2,draw_predicate,(*tls_state).render_if_equal,(*tls_state).render_condition,0);
    }
    if(gpu_counters.predicate_values){const u32 slot=draw_predicate;const bool inverted=draw_inverted;const u64 at=(*tls_state).render_condition;const bool if_equal=(*tls_state).render_if_equal;
        AfterFlush([slot,inverted,at,if_equal]{const auto* o=static_cast<const u8*>(ResolveAddress(at,24));u64 a,b;std::memcpy(&a,o,8);std::memcpy(&b,o+16,8);
            const u32 cpu=(a==b)==if_equal;const u32 gpu=(gpu_counters.predicate_values[slot]!=0)!=inverted;static u32 n=0;
            if(cpu!=gpu&&n++<40)RT_LOG("predicate mismatch slot %u%s @%#llx: gpu %u cpu %u (%llu %llu eq %d)",slot,inverted?" inverted":"",(unsigned long long)at,gpu,cpu,(unsigned long long)a,(unsigned long long)b,(int)if_equal);});}
    return true;
}
struct DrawStats { u64 draws = 0, skipped = 0, clears = 0, pending = 0; } stats;

struct SampleCounter {
    VkQueryPool pool = VK_NULL_HANDLE;
    u32 next = 0;
    u64 value = 0;
    bool enabled = false;
    static constexpr u32 Capacity = 65536;
} samples;
u32 AllocateSampleQuery() {
    if(!samples.enabled)return ~0u;
    if(!samples.pool) {
        VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        info.queryType=VK_QUERY_TYPE_OCCLUSION;info.queryCount=SampleCounter::Capacity;
        VK_CHECK(vkCreateQueryPool(Vk().device,&info,nullptr,&samples.pool));
    }
    if(samples.next==SampleCounter::Capacity){Flush();samples.next=0;}
    return samples.next++;
}
u64 frame_number = 0;
// OOT_DRAW_CPU_PROFILE: sample host-side draw recording without changing command
// submission or synchronization. Timings are diagnostic, not a benchmark mode.
struct DrawCpuProfile {
    std::atomic<u64> draws{0}, total_ns{0}, lookup_ns{0}, descriptors_ns{0}, pass_ns{0}, fast_draws{0}, fast_ns{0};
    std::map<u64, std::pair<u64, u64>> shaders; // shader address -> host ns, draws (executor thread)
} draw_cpu_profile;
thread_local bool draw_was_fast = false; // (profiling: the draw being timed took the fast path)
bool DrawCpuProfiling() { static const bool on = std::getenv("OOT_DRAW_CPU_PROFILE") != nullptr; return on; }
void ReportDrawCpuProfile() {
    if (!DrawCpuProfiling()) return;
    const u64 draws = draw_cpu_profile.draws.exchange(0, std::memory_order_relaxed);
    const u64 total = draw_cpu_profile.total_ns.exchange(0, std::memory_order_relaxed);
    const u64 lookup = draw_cpu_profile.lookup_ns.exchange(0, std::memory_order_relaxed);
    const u64 descriptors = draw_cpu_profile.descriptors_ns.exchange(0, std::memory_order_relaxed);
    const u64 pass = draw_cpu_profile.pass_ns.exchange(0, std::memory_order_relaxed);
    {const u64 fd = draw_cpu_profile.fast_draws.exchange(0, std::memory_order_relaxed), fn = draw_cpu_profile.fast_ns.exchange(0, std::memory_order_relaxed);
     RT_LOG("draw CPU by path: fast %.0f draws/frame at %.2f us each (%.2f ms/frame); full %.0f at %.2f us each (%.2f ms/frame)", fd / 300.0, fd ? fn / 1000.0 / fd : 0.0, fn / 300e6,
            (draws - fd) / 300.0, draws > fd ? (total - fn) / 1000.0 / (draws - fd) : 0.0, (total - fn) / 300e6);}
    RT_LOG("draw CPU profile: %.2f draws/frame, %.2f ms/frame total (%.2f us/draw), geometry lookup %.2f, descriptors %.2f, pass recording %.2f, other %.2f ms/frame",
           draws / 300.0, total / 300e6, draws ? (double)total / draws / 1000.0 : 0.0,
           lookup / 300e6, descriptors / 300e6, pass / 300e6,
           total > lookup + descriptors + pass ? (total - lookup - descriptors - pass) / 300e6 : 0.0);
    std::vector<std::pair<u64, u64>> top;
    for (const auto& [shader, stat] : draw_cpu_profile.shaders) top.emplace_back(stat.first, shader);
    std::sort(top.rbegin(), top.rend());
    for (size_t i = 0; i < std::min<size_t>(top.size(), 8); ++i) {
        const auto [ns, shader] = top[i];
        const auto count = draw_cpu_profile.shaders[shader].second;
        RT_LOG("  draw CPU shader %#llx: %.3f ms/frame, %.1f draws/frame",
               (unsigned long long)shader, ns / 300e6, count / 300.0);
    }
    draw_cpu_profile.shaders.clear();
}
// OOT_DRAW_LOG=<frame>: logs every clear and draw of that presented frame.
// OOT_DRAW_LOG_TRIGGER=<path>: when that file appears, the next frame is logged (and the file removed).
s64 triggered_frame = -1;
bool LogThisFrame() {
    static const s64 frame = std::getenv("OOT_DRAW_LOG") ? std::atoll(std::getenv("OOT_DRAW_LOG")) : -1;
    return (s64)frame_number == frame || (s64)frame_number == triggered_frame;
}
constexpr u32 ShaderPublicStage[] = {0,1,2,3,4,5,6,7};
constexpr VkShaderStageFlagBits ShaderVkStage[] = {VK_SHADER_STAGE_VERTEX_BIT,VK_SHADER_STAGE_FRAGMENT_BIT,VK_SHADER_STAGE_GEOMETRY_BIT,
    VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,VK_SHADER_STAGE_COMPUTE_BIT,
    VK_SHADER_STAGE_TASK_BIT_EXT,VK_SHADER_STAGE_MESH_BIT_EXT};
// Uniform updates recorded but not yet executed, as the CPU must see them (shader specialization
// and handle evaluation read constants on the CPU). Each is dropped when its own batch retires:
// with batches in flight, later batches' updates must survive an earlier batch's retirement.
struct PendingUniformUpdate { u64 address; std::vector<u8> bytes; u64 batch; };
std::vector<PendingUniformUpdate> pending_uniform_updates;
u64 uniform_retirement_batch=~0ull; // open batch whose retirement action is registered
// Host pages -> indices of the pending updates covering them (in recording order).
std::unordered_map<u64,std::vector<u32>> uniform_update_pages;
void IndexUniformUpdate(u32 index){
    const auto& u=pending_uniform_updates[index];
    for(u64 page=u.address>>12;page<=(u.address+u.bytes.size()-1)>>12;++page)uniform_update_pages[page].push_back(index);
}
void RetireUniformUpdates(u64 batch){
    std::erase_if(pending_uniform_updates,[batch](const PendingUniformUpdate& u){return u.batch<=batch;});
    uniform_update_pages.clear();
    for(u32 i=0;i<pending_uniform_updates.size();++i)IndexUniformUpdate(i);
}
u32 ReadWord(u64 address) {
    const u64 cpu=reinterpret_cast<u64>(ResolveAddress(address,4));
    if(const auto found=uniform_update_pages.find(cpu>>12);found!=uniform_update_pages.end())
        for(auto it=found->second.rbegin();it!=found->second.rend();++it){const auto& u=pending_uniform_updates[*it];
            if(cpu>=u.address&&cpu-u.address+4<=u.bytes.size()){u32 word;std::memcpy(&word,u.bytes.data()+(cpu-u.address),4);return word;}}
    u32 word;std::memcpy(&word,reinterpret_cast<const void*>(cpu),4);return word;
}
u64 ReadPointer(void*,u64 address) {
    const auto range=LookupAddress(address);u64 value=0;
    if(range.pool&&range.available>=8)value=ReadWord(address)|(u64)ReadWord(address+4)<<32;
    return value;
}
// ---- Uniform update overlays (from Hookshot's ub_overlay, vk_replay.cpp) ----
// nvnCommandBufferUpdateUniformBuffer used to be a vkCmdUpdateBuffer between draws: it ended the
// render pass and drained the GPU (a full barrier) every time - ~40% of all pass breaks. Now the
// update is only queued; draws after it bind a copy of the uniform range (in the driver ring) with
// every pending update applied in order, and the batch writes the updates to memory once, before
// it is submitted. Draws before the update still read the unmodified memory, as on the console.
// OOT_UNIFORM_OVERLAY=0: the old path.
struct UnwrittenUniformUpdate { VkBuffer buffer; u64 offset; u64 cpu; std::vector<u8> bytes; };
std::vector<UnwrittenUniformUpdate> unwritten_uniform_updates;
std::unordered_map<u64,u32> unwritten_uniform_pages; // host page -> count of unwritten updates on it
u64 unwritten_lo=~0ull,unwritten_hi=0;
u64 uniform_overlays_made=0,uniform_updates_deferred=0;
std::map<std::pair<u64,u64>,u64> uniform_overlay_cache; // (range -> ring offset) valid until the next update or ring reset
bool UniformOverlayEnabled(){static const bool on=!std::getenv("OOT_UNIFORM_OVERLAY")||std::atoi(std::getenv("OOT_UNIFORM_OVERLAY"))!=0;return on;}
bool UnwrittenUniformOverlap(u64 cpu,u64 size){
    if(unwritten_uniform_updates.empty()||cpu>=unwritten_hi||cpu+size<=unwritten_lo)return false;
    for(u64 page=cpu>>12;page<=(cpu+size-1)>>12;++page)if(unwritten_uniform_pages.count(page))return true;
    return false;
}
// Writes the queued updates to memory now (one item: a pass break and barrier for all of them).
void WriteUnwrittenUniformUpdates(){
    if(unwritten_uniform_updates.empty())return;
    auto updates=std::move(unwritten_uniform_updates);
    unwritten_uniform_updates.clear();unwritten_uniform_pages.clear();unwritten_lo=~0ull;unwritten_hi=0;uniform_overlay_cache.clear();
    Immediate([&](VkCommandBuffer cmd){
        std::map<VkBuffer,std::map<u64,u64>> written;
        for(const auto& u:updates){
            const u64 end=u.offset+u.bytes.size();
            const auto overlaps=[&]{const auto& ranges=written[u.buffer];const auto next=ranges.lower_bound(u.offset);
                return (next!=ranges.end()&&next->first<end)||(next!=ranges.begin()&&std::prev(next)->second>u.offset);};
            if(overlaps()){
                VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                barrier.srcAccessMask=barrier.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
                vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
                written.clear();
            }
            vkCmdUpdateBuffer(cmd,u.buffer,u.offset,u.bytes.size(),u.bytes.data());
            written[u.buffer][u.offset]=end;
        }
    });
}
void WriteUniformUpdatesOverlapping(u64 cpu,u64 size){if(UnwrittenUniformOverlap(cpu,size))WriteUnwrittenUniformUpdates();}
// Hardware constant banks as NVN2 lays them out: the game's uniform binding N is bank N + 3;
// bank 2 is the driver's handle table (bound textures at 0x20 + 8i, bound images at 0x120 + 8i).
constexpr u32 UniformBankBase = 3;
// NVN2 push constants live in root table 0 (hardware bank 24), which lowered shaders read as
// bank 17 when the game binds nothing there.
constexpr u32 RootBank = 17;
// Bank 0 SSBO tables (16 entries of 16 bytes) per hardware stage: VS, TCS, TES, GS, FS at
// 0x110 + 0x100 * stage; the task and mesh tables sit 0x60 further (0x670, 0x770: what the game's
// task/mesh programs read; at 0x610/0x710 they found null buffer pointers and drew nothing).
constexpr u32 StorageTableBase = 0x110, StorageTableEnd = 0x770 + 0x100;
// Draw parameters the NVN driver keeps in bank 0 for graphics programs: base vertex at 0x640 and
// base instance at 0x644 (instanced text adds it to the instance ID to index its glyph data;
// with 0 every glyph of a draw starting at instance 1 read the wrong record: invisible labels).
constexpr u32 DrawBaseVertexOffset = 0x640, DrawBaseInstanceOffset = 0x644;
thread_local u32 draw_base_vertex = 0, draw_base_instance = 0;
constexpr u32 StorageTableOffset(u32 hw) { return 0x110 + hw * 0x100 + (hw >= 5 ? 0x60 : 0); }
// The hardware stage and byte within its table of a bank-0 offset (false outside every table).
inline bool StorageTableEntry(u32 offset, u32& hw, u32& within) {
    for (hw = 0; hw < 7; ++hw) { const u32 base = StorageTableOffset(hw); if (offset >= base && offset < base + 0x100) { within = offset - base; return true; } }
    return false;
}
u32 frame_render_width=0; // widest compute render size this frame (dynamic resolution)
u32 dispatch_grid[3]{1, 1, 1}; // workgroup counts of the current dispatch (compute bank 0 +0x4..+0xc)
thread_local bool synthesizing = false; // SynthesizeDriverBank probes whole banks
#ifdef OOT_SHADER_LONGSHOT
// The pointer table Longshot's translator reads at driver bank 0 + 0x880 (translate_program.cpp AddGlobalTables): three
// {address, size} entries naming the memory a task, mesh or compute program reaches through pointers it loads from its
// small storage buffers (bones, instances, the culling passes' lists), which no binding names. Here: the words of the
// stage's storage buffers of up to 256 bytes that are GPU addresses outside every bound buffer, and for each the whole
// pool (or mapped range) it lies in. Kept per thread for the bindings it was made from.
constexpr u32 PointerTableOffset = 0x880, PointerTableEntries = 3;
struct PointerRegion { u64 address = 0, size = 0; };
// What a program gets is the regions it has been seen to reach (up to three, the last used kept), not only those of
// this draw's buffers: the tables are few (some thirty in a scene, of a handful of pools), while the buffers they are
// read from are rewritten by GPU work every frame and often lie somewhere else each frame. Reading them per draw made
// every mesh draw wait for that work and sent it from the threads resolving draws back to the executor. So the buffers
// are read when nothing is writing them and the same bindings were not read in the last 120 frames; otherwise the
// draw takes what its program reached before.
const PointerRegion* PointerTable(const StageState& stage) {
    struct Program { u64 generation = ~0ull; PointerRegion regions[PointerTableEntries]; u64 used[PointerTableEntries]{}; };
    static std::shared_mutex mutex; static std::unordered_map<u64, std::pair<u64, u64>> read; // bindings -> (pool generation, frame)
    static std::unordered_map<u64, Program> programs; static u64 clock = 0;
    static thread_local PointerRegion result[PointerTableEntries];
    u64 key = 0xcbf29ce484222325ull; bool any = false;
    for (const Buffer& b : stage.storage) { key = (key ^ b.address) * 0x100000001b3ull; key = (key ^ b.size) * 0x100000001b3ull; any |= b.address && b.size && b.size <= 0x100; }
    const u64 program = stage.shader ? stage.shader->address : 0;
    key = (key ^ program) * 0x100000001b3ull;
    for (auto& r : result) r = {};
    if (!any) return result;
    const u64 generation = PoolGeneration();
    bool have = false, fresh = false, stale = false;
    { std::shared_lock lock{mutex};
      if (const auto it = programs.find(program); it != programs.end() && it->second.generation == generation) { have = true; std::memcpy(result, it->second.regions, sizeof result); } // (known, be it that it reaches nothing)
      else if (it != programs.end()) stale = true;
      if (const auto it = read.find(key); it != read.end() && it->second.first == generation) fresh = frame_number - it->second.second < 120; }
    if (stale) {
        // Pools or mappings changed since: what the program reached is kept where it still is one region of memory
        // (it was dropped, and for the frames until its buffers were read again the draws resolved ahead read nothing
        // through their pointers: the pass that shades a model drew it elsewhere than its depth pass, a white flash).
        std::unique_lock lock{mutex};
        Program& p = programs[program];
        if (p.generation != generation) {
            for (u32 i = 0; i < PointerTableEntries; ++i) { if (!p.regions[i].size) continue;
                u64 start = 0, end = 0;
                if (!GuestRegionAround(p.regions[i].address, &start, &end) || start > p.regions[i].address || end < p.regions[i].address + p.regions[i].size) { p.regions[i] = {}; p.used[i] = 0; } }
            for (u32 i = 0; i < PointerTableEntries; ++i) if (!p.regions[i].size) for (u32 j = i + 1; j < PointerTableEntries; ++j) if (p.regions[j].size) { std::swap(p.regions[i], p.regions[j]); std::swap(p.used[i], p.used[j]); break; }
            p.generation = generation;
        }
        have = true; std::memcpy(result, p.regions, sizeof result);
    }
    { static const int test = std::getenv("OOT_PT_TEST") ? std::atoi(std::getenv("OOT_PT_TEST")) : 0; // (bisection: 1 such draws are the executor's, 2 no table at all)
      if (test == 1 && resolve_worker) { RESOLVE_BAIL(); return result; }
      const u64 index = (u64)(&stage - &(*tls_state).stages[0]);
      if (test == 3 && resolve_worker && (index == 6 || index == 7)) { RESOLVE_BAIL(); return result; } // (mesh draws only)
      if (test == 4 && resolve_worker && index != 6 && index != 7) { RESOLVE_BAIL(); return result; }   // (the others only)
      if (test == 5 && resolve_worker && index == 6) { RESOLVE_BAIL(); return result; }
      if (test == 6 && resolve_worker && index == 7) { RESOLVE_BAIL(); return result; }
      if (test == 2) { for (auto& r : result) r = {}; return result; } }
    if (fresh || (have && resolve_worker)) return result;
    bool pending = false;
    for (const Buffer& b : stage.storage) if (b.address && b.size && b.size <= 0x100) pending |= HasPendingWrite(b.address, b.size);
    if (pending && have) return result;
    if (pending && resolve_worker) { RESOLVE_BAIL(); return result; } // (nothing known of this program yet: the executor reads, it may wait for the work)
    PointerRegion found[PointerTableEntries]; u32 regions = 0;
    const u64 reach = std::min<u64>(Vk().properties.limits.maxStorageBufferRange, 256ull << 20) & ~0xffull;
    for (const Buffer& b : stage.storage) {
        if (!b.address || !b.size || b.size > 0x100) continue;
        const auto range = LookupAddress(b.address);
        if (!range.pool || range.available < b.size) continue;
        for (u64 at = 0; at + 8 <= b.size && regions < PointerTableEntries; at += 8) {
            const u64 v = ReadWord(b.address + at) | (u64)ReadWord(b.address + at + 4) << 32;
            if ((v & 3) || v < (1ull << 32) || v >= (1ull << 40)) continue;
            bool bound = false;
            for (const Buffer& o : stage.storage) bound |= o.address && v >= o.address && v < o.address + o.size;
            for (u32 i = 0; i < regions; ++i) bound |= v >= found[i].address && v < found[i].address + found[i].size;
            u64 start = 0, end = 0;
            if (bound || !GuestRegionAround(v, &start, &end)) continue;
            // (a pool larger than a binding reaches: a window of that size with the pointer in it)
            if (end - start > reach) { start = std::max<u64>(start, (v & ~0xffffull) - std::min<u64>(v & ~0xffffull, reach / 2)); end = std::min(end, start + reach); }
            found[regions++] = {start, end - start};
        }
    }
    { static const bool trace = std::getenv("OOT_POINTER_TABLE_TRACE") != nullptr; // (diagnostic: the distinct readings, by stage)
      if (trace) { static std::mutex m; static std::map<std::array<u64, 7>, u64> seen; std::lock_guard lock{m};
        const u64 st = (u64)(&stage - &(*tls_state).stages[0]);
        std::array<u64, 7> t{st, found[0].address, found[0].size, found[1].address, found[1].size, found[2].address, found[2].size};
        if (++seen[t] == 1 && seen.size() < 200) RT_LOG("pointer table stage %llu: %#llx+%#llx %#llx+%#llx %#llx+%#llx", (unsigned long long)st, (unsigned long long)t[1], (unsigned long long)t[2],
            (unsigned long long)t[3], (unsigned long long)t[4], (unsigned long long)t[5], (unsigned long long)t[6]); } }
    std::unique_lock lock{mutex};
    if (read.size() > 65536) read.clear();
    read[key] = {generation, frame_number};
    Program& p = programs[program];
    if (p.generation == ~0ull) p.generation = generation; // (a new one; an older one was looked over above)
    for (u32 i = 0; i < regions; ++i) {
        u32 slot = PointerTableEntries, oldest = 0;
        for (u32 j = 0; j < PointerTableEntries; ++j) {
            if (p.regions[j].address == found[i].address && p.regions[j].size == found[i].size) { slot = j; break; }
            if (p.used[j] < p.used[oldest]) oldest = j;
        }
        if (slot == PointerTableEntries) { slot = oldest;
            if (p.regions[slot].size) { static std::atomic<u32> shown{0}; if (shown++ < 20) RT_LOG("pointer table of program %#llx is full: %#llx+%#llx gives way to %#llx+%#llx", (unsigned long long)program,
                (unsigned long long)p.regions[slot].address, (unsigned long long)p.regions[slot].size, (unsigned long long)found[i].address, (unsigned long long)found[i].size); }
            p.regions[slot] = found[i]; }
        p.used[slot] = ++clock;
    }
    // (filled slots first: the search takes the first region that holds an address, and a bank that reads the same is reused)
    for (u32 i = 0; i < PointerTableEntries; ++i) if (!p.regions[i].size) for (u32 j = i + 1; j < PointerTableEntries; ++j) if (p.regions[j].size) { std::swap(p.regions[i], p.regions[j]); std::swap(p.used[i], p.used[j]); break; }
    std::memcpy(result, p.regions, sizeof result);
    return result;
}
#endif
u32 Cbuf(void* context, u32 bank, u32 offset) {
    auto& stage = *static_cast<StageState*>(context);
#ifdef OOT_SHADER_LONGSHOT
    if (bank == 0 && offset >= PointerTableOffset && offset < PointerTableOffset + 16 * PointerTableEntries) {
        const PointerRegion& e = PointerTable(stage)[(offset - PointerTableOffset) / 16];
        switch (offset & 0xf) { case 0: return (u32)e.address; case 4: return (u32)(e.address >> 32); case 8: return (u32)e.size; default: return 0; }
    }
#endif
    static const bool trace = std::getenv("OOT_CBUF_TRACE") != nullptr;
    if (trace) RT_LOG("cbuf read bank %u offset %#x", bank, offset);
    // Compute bank 0 is packed per program (counts in control +0xa9/+0xb1/+0xb2/+0xb3/+0xb6): from
    // 0x10, the push constants (+0xa9 words, 16-byte aligned), A uniform-buffer entries and B storage-buffer entries of 16 bytes {address, size}, then C
    // texture handles and D image handles of 8 bytes, each for bindings 0.. in order.
    if (bank == 0 && &stage == &(*tls_state).stages[5]) {
        static const std::vector<u8> no_control; // (a temporary here copied the program's control section at every read)
        const std::vector<u8>& control = stage.shader ? stage.shader->control : no_control;
        const auto count = [&](u32 at) -> u32 { return at < control.size() ? control[at] : 0; };
        const u32 a = count(0xb1), b = count(0xb2), t = count(0xb3), im = count(0xb6), push = count(0xa9) * 4;
        const auto entry = [&](const Buffer& buf, u32 word) -> u32 { return word == 0 ? (u32)buf.address : word == 1 ? (u32)(buf.address >> 32) : word == 2 ? (u32)buf.size : 0; };
        const auto half = [&](u64 v) { return (u32)(offset & 4 ? v >> 32 : v); };
        // 0x4/0x8/0xc: the dispatch's workgroup counts (shaders flatten ctaid with them).
        if (offset < 0x10) return offset == 0 ? 0 : dispatch_grid[offset / 4 - 1];
        u32 base = 0x10;
        // Push constants (root table 0, written by PushComputeConstants) come first: +0xa9 words.
        if (offset >= base && offset < base + push) return offset - base < 0x100 ? (*tls_state).root[1][0][(offset - base) / 4] : 0;
        base += (push + 15) & ~15u;
        if (offset >= base && offset < base + 16 * a) return entry(stage.uniforms.at((offset - base) / 16), (offset - base) % 16 / 4);
        base += 16 * a;
        if (offset >= base && offset < base + 16 * b) return entry(stage.storage.at((offset - base) / 16), (offset - base) % 16 / 4);
        base += 16 * b;
        if (offset >= base && offset < base + 8 * t) return half(stage.textures.at((offset - base) / 8));
        base += 8 * t;
        if (offset >= base && offset < base + 8 * im) return half(stage.images.at((offset - base) / 8));
        static std::map<u32, bool> reported;
        if (!synthesizing && !reported[offset]) { reported[offset] = true; RT_LOG("compute shader reads driver bank 0 offset %#x outside its layout (returned 0)", offset); }
        return 0;
    }
    if(bank==1){
        if(!stage.shader||offset+4>stage.shader->constants.size()){
            static std::map<std::pair<u64,u32>,bool> reported;const u64 at=stage.shader?stage.shader->address:0;
            if(!synthesizing&&!resolve_worker&&!reported[{at,offset}]){reported[{at,offset}]=true;RT_LOG("shader %#llx reads literal bank 1 at %#x beyond its %zu constant bytes (returned 0)",(unsigned long long)at,offset,stage.shader?stage.shader->constants.size():(size_t)0);}
            return 0;}
        u32 value;std::memcpy(&value,stage.shader->constants.data()+offset,4);return value;
    }
    if (bank == 2) {
        const auto half = [&](u64 v) { return (u32)(offset & 4 ? v >> 32 : v); };
        if (offset >= 0x20 && offset < 0x120) return half(stage.textures.at((offset - 0x20) / 8));
        if (offset >= 0x120 && offset < 0x220) return half(stage.images.at((offset - 0x120) / 8));
    }
    // Bank 0 is one driver buffer shared by all stages; it holds a 16-entry SSBO table per
    // hardware stage (VS, TCS, TES, GS, FS, task, mesh) at 0x110 + 0x100 * hardware stage.
    u32 table_hw = 0, table_within = 0;
    if (bank == 0 && StorageTableEntry(offset, table_hw, table_within)) {
        constexpr u32 HardwareToPublic[] = {0, 3, 4, 2, 1, 6, 7};
        const bool compute = &stage == &(*tls_state).stages[5];
        if (compute && table_hw != 0) return 0;
        auto& owner = compute ? stage : (*tls_state).stages[HardwareToPublic[table_hw]];
        const auto& b = owner.storage.at(table_within / 0x10);
        switch (offset & 0xf) {
        case 0: return (u32)b.address;
        case 4: return (u32)(b.address >> 32);
        case 8: return (u32)b.size;
        default: return 0;
        }
    }
    if (bank == RootBank && !stage.uniforms[RootBank - UniformBankBase].address) {
        if (offset >= 0x100) return 0;
        return (*tls_state).root[&stage == &(*tls_state).stages[5]][0][offset / 4];
    }
    if (bank < UniformBankBase) {
        if (resolve_worker) return 0;
        static std::map<std::pair<u32, u32>, bool> reported;
        if (!reported[{bank, offset}]) {
            reported[{bank, offset}] = true;
            RT_LOG("shader reads driver constant bank %u offset %#x (returned 0)", bank, offset);
        }
        return 0;
    }
    const u32 index = bank - UniformBankBase;
    if (index >= stage.uniforms.size() || !stage.uniforms[index].address || offset + 4 > stage.uniforms[index].size)
        Fatal("shader reads unbound uniform buffer %u (bank %u) +%#x", index, bank, offset);
    return ReadWord(stage.uniforms[index].address + offset);
}
// Driver constant banks (0: SSBO table at 0x110 + 16i; 2: texture/image handles) as GPU-visible
// bytes, for shaders that read them at run time.
constexpr u32 DriverBankSize = 0x1000;
// Bytes of a driver bank that carry data (the rest reads as zero: bound ranges end there, and
// robustBufferAccess2 returns zero beyond them).
u32 DriverBankUsed(const StageState& stage, u32 bank) {
    if (bank == RootBank) return 0x100;
    if (bank == 1) return stage.shader ? std::max<u32>(16, (u32)(stage.shader->constants.size() + 15) & ~15u) : 16;
#ifdef OOT_SHADER_LONGSHOT
    if (bank == 0) return PointerTableOffset + 16 * PointerTableEntries; // (the pointer table after the storage tables)
#endif
    if (bank == 0 && &stage == &(*tls_state).stages[5]) return 0x400;
    if (bank == 0) return StorageTableEnd;
    if (bank == 2) return 0x220;
    return DriverBankSize;
}
void SynthesizeDriverBank(StageState& stage, u32 bank, u8* out) {
    std::memset(out, 0, DriverBankUsed(stage, bank));
#ifdef OOT_SHADER_LONGSHOT
    if (bank == 0) { // (the stage's pointer table: its programs read these words from the bank itself)
        const PointerRegion* e = PointerTable(stage);
        for (u32 i = 0; i < PointerTableEntries; ++i) { const u32 words[4] = {(u32)e[i].address, (u32)(e[i].address >> 32), (u32)e[i].size, 0};
            std::memcpy(out + PointerTableOffset + 16 * i, words, 16); }
    }
#endif
    if (bank == 0 && &stage != &(*tls_state).stages[5]) { // the shared SSBO tables, written directly
        constexpr u32 HardwareToPublic[] = {0, 3, 4, 2, 1, 6, 7};
        for (u32 hw = 0; hw < 7; ++hw) {
            const auto& owner = (*tls_state).stages[HardwareToPublic[hw]];
            for (u32 i = 0; i < 16; ++i) {
                const Buffer& b = owner.storage[i];
                if (!b.address) continue;
                const u32 words[2] = {(u32)b.address, (u32)(b.address >> 32)};
                u8* at = out + StorageTableOffset(hw) + i * 0x10;
                std::memcpy(at, words, 8); const u32 size = (u32)b.size; std::memcpy(at + 8, &size, 4);
            }
        }
        std::memcpy(out + DrawBaseVertexOffset, &draw_base_vertex, 4);
        std::memcpy(out + DrawBaseInstanceOffset, &draw_base_instance, 4);
        return;
    }
    if (bank == 2) { // texture and image handles
        std::memcpy(out + 0x20, stage.textures.data(), 0x100);
        std::memcpy(out + 0x120, stage.images.data(), 0x100);
        return;
    }
    if(bank==RootBank){std::memcpy(out,(*tls_state).root[&stage==&(*tls_state).stages[5]][0],0x100);return;}
    if(bank==1){
        if(!stage.shader||stage.shader->constants.size()>DriverBankSize)Fatal("shader literal bank exceeds driver ring allocation");
        std::memcpy(out,stage.shader->constants.data(),stage.shader->constants.size());return;
    }
    u32 first = 0, last = DriverBankSize; // the only offsets synthesized below
    if (bank == 0 && &stage == &(*tls_state).stages[5]) last = 0x400;
    else if (bank == 0) { first = StorageTableBase; last = StorageTableEnd; }
    else if (bank == 2) { first = 0x20; last = 0x220; }
    for (u32 offset = first; offset < last; offset += 4) {
        const bool known = (bank == 0 && &stage == &(*tls_state).stages[5] && offset < 0x400) || (bank == 0 && &stage != &(*tls_state).stages[5] && offset >= StorageTableBase && offset < StorageTableEnd) || (bank == 2 && offset >= 0x20 && offset < 0x220);
        if (!known) continue;
        synthesizing = true;
        const u32 v = Cbuf(&stage, bank, offset);
        synthesizing = false;
        std::memcpy(out + offset, &v, 4);
    }
}
// A host-visible ring for per-draw driver banks (draws complete before the ring wraps around).
struct DriverRing {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    u8* mapped = nullptr;
    u64 size = 64u << 20, head = 0; // (driver banks and uniform overlays)
} ring;
u64 ring_generation=0; // (the ring started over: offsets handed out before are not to be used)
std::mutex ring_mutex; // (allocations in the ring and its tables: draws are resolved on several threads)
std::unordered_map<u64, u64> ring_banks; // bank content hash -> ring offset (this ring cycle)
std::vector<u8> ring_shadow;              // CPU copy of the deduplicated banks
u64 compute_bank0_offset = 0; // ring offset of the last synthesized compute driver bank
VkDescriptorBufferInfo DriverBankDescriptor(StageState& stage, u32 bank) {
    auto& vk = Vk();
    if (!ring.buffer) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = ring.size;
        bi.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VK_CHECK(TracedCreateBuffer(&bi,&ring.buffer,__FILE__,__LINE__));
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(vk.device, ring.buffer, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = FindMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK(vkAllocateMemory(vk.device, &ai, nullptr, &ring.memory));
        VK_CHECK(vkBindBufferMemory(vk.device, ring.buffer, ring.memory, 0));
        VK_CHECK(vkMapMemory(vk.device, ring.memory, 0, ring.size, 0, reinterpret_cast<void**>(&ring.mapped)));
        ring_shadow.resize(ring.size);
    }
    // Identical banks (most consecutive draws share them) reuse the ring copy written earlier in
    // this ring cycle; the ring then wraps (a full flush) far less often.
    alignas(16) static thread_local u8 bank_data[DriverBankSize];
    const u32 used = DriverBankUsed(stage, bank);
    SynthesizeDriverBank(stage, bank, bank_data);
    const bool compute_bank0 = bank == 0 && &stage == &(*tls_state).stages[5]; // patched by indirect dispatch
    u64 hash = (0xcbf29ce484222325ull ^ bank) * 0x100000001b3ull ^ used;
    const u64 align = std::max<u64>(vk.properties.limits.minUniformBufferOffsetAlignment, 256);
    if (!compute_bank0) {
        // Each thread writes its banks into a piece of the ring of its own (draws are resolved on several threads:
        // one lock around every bank was where they queued), and knows the banks it wrote there.
        struct LocalRing { u64 generation = ~0ull, base = 0, used = 0, size = 0; std::unordered_map<u64, u64> banks; };
        static thread_local LocalRing local;
        constexpr u64 Piece = 256u << 10;
        if (local.generation != ring_generation) { local.banks.clear(); local.used = local.size = 0; local.generation = ring_generation; }
        // The bank this thread made last for the same stage and bank number is the likeliest match (consecutive draws
        // mostly share their banks): compared directly, before anything is hashed. OOT_BANK_MEMO=1 turns it on.
        struct Last { u64 generation = ~0ull, at = 0; u32 used = 0; const State* state = nullptr; std::vector<u8> data; };
        static thread_local Last last[9][4];
        static const bool memo = std::getenv("OOT_BANK_MEMO") != nullptr; // (off unless asked for: nothing measured)
        const u32 stage_index = (u32)(&stage - &(*tls_state).stages[0]);
        Last* const mine = memo && stage_index < 9 && bank < 4 ? &last[stage_index][bank] : nullptr;
        if (mine && mine->generation == ring_generation && mine->used == used && std::memcmp(mine->data.data(), bank_data, used) == 0)
            return {ring.buffer, mine->at, used};
        const auto remember = [&](u64 at) { if (mine) { mine->generation = ring_generation; mine->at = at; mine->used = used; mine->data.assign(bank_data, bank_data + used); } };
        const u64* words = reinterpret_cast<const u64*>(bank_data);
        for (u32 i = 0; i < used / 8; ++i) hash = (hash ^ words[i]) * 0x100000001b3ull;
        const auto found = local.banks.find(hash);
        // (Compared against a CPU shadow: reading the host-visible ring itself is very slow.)
        if (found != local.banks.end() && std::memcmp(ring_shadow.data() + found->second, bank_data, used) == 0) {
            remember(found->second); return {ring.buffer, found->second, used}; }
        local.used = (local.used + align - 1) & ~(align - 1);
        if (local.used + DriverBankSize > local.size) {
            std::lock_guard ring_lock{ring_mutex};
            ring.head = (ring.head + align - 1) & ~(align - 1);
            if (ring.head + Piece > ring.size) Fatal("driver constant ring exhausted within one draw");
            local.base = ring.head; local.used = 0; local.size = Piece; ring.head += Piece;
        }
        const u64 at = local.base + local.used;
        std::memcpy(ring.mapped + at, bank_data, used);
        std::memcpy(ring_shadow.data() + at, bank_data, used);
        local.banks[hash] = at; local.used += used;
        remember(at);
        return {ring.buffer, at, used};
    }
    std::lock_guard ring_lock{ring_mutex};
    ring.head = (ring.head + align - 1) & ~(align - 1);
    if (ring.head + DriverBankSize > ring.size) Fatal("driver constant ring exhausted within one draw");
    std::memcpy(ring.mapped + ring.head, bank_data, used);
    compute_bank0_offset = ring.head;
    const VkDescriptorBufferInfo info{ring.buffer, ring.head, used};
    ring.head += used;
    return info;
}
// A ring copy of a uniform range with the pending updates applied (Buffer() when none overlaps).
VkDescriptorBufferInfo UniformOverlay(u64 cpu,u64 bytes){
    static const bool snapshot_all=std::getenv("OOT_UNIFORM_SNAPSHOT")!=nullptr; // (experiment: every uniform binding as memory holds it when the draw is translated)
    if(!UniformOverlayEnabled()||(!snapshot_all&&!UnwrittenUniformOverlap(cpu,bytes)))return {};
    auto& vk=Vk();
    if(!ring.buffer||bytes>(1u<<20))return {};
    std::lock_guard ring_lock{ring_mutex};
    if(const auto c=uniform_overlay_cache.find({cpu,bytes});!snapshot_all&&c!=uniform_overlay_cache.end())return {ring.buffer,c->second,bytes};
    const u64 align=std::max<u64>(vk.properties.limits.minUniformBufferOffsetAlignment,256);
    const u64 at=(ring.head+align-1)&~(align-1);
    if(at+bytes>ring.size)return {};
    u8* out=ring.mapped+at;
    std::memcpy(out,reinterpret_cast<const void*>(cpu),bytes);
    // Every pending update (earlier batches' too: memory may not have them yet), in order.
    std::vector<u32> hits;
    for(u64 page=cpu>>12;page<=(cpu+bytes-1)>>12;++page)
        if(const auto found=uniform_update_pages.find(page);found!=uniform_update_pages.end())hits.insert(hits.end(),found->second.begin(),found->second.end());
    std::sort(hits.begin(),hits.end());hits.erase(std::unique(hits.begin(),hits.end()),hits.end());
    for(u32 i:hits){const auto& u=pending_uniform_updates[i];
        const u64 lo=std::max(cpu,u.address),hi=std::min(cpu+bytes,u.address+u.bytes.size());
        if(lo<hi)std::memcpy(out+(lo-cpu),u.bytes.data()+(lo-u.address),hi-lo);}
    ring.head=at+bytes;++uniform_overlays_made;uniform_overlay_cache[{cpu,bytes}]=at;
    return {ring.buffer,at,bytes};
}
bool HandleRegistered(u32 handle) {
    const u32 id = handle & 0xfffff;
    return (*tls_state).textures && id < (*tls_state).textures->entries.size() && (*tls_state).textures->entries[id].host;
}
bool SamplerRegistered(u32 handle) {
    const u32 id=handle>>20;
    return (*tls_state).samplers&&id<(*tls_state).samplers->entries.size()&&(*tls_state).samplers->entries[id].valid;
}
TextureBinding& Binding(u32 handle) {
    const u32 id = handle & 0xfffff;
    if (!(*tls_state).textures || id >= (*tls_state).textures->entries.size() || !(*tls_state).textures->entries[id].host) {
        for (u32 s = 0; s < 2; ++s) {
            RT_LOG("stage %u bound textures: %#llx %#llx %#llx %#llx", s, (unsigned long long)(*tls_state).stages[s].textures[0],
                   (unsigned long long)(*tls_state).stages[s].textures[1], (unsigned long long)(*tls_state).stages[s].textures[2], (unsigned long long)(*tls_state).stages[s].textures[3]);
            const auto& u = (*tls_state).stages[s].uniforms[2];
            if (u.address) {
                const u32* w = static_cast<const u32*>(ResolveAddress(u.address, 0x40));
                RT_LOG("stage %u bank 2 words: %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x", s, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10], w[11]);
            }
        }
        Fatal("unregistered shader texture handle %#x", handle);
    }
    return (*tls_state).textures->entries[id];
}
u32 TextureType(void*, u32 handle) {
    if (!HandleRegistered(handle)) return 2; // reported (and the work skipped) at descriptor time
    const auto& binding = Binding(handle); const auto* t = binding.host.get();
    const auto target = binding.view.target ? (Target)(binding.view.target-1) : t->target;
    switch (target) {
    case Target::T1D: return 0; case Target::T1DArray: return 1;
    case Target::T2D: case Target::T2DMultisample: return 2;
    case Target::T2DArray: case Target::T2DMultisampleArray: return 3;
    case Target::T3D: return 4; case Target::Cube: return 5; case Target::CubeArray: return 6;
    case Target::Buffer: return 7; case Target::Rectangle: return 8;
    } Fatal("unknown texture type");
}
// The view dimension the shader declares (Shader::TextureType: 1D, 1D array, 2D, 2D array, 3D,
// cube, cube array, buffer, rectangle); views must match it exactly.
u32 ShaderViewTarget(u32 shader_type, const Texture& t) {
    const bool ms = t.samples > 1;
    switch (shader_type) {
    case 0: return (u32)Target::T1D; case 1: return (u32)Target::T1DArray;
    case 2: case 8: return (u32)(ms ? Target::T2DMultisample : Target::T2D);
    case 3: return (u32)(ms ? Target::T2DMultisampleArray : Target::T2DArray);
    case 4: return (u32)Target::T3D; case 5: return (u32)Target::Cube; case 6: return (u32)Target::CubeArray;
    default: return (u32)t.target;
    }
}
u32 TextureFormat(void*, u32 handle) { if (!HandleRegistered(handle)) return 37; auto& b = Binding(handle); return b.view.format ? b.view.format : b.host.get()->format; }
VkSampler Sampler(u32 handle) {
    const u32 id = handle >> 20;
    if (!(*tls_state).samplers || id >= (*tls_state).samplers->entries.size() || !(*tls_state).samplers->entries[id].valid) {
        if (resolve_worker) { RESOLVE_BAIL(); return VK_NULL_HANDLE; }
        // (the movie player samples its frames with sampler 0 without registering one: a linear clamp sampler,
        // as an unset sampler descriptor reads on the console)
        static VkSampler fallback = VK_NULL_HANDLE;
        if (!fallback) { VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; si.magFilter = si.minFilter = VK_FILTER_LINEAR;
            si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; si.maxLod = VK_LOD_CLAMP_NONE;
            VK_CHECK(vkCreateSampler(Vk().device, &si, nullptr, &fallback)); }
        static std::set<u32> told; if (told.insert(id).second && told.size() < 8) RT_LOG("sampler %u is not registered: a linear clamp sampler stands in", id);
        return fallback;
    }
    auto& e = (*tls_state).samplers->entries[id]; if (e.sampler && e.settings_generation == gfx::Generation()) return e.sampler;
    if (resolve_worker) { RESOLVE_BAIL(); return VK_NULL_HANDLE; } // (made by the executor)
    if (e.sampler) { const VkSampler old = e.sampler; AfterFlush([old] { vkDestroySampler(Vk().device, old, nullptr); }); e.sampler = VK_NULL_HANDLE; }
    u32 packed; std::memcpy(&packed, e.state + 0x2c, 4);
    const auto f = [&](u32 offset) { float v; std::memcpy(&v,e.state+offset,4); return v; };
    const u32 min = packed >> 16 & 15, mag = packed >> 20 & 15;
    VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    // NVN: nearest=0, linear=1, nearest/linear mipmap nearest=2/3, nearest/linear mipmap linear=4/5.
    info.minFilter = min & 1 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST; info.magFilter = mag & 1 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    info.mipmapMode = min >= 4 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    // NVNwrapMode: CLAMP, REPEAT, MIRROR_CLAMP, MIRROR_CLAMP_TO_EDGE, MIRROR_CLAMP_TO_BORDER,
    // CLAMP_TO_BORDER, MIRRORED_REPEAT, CLAMP_TO_EDGE (the API enum the builder stores).
    const VkSamplerAddressMode wraps[] = {VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,VK_SAMPLER_ADDRESS_MODE_REPEAT,
        VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
        VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
    const auto wrap = [&](u32 shift) { const u32 value = packed >> shift & 15; if (value >= std::size(wraps)) Fatal("unknown sampler wrap %u",value); return wraps[value]; };
    info.addressModeU = wrap(0); info.addressModeV = wrap(4); info.addressModeW = wrap(8);
    info.minLod = std::max(f(8),0.f); info.maxLod = min < 2 ? info.minLod : std::max(f(12),info.minLod);
    if (std::getenv("OOT_DEBUG_LOD0")) info.maxLod = info.minLod = 0;
    info.mipLodBias = std::clamp(f(16),-Vk().properties.limits.maxSamplerLodBias,Vk().properties.limits.maxSamplerLodBias);
    info.anisotropyEnable = f(20) > 1; info.maxAnisotropy = std::clamp(f(20),1.f,Vk().properties.limits.maxSamplerAnisotropy);
    info.compareEnable = (packed >> 24 & 3) != 0; const u32 compare = packed >> 26;
    info.compareOp = (VkCompareOp)(compare ? compare - 1 : 0);
    const float r=f(28),g=f(32),b=f(36),a=f(40);
    info.borderColor = r==1 && g==1 && b==1 ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE : a==1 ? VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    // The Graphics screen's texture filtering and detail (gfx): only filtered, mipmapped colour
    // samplers (min filter with mipmaps, no depth compare) - lookups, UI and shadows stay as the game set them.
    if (!info.compareEnable && min >= 2 && (min & 1)) {
        if (gfx::ForceBilinear()) { info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST; info.anisotropyEnable = VK_FALSE; info.maxAnisotropy = 1; }
        if (gfx::ForceTrilinear()) info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        if (const int a = gfx::Anisotropy(); a > 1) { info.anisotropyEnable = VK_TRUE; info.maxAnisotropy = std::min((float)a, Vk().properties.limits.maxSamplerAnisotropy); }
        info.mipLodBias = std::clamp(info.mipLodBias + gfx::LodBias(), -Vk().properties.limits.maxSamplerLodBias, Vk().properties.limits.maxSamplerLodBias);
    }
    e.settings_generation = gfx::Generation();
    VK_CHECK(vkCreateSampler(Vk().device,&info,nullptr,&e.sampler)); return e.sampler;
}
// SET_MESH_SHADER_A/B and SET_MESH_INIT_SHADER from the NVN2 control header: +0x58 threads
// (15:0) and task payload bytes (31:16); +0x5c output topology (31:24, 3 = triangles); +0x60
// max vertices (31:16) and max primitives (15:0).
std::array<u32,3> MeshRegisters(const ShaderCode& code){
    const auto word=[&](u32 offset){u32 v=0;if(offset+4<=code.control.size())std::memcpy(&v,code.control.data()+offset,4);return v;};
    const u32 c58=word(0x58),c5c=word(0x5c),c60=word(0x60);
    // The control header stores bytes, whereas SET_MESH_INIT_SHADER stores 128-byte
    // lines. Reading only the high byte halved the fog task's 256-byte payload:
    // its final meshlet records aliased the last word and generated tiled strips.
    const u32 threads=std::min(c58&0xffff,0x3fu),lines=((c58>>16)+127)/128,topology=c5c>>24;
    // (Hookshot, mesh_registers_of: NVN's topology 3 (triangles) is the register's 4; points and lines go as they are.
    // Lines were a Fatal here: the shader compiler emits them, or the program is skipped as untranslatable.)
    if(code.stage==7&&topology!=3){static std::set<u64> told;if(told.insert(code.address).second)RT_LOG("mesh shader %#llx outputs topology %u (not triangles)",(unsigned long long)code.address,topology);}
#ifdef OOT_SHADER_LONGSHOT
    const u32 topology_register=topology==3?4u:topology==1?0u:(topology&0xfu); // (the control section's 1 is points: Longshot issue 302)
#else
    const u32 topology_register=topology==3?4u:(topology&0xfu);
#endif
    // Control +0x60: maximum vertices in the low half, maximum primitives in the high half (the
    // game's meshlets: 64 vertices / 126 triangles, grass 128 / 64). SET_MESH_SHADER_A holds
    // vertices at bits 4-12 and primitives at bits 13-22; swapping them clamped real meshlets
    // (rock depth passes drew garbage triangles: black shards).
    // (Hookshot, vk_shader.cpp mesh_registers_of: the pair is not always vertices and primitives in
    // that order - the Kokiri door's says 126/64 while its code writes 64 vertices and up to 128
    // primitives; held to 64, the rest of each meshlet was never output (holes, white door glow).
    // Both limits take the larger value. OOT_MESH_EXACT_LIMITS=1: the old split.)
    static const bool exact=std::getenv("OOT_MESH_EXACT_LIMITS")!=nullptr;
    static const u32 round_to=std::getenv("OOT_MESH_ROUND")?(u32)std::atoi(std::getenv("OOT_MESH_ROUND")):0; // (experiment: limits rounded up to a multiple)
    u32 most=std::min(std::max(c60&0xffff,c60>>16),0x100u);if(round_to)most=std::min((most+round_to-1)/round_to*round_to,0x100u);
    const u32 a=exact?topology_register|(std::min(c60&0xffff,0x1ffu)<<4)|(std::min(c60>>16,0x3ffu)<<13):topology_register|(most<<4)|(most<<13);
    return {a,threads<<20,threads|(lines<<12)|(lines<<22)};
}
// What a compiled shader assumes about a texture handle: its type and whether it is integer.
u32 DescribeTexture(void*,u32 handle){
    if(!HandleRegistered(handle))return ~0u;
    const u32 f=TextureFormat(nullptr,handle);
    static const u32 integer[]={3,4,8,9,11,12,15,16,20,21,23,24,27,28,32,33,35,36,39,40,44,45,47,48,62,106,111,112,116,117,119,120};
    return TextureType(nullptr,handle)|(std::find(std::begin(integer),std::end(integer),f)!=std::end(integer)?1u<<16:0);
}
// Which constant words a compiled variant depends on (0 value, 1 texture handle, 2 none):
// buffer addresses and push constants change per draw without changing the code.
u32 ClassifyConstant(void* context,u32 bank,u32 offset){
    auto& stage=*static_cast<StageState*>(context);
    if(bank==2)return 1;
    if(bank==RootBank)return 2;
    if(bank==0&&&stage==&(*tls_state).stages[5]){
        if(offset>=4&&offset<0x10&&std::getenv("OOT_GRID_TRACE")){static std::map<u64,bool> seen;const u64 a=stage.shader?stage.shader->address:0;if(!seen[a]){seen[a]=true;RT_LOG("compute shader %#llx compiled with grid word +%#x = %u",(unsigned long long)a,offset,Cbuf(&stage,0,offset));}}
        const auto& control=stage.shader?stage.shader->control:std::vector<u8>{};
        const auto count=[&](u32 at)->u32{return at<control.size()?control[at]:0;};
        const u32 push=(count(0xa9)*4+15)&~15u,buffers=16*(count(0xb1)+count(0xb2));
        const u32 handles=0x10+push+buffers;
        if(offset<handles)return 2;
        if(offset<handles+8*(count(0xb3)+count(0xb6)))return 1;
        return 0;
    }
    if(bank==0&&offset>=StorageTableBase&&offset<StorageTableEnd)return 2;
    return 0;
}
bool ConstantPointerAvailable(void* context,u32 bank,u32 offset){
    auto& stage=*static_cast<StageState*>(context);
    if(bank<3||(bank==RootBank&&!stage.uniforms[RootBank-UniformBankBase].address))return offset+8<=0x1000;
    const u32 index=bank-UniformBankBase;
    return index<stage.uniforms.size()&&stage.uniforms[index].address&&u64(offset)+8<=stage.uniforms[index].size;
}
u32 CurrentDependencyValue(StageState& stage,const nvshader::Dependency& d){
    switch(d.kind){
    case nvshader::DependencyKind::Mapped:{if(!ConstantPointerAvailable(&stage,d.bank,d.offset))return 0;const u64 a=Cbuf(&stage,d.bank,d.offset)|(u64)Cbuf(&stage,d.bank,d.offset+4)<<32;return LookupAddress(a).pool!=nullptr;}
    case nvshader::DependencyKind::IndirectMapped:{if(!ConstantPointerAvailable(&stage,d.bank,d.offset))return 0;const u64 a=Cbuf(&stage,d.bank,d.offset)|(u64)Cbuf(&stage,d.bank,d.offset+4)<<32;return LookupAddress(ReadPointer(nullptr,a+d.indirect_offset)).pool!=nullptr;}
    case nvshader::DependencyKind::Texture:return DescribeTexture(nullptr,Cbuf(&stage,d.bank,d.offset));
    default:return Cbuf(&stage,d.bank,d.offset);
    }
}
struct ShaderVariant { const ShaderCode* source; nvshader::Request request; nvshader::Result compiled; VkShaderModule module=VK_NULL_HANDLE; VkShaderModule fallback_vertex_module=VK_NULL_HANDLE; /* (mesh fallback: the generated vertex stage of a mesh program) */ std::vector<std::array<u32,3>> texture_dependencies;
    u64 cache_key=0,identity=0;
    mutable std::atomic<u64> resource_hash{0}; }; // (see SameResources) // (identity: what the translation is, the same in every run; cache_key 0: not in the shader cache)
struct CompileTime { double translate=0, pipelines=0; u32 created=0, cached=0; } compile_time; // seconds since the last report
// This frame's executor time in work that varies with new content (slow-frame diagnosis).
struct Hitch { double translate=0, pipeline_wait=0, lookup=0, module=0; } hitch; // (lookup: GetPipeline in all, with the translation inside it)
std::vector<std::unique_ptr<ShaderVariant>> shaders;
// The variants of each program, so a lookup scans only that program's few variants.
std::unordered_map<const ShaderCode*,std::vector<ShaderVariant*>> variants_by_code;
// Shader I/O must match the view format, including when compressed storage is
// exposed as integer texels to generate its blocks on the GPU.
u8 AttributeType(VkFormat format) {
    switch(format) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8B8_UINT:
    case VK_FORMAT_B8G8R8_UINT: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_B8G8R8A8_UINT:
    case VK_FORMAT_A8B8G8R8_UINT_PACK32: case VK_FORMAT_A2R10G10B10_UINT_PACK32:
    case VK_FORMAT_A2B10G10R10_UINT_PACK32: case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16B16_UINT: case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R32_UINT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32B32_UINT:
    case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R64_UINT: return 2;
    case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8_SINT:
    case VK_FORMAT_B8G8R8_SINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_B8G8R8A8_SINT:
    case VK_FORMAT_A8B8G8R8_SINT_PACK32: case VK_FORMAT_A2R10G10B10_SINT_PACK32:
    case VK_FORMAT_A2B10G10R10_SINT_PACK32: case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16B16_SINT: case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32B32_SINT:
    case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_R64_SINT: return 1;
    default: return 0;
    }
}
// ---- persistent translated-shader cache ---------------------------------------------------------
// Translating a shader takes milliseconds; entering an area translated hundreds (up to ~1 s).
// Each translation is stored with the constants it depended on, keyed by everything the variant
// match above compares; a later run whose constants match uses it instead of translating.
// Bump ShaderCacheVersion when the translator's output changes. OOT_NO_SHADER_CACHE=1 disables it.
// (Mesh programs carry a "u" in the key: since the empty-output ending of unreachable branches, only they were retranslated.)
constexpr u32 ShaderCacheVersion=12; // 12: the depth conversion exact; // 11: mesh output defaults, index and task count bounds (from Longshot); // 10: the mesh index map in whole 32-primitive blocks; 9: positions Invariant; // 8: task payload byte size converted to 128-byte lines correctly
bool ShaderCacheEnabled(){static const bool on=std::getenv("OOT_NO_SHADER_CACHE")==nullptr;return on;}
std::string ShaderCacheDirectory(){
    static const std::string dir=[]{std::string d=rt::DataDir()+"/shader_cache";
        if(const char* e=std::getenv("OOT_SHADER_CACHE_DIR"))d=e; // (tests: a cold start without touching the real cache)
        std::filesystem::create_directories(d);return d;}();
    return dir;
}
u64 ShaderCacheKey(const ShaderCode& code,const nvshader::Request& r){
    u64 h=0xcbf29ce484222325ull;const auto mix=[&](const void* p,size_t n){const u8* b=static_cast<const u8*>(p);for(size_t i=0;i<n;++i)h=(h^b[i])*0x100000001b3ull;};
    mix(&ShaderCacheVersion,4);
#ifdef OOT_SHADER_LONGSHOT
    mix("longshot-3",10); // (translations of the other translator are not this one's)
#endif
    {const char* inv=std::getenv("OOT_POSITION_INVARIANT");if(inv&&!std::atoi(inv))mix("noinv",5);}{if(std::getenv("OOT_NO_CONTRACTION"))mix("nocontract",10);}{if(std::getenv("OOT_FMA_SPLIT"))mix("fmasplit",8);}{if(std::getenv("OOT_SHUFFLE_BROADCAST"))mix("shflbm",6);}{if(const char* r=std::getenv("OOT_MESH_ROUND"))mix(r,std::strlen(r));}{const char* fog=std::getenv("OOT_FOG_SCALE");const char* nan=std::getenv("OOT_NAN_GUARD");const char* lz=std::getenv("OOT_LOCAL_ZERO");const char* crs=std::getenv("OOT_LOCAL_CRS");const char* mz=std::getenv("OOT_MESH_NO_ZERO");const char* nc=std::getenv("OOT_MESH_NOCULL");const std::string k=std::string(code.stage==7?"u":"")+(fog?fog:"")+"/"+(nan?nan:"")+"/"+(lz?lz:"")+"/"+(crs?crs:"")+"/"+(mz?mz:"")+"/"+(nc?nc:"");mix(k.data(),k.size());}mix(code.code.data(),code.code.size());mix(code.control.data(),code.control.size());
    mix(&r.stage,4);mix(&r.binding_base,4);mix(r.input_types.data(),r.input_types.size());mix(r.color_types.data(),r.color_types.size());
    mix(&r.have_previous_stores,1);mix(r.previous_stores.data(),sizeof(u64)*r.previous_stores.size());mix(&r.topology,4);mix(&r.convert_depth,1);
    mix(&r.point_size,4);mix(r.mesh_registers.data(),4*r.mesh_registers.size());mix(&r.storage_alignment,8);mix(&r.buffer_float32_atomic_add,1);
#ifdef OOT_SHADER_LONGSHOT
    {const u8 other=r.vendor_id!=0x10DE;mix(&other,1);} // (translated differently for cards that are not NVIDIA's)
    if(r.vendor_id!=0x10DE){mix("lanes1",6);mix(&r.unpinned_lanes,4);} // (gather offsets, and stages translated for wider subgroups)
#endif
    mix(&r.negate_y,1);mix(&r.force_early_depth,1);mix(&r.dual_source,1);
    if(r.stage==5&&nvshader::NativeMatrixMultiply())mix("coopmat1",8); // (compute programs' matrix multiplies on cooperative matrices)
    // (the mesh fallback's stages, and a fragment stage whose mesh stage writes per primitive or draws points: keys of their own)
    if(r.mesh_fallback||(r.previous_mesh&&(r.previous_per_primitive||r.previous_points))){mix("meshfb2",7);mix(&r.mesh_fallback,1);mix(&r.mesh_fallback_task,1);mix(&r.mesh_fallback_task_threads,4);mix(&r.previous_per_primitive,4);mix(&r.previous_points,1);}
    return h;
}
struct ShaderCacheWriter{std::vector<u8> bytes;
    template<typename T> void put(const T& v){const u8* b=reinterpret_cast<const u8*>(&v);bytes.insert(bytes.end(),b,b+sizeof v);}
    template<typename T> void vec(const std::vector<T>& v){put((u32)v.size());const u8* b=reinterpret_cast<const u8*>(v.data());bytes.insert(bytes.end(),b,b+v.size()*sizeof(T));}};
struct ShaderCacheReader{const u8* p;const u8* end;bool ok=true;
    template<typename T> void get(T& v){if(p+sizeof v>end){ok=false;return;}std::memcpy(&v,p,sizeof v);p+=sizeof v;}
    template<typename T> void vec(std::vector<T>& v){u32 n=0;get(n);if(!ok||p+(u64)n*sizeof(T)>end){ok=false;return;}v.resize(n);std::memcpy(v.data(),p,(u64)n*sizeof(T));p+=(u64)n*sizeof(T);}};
void SerializeResult(ShaderCacheWriter& w,const nvshader::Result& r){
    w.put(u32{0x53434832});w.vec(r.spirv);w.vec(r.dependencies);w.put(r.stores);w.put(r.global_loads);w.put(r.global_stores);
    w.put(r.mesh_threads);w.put(r.mesh_payload_words);w.put(r.mesh_max_vertices);w.put(r.mesh_max_primitives);w.put(r.mesh_topology);w.put(r.mesh_per_primitive);w.put(r.mesh_record_words);w.vec(r.fallback_vertex_spirv);
    w.put((u32)r.resources.size());
    for(const auto& x:r.resources){w.put(x.kind);w.put(x.binding);w.put(x.count);w.put(x.bank);w.put(x.offset);w.put(x.secondary_bank);w.put(x.secondary_offset);
        w.put(x.shift);w.put(x.secondary_shift);w.put(x.size_shift);w.put(x.texture_type);w.put(x.image_format);w.put(x.indirect_offset);
        w.put(x.storage_index);w.put(x.storage_offset);w.put(x.storage_stride);w.put(x.sized_storage);w.put(x.secondary);w.put(x.depth);
        w.put(x.multisample);w.put(x.integer);w.put(x.written);w.vec(x.handle_program);}
}
bool DeserializeResult(ShaderCacheReader& rd,nvshader::Result& r){
    u32 magic=0;rd.get(magic);if(!rd.ok||(magic!=0x53434831&&magic!=0x53434832))return false;
    rd.vec(r.spirv);rd.vec(r.dependencies);rd.get(r.stores);rd.get(r.global_loads);rd.get(r.global_stores);
    if(magic==0x53434832){rd.get(r.mesh_threads);rd.get(r.mesh_payload_words);rd.get(r.mesh_max_vertices);rd.get(r.mesh_max_primitives);rd.get(r.mesh_topology);rd.get(r.mesh_per_primitive);rd.get(r.mesh_record_words);rd.vec(r.fallback_vertex_spirv);} // (entries written before the mesh fallback: none of that)
    {static std::atomic<u32> mv{0},mp{0},mw{0};if(r.mesh_max_vertices>mv.load()||r.mesh_max_primitives>mp.load()||r.mesh_payload_words>mw.load()){mv=std::max(mv.load(),r.mesh_max_vertices);mp=std::max(mp.load(),r.mesh_max_primitives);mw=std::max(mw.load(),r.mesh_payload_words);
        RT_LOG("mesh programs so far: up to %u vertices, %u primitives, %u payload words, %u threads (this one)",mv.load(),mp.load(),mw.load(),r.mesh_threads);}}
    u32 n=0;rd.get(n);if(!rd.ok||n>4096)return false;r.resources.resize(n);
    for(auto& x:r.resources){rd.get(x.kind);rd.get(x.binding);rd.get(x.count);rd.get(x.bank);rd.get(x.offset);rd.get(x.secondary_bank);rd.get(x.secondary_offset);
        rd.get(x.shift);rd.get(x.secondary_shift);rd.get(x.size_shift);rd.get(x.texture_type);rd.get(x.image_format);rd.get(x.indirect_offset);
        rd.get(x.storage_index);rd.get(x.storage_offset);rd.get(x.storage_stride);rd.get(x.sized_storage);rd.get(x.secondary);rd.get(x.depth);
        rd.get(x.multisample);rd.get(x.integer);rd.get(x.written);rd.vec(x.handle_program);}
    return rd.ok;
}
std::unordered_map<u64,std::vector<nvshader::Result>> shader_cache_entries; // key -> cached translations (loaded on first use)
std::unordered_set<u64> shader_cache_loaded;
std::mutex shader_cache_mutex; // (the executor and the prewarm threads)
// What a translation is, the same in every run: its SPIR-V (before the address table is patched in) and the layout of its resources.
u64 ShaderIdentity(const nvshader::Result& r,u32 stage){
    u64 h=0xcbf29ce484222325ull^stage;
    for(u32 w:r.spirv)h=(h^w)*0x100000001b3ull;
    for(const auto& x:r.resources){h=(h^(u64)x.binding)*0x100000001b3ull;h=(h^(u64)(u32)x.kind)*0x100000001b3ull;h=(h^(u64)x.count)*0x100000001b3ull;}
    return h;
}
// (an entry used again after a day gets its file's time renewed: the storage cleanup in prewarm.cpp removes what is long unused)
void ShaderCacheRenew(const std::string& path){
    std::error_code ec;const auto now=std::filesystem::file_time_type::clock::now();
    const auto written=std::filesystem::last_write_time(path,ec);
    if(!ec&&now-written>std::chrono::hours(24))std::filesystem::last_write_time(path,now,ec);
}
const std::vector<nvshader::Result>& ShaderCacheLookupLocked(u64 key){
    auto& list=shader_cache_entries[key];
    if(shader_cache_loaded.insert(key).second){
        char path[64];std::snprintf(path,sizeof path,"/%016llx.bin",(unsigned long long)key);
        const std::string file=ShaderCacheDirectory()+path;
        if(FILE* f=std::fopen(file.c_str(),"rb")){
            std::vector<u8> data;u8 buf[65536];size_t n;while((n=std::fread(buf,1,sizeof buf,f))>0)data.insert(data.end(),buf,buf+n);std::fclose(f);
            ShaderCacheReader rd{data.data(),data.data()+data.size()};
            while(rd.p<rd.end){nvshader::Result r;if(!DeserializeResult(rd,r))break;list.push_back(std::move(r));}
            ShaderCacheRenew(file);
        }
    }
    return list;
}
const std::vector<nvshader::Result>& ShaderCacheLookup(u64 key){
    std::lock_guard lock{shader_cache_mutex};
    return ShaderCacheLookupLocked(key);
}
// The translation a pipeline record names (prewarm threads): a copy, the cache's lists belong to the executor.
bool ShaderCacheFind(u64 key,u64 identity,u32 stage,nvshader::Result& out){
    std::lock_guard lock{shader_cache_mutex};
    for(const auto& r:ShaderCacheLookupLocked(key))if(ShaderIdentity(r,stage)==identity){out=r;return true;}
    return false;
}
// Programs reading global memory embed the address of this run's guest address table: they are translated with
// a placeholder there, cached as such, and get the real address just before their module is made
// (they were translated again every run: 100-900 ms of hitches).
constexpr u64 AddressTablePlaceholder=0x5eedf00dcafe0008ull;
constexpr u32 GuestPagesPushOffset=48; // (Longshot's translator: GUEST_PAGES_PUSH_OFFSET, after its rescaling and render area blocks; unused words otherwise)
void PatchAddressTable(std::vector<u32>& spirv){
    const u32 lo=(u32)AddressTablePlaceholder,hi=(u32)(AddressTablePlaceholder>>32);const u64 real=GlobalAddressTable(false);
    for(size_t i=0;i+1<spirv.size();++i)if(spirv[i]==lo&&spirv[i+1]==hi){spirv[i]=(u32)real;spirv[i+1]=(u32)(real>>32);}
}
void ShaderCacheStore(u64 key,const nvshader::Result& r){
    ShaderCacheWriter w;SerializeResult(w,r);
    char path[64];std::snprintf(path,sizeof path,"/%016llx.bin",(unsigned long long)key);
    if(FILE* f=std::fopen((ShaderCacheDirectory()+path).c_str(),"ab")){std::fwrite(w.bytes.data(),1,w.bytes.size(),f);std::fclose(f);}
    std::lock_guard lock{shader_cache_mutex};
    shader_cache_entries[key].push_back(r);
}
ShaderVariant& CompileShader(const ShaderCode& code, u32 binding_base, const ShaderVariant* previous) {
    auto& stage = (*tls_state).stages[ShaderPublicStage[code.stage]];
    nvshader::Request request; request.stage=code.stage; request.binding_base=binding_base; request.code=&code.code; request.control=&code.control;
    request.context=&stage; request.read_constant=Cbuf; request.mapped_address=[](void*,u64 a){return LookupAddress(a).pool != nullptr;};
    request.read_pointer=ReadPointer;
    request.constant_pointer_available=ConstantPointerAvailable;
    request.texture_type=TextureType; request.texture_format=TextureFormat;
    request.classify_constant=ClassifyConstant; request.describe_texture=DescribeTexture;
    request.global_address_table=AddressTablePlaceholder; request.storage_alignment=Vk().properties.limits.minStorageBufferOffsetAlignment;
    request.buffer_float32_atomic_add=Vk().buffer_float32_atomic_add;
    request.vendor_id=Vk().properties.vendorID;
    if(Vk().subgroup_max>32&&code.stage<sizeof ShaderVkStage/sizeof *ShaderVkStage&&!(Vk().pin_stages&ShaderVkStage[code.stage])&&!(code.stage==5&&Vk().pin_subgroup))request.unpinned_lanes=Vk().subgroup_max;
    // Input types describe vertex fetch, so they apply to the vertex stage only; later stages take
    // float inputs from the previous stage (gated by its store mask).
    request.input_types.fill(code.stage==0?5:0);
    for(u32 i=0;i<32;++i) if(code.stage==0&&((*tls_state).attributes[i]&0x40)) {
        request.input_types[i]=AttributeType(GetFormat((*tls_state).attributes[i]>>21&0x3ff).vk);
    }
    if(code.stage==1)for(u32 i=0;i<(*tls_state).color_count;++i)if((*tls_state).targets.colors[i]) {
        auto* t=HostTexture((*tls_state).targets.colors[i]);
        request.color_types[i]=AttributeType((*tls_state).targets.color_views[i].format?
            GetFormat((*tls_state).targets.color_views[i].format).vk:t->info.vk);
    }
    if(previous) {request.have_previous_stores=true;request.previous_stores=previous->compiled.stores;}
    // Mesh fallback (vk.cpp): the task and mesh programs as compute programs; the fragment program is told what its mesh program does.
    if(Vk().mesh_fallback&&(code.stage==6||code.stage==7)){request.mesh_fallback=true;
        if(code.stage==7&&previous&&previous->source&&previous->source->stage==6){request.mesh_fallback_task=true;request.mesh_fallback_task_threads=std::max(previous->compiled.mesh_threads,1u);}}
    if(code.stage==1&&previous&&previous->source&&previous->source->stage==7){request.previous_mesh=true;request.mesh_fallback=Vk().mesh_fallback;
        request.previous_per_primitive=previous->compiled.mesh_per_primitive;request.previous_points=(MeshRegisters(*previous->source)[0]&0xf)==0;}
    request.topology=(*tls_state).primitive<=1?0:(*tls_state).primitive<=3?1:3;
    // OpenGL-style clip depth (-w..w) is remapped to Vulkan's 0..w by the last pre-raster stage.
    const u32 last_pre_raster=code.stage>=6?7:(*tls_state).stages[2].shader?2:(*tls_state).stages[4].shader?4:0;
    if(code.stage==6||code.stage==7)request.mesh_registers=MeshRegisters(code);
    request.convert_depth=DepthNearIsMinusW()&&ShaderPublicStage[code.stage]==last_pre_raster;
    if((*tls_state).primitive==0&&ShaderPublicStage[code.stage]==last_pre_raster)request.point_size=std::max((*tls_state).point_size,1.f);
    // The variant that matched last time usually matches again: matches move to the front.
    static std::vector<ShaderVariant*> no_candidates;
    const auto known=resolve_worker?variants_by_code.find(&code):variants_by_code.end();
    auto& candidates=resolve_worker?(known!=variants_by_code.end()?known->second:no_candidates):variants_by_code[&code];
    for (size_t index=0;index<candidates.size();++index) {
        auto* variant=candidates[index];
        const auto& q=variant->request;
        if(variant->source!=&code || q.binding_base!=binding_base || q.input_types!=request.input_types || q.color_types!=request.color_types || q.previous_stores!=request.previous_stores || q.topology!=request.topology || q.convert_depth!=request.convert_depth || q.point_size!=request.point_size || q.mesh_fallback!=request.mesh_fallback || q.mesh_fallback_task!=request.mesh_fallback_task || q.mesh_fallback_task_threads!=request.mesh_fallback_task_threads || q.previous_mesh!=request.previous_mesh || q.previous_per_primitive!=request.previous_per_primitive || q.previous_points!=request.previous_points) continue;
        bool match=true;static const bool why=std::getenv("OOT_VARIANT_TRACE")!=nullptr;
        for(const auto& d:variant->compiled.dependencies){
            const u32 now=CurrentDependencyValue(stage,d);
            if(now!=d.value){match=false;if(why)RT_LOG("  variant miss %#llx: kind %u cbuf %u+%#x was %#x now %#x",(unsigned long long)code.address,(u32)d.kind,d.bank,d.offset,d.value,now);break;}}
        if(match){if(index&&!resolve_worker)std::swap(candidates[0],candidates[index]);return *variant;}
    }
    if(resolve_worker){static ShaderVariant none;RESOLVE_BAIL();return none;} // (a new variant: translated by the executor)
    auto variant=std::make_unique<ShaderVariant>(); variant->source=&code; variant->request=request;
    // Content hash: stable across runs (UI programs load at different addresses each time).
    const u64 code_hash=[&]{u64 h=0xcbf29ce484222325ull;for(u8 b:code.code)h=(h^b)*0x100000001b3ull;return h;}();
    RT_LOG("compiling native shader stage %u at %#llx (variant %zu, code hash %016llx)",code.stage,(unsigned long long)code.address,shaders.size(),(unsigned long long)code_hash);
    if(const char* dir=std::getenv("OOT_SHADER_DUMP")){
        char path[512];std::snprintf(path,sizeof path,"%s/%llx-stage%u.bin",dir,(unsigned long long)code.address,code.stage);
        if(FILE* f=std::fopen(path,"wb")){std::fwrite(code.code.data(),1,code.code.size(),f);std::fclose(f);}
        std::snprintf(path,sizeof path,"%s/%llx-stage%u.ctl",dir,(unsigned long long)code.address,code.stage); // (its control section: offline translation needs it)
        if(FILE* f=std::fopen(path,"wb")){std::fwrite(code.control.data(),1,code.control.size(),f);std::fclose(f);}
    }
    if(std::getenv("OOT_STORES_TRACE")&&previous){std::string bits;for(u32 a=32;a<32+4*8;++a)bits+=(previous->compiled.stores[a/64]>>(a%64)&1)?'1':'0';RT_LOG("  previous stage %#llx stores generics 0-7: %s",(unsigned long long)previous->source->address,bits.c_str());}
    // A translation cached by an earlier run whose constants match the current state.
    const u64 cache_key=ShaderCacheEnabled()&&!std::getenv("OOT_SHADER_OVERRIDE")?ShaderCacheKey(code,request):0;
    bool from_cache=false;
    if(cache_key)for(const auto& cached:ShaderCacheLookup(cache_key)){
        bool match=true;for(const auto& d:cached.dependencies)if(CurrentDependencyValue(stage,d)!=d.value){match=false;break;}
        if(match){variant->compiled=cached;from_cache=true;++compile_time.cached;break;}
    }
    if(!from_cache)
    try {{rt::CoreRelease release;const auto t0=std::chrono::steady_clock::now();variant->compiled=nvshader::Compile(request);
            const double spent=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();compile_time.translate+=spent;hitch.translate+=spent;} // long host work
        if(cache_key)ShaderCacheStore(cache_key,variant->compiled);
        if(std::getenv("OOT_SHADER_RESOURCES"))for(const auto& r:variant->compiled.resources)RT_LOG("  stage %u resource kind %d bank %u offset %#x count %u",code.stage,(int)r.kind,r.bank,r.offset,r.count);} catch(const std::exception& error){
        // Like the console-driver translation this comes from, a program the translator cannot
        // express is skipped (its draws/dispatches are dropped) instead of stopping the game.
        RT_LOG("shader %#llx (stage %u) not translatable, its work is skipped: %s",(unsigned long long)code.address,code.stage,error.what());
        variant->module=VK_NULL_HANDLE;auto* result=variant.get();variants_by_code[&code].push_back(result);shaders.push_back(std::move(variant));return *result;}
    // OOT_SHADER_OVERRIDE=<dir>: <address>-stage<N>.spv there replaces the translated module
    // (same interface; for instrumenting a shader while debugging).
    if(const char* dir=std::getenv("OOT_SHADER_OVERRIDE")){
        char path[512];std::snprintf(path,sizeof path,"%s/%llx-stage%u.spv",dir,(unsigned long long)code.address,code.stage);
        FILE* f=std::fopen(path,"rb");
        if(!f){std::snprintf(path,sizeof path,"%s/h%016llx-stage%u.spv",dir,(unsigned long long)code_hash,code.stage);f=std::fopen(path,"rb");} // (or by code hash)
        if(f){std::vector<u32> words;u32 w;while(std::fread(&w,4,1,f)==1)words.push_back(w);std::fclose(f);
            RT_LOG("shader %#llx stage %u: using override %s (%zu words)",(unsigned long long)code.address,code.stage,path,words.size());variant->compiled.spirv=std::move(words);}
    }
    variant->cache_key=cache_key;variant->identity=ShaderIdentity(variant->compiled,code.stage);
    if(variant->compiled.global_loads||variant->compiled.global_stores)PatchAddressTable(variant->compiled.spirv);
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; info.codeSize=variant->compiled.spirv.size()*4;info.pCode=variant->compiled.spirv.data();
    {const auto module_t0=std::chrono::steady_clock::now();VK_CHECK(vkCreateShaderModule(Vk().device,&info,nullptr,&variant->module));hitch.module+=std::chrono::duration<double>(std::chrono::steady_clock::now()-module_t0).count();}
    if(!variant->compiled.fallback_vertex_spirv.empty()){VkShaderModuleCreateInfo vi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};vi.codeSize=variant->compiled.fallback_vertex_spirv.size()*4;vi.pCode=variant->compiled.fallback_vertex_spirv.data();
        VK_CHECK(vkCreateShaderModule(Vk().device,&vi,nullptr,&variant->fallback_vertex_module));
        if(const char* dir=std::getenv("OOT_SHADER_DUMP")){char path[512];std::snprintf(path,sizeof path,"%s/%llx-v%zu-fbvertex.spv",dir,(unsigned long long)code.address,shaders.size());if(FILE* file=std::fopen(path,"wb")){std::fwrite(vi.pCode,1,vi.codeSize,file);std::fclose(file);}}}
    if(std::getenv("OOT_SHADER_DUMP")) {char path[512];std::snprintf(path,sizeof path,"%s/%llx-v%zu.spv",std::getenv("OOT_SHADER_DUMP"),(unsigned long long)code.address,shaders.size());FILE* file=std::fopen(path,"wb");if(file){std::fwrite(info.pCode,1,info.codeSize,file);std::fclose(file);}}
    auto* result=variant.get(); variants_by_code[&code].push_back(result);shaders.push_back(std::move(variant));return *result;
}

// Background pipeline compilation.
// The threads that make pipelines. Jobs carry a key (their Pipeline): Urgent(key) moves a waiting job ahead of the
// others - a pipeline a draw of this frame is left out for goes before the hundreds a new area asks for at once.
// Extra threads join while more than a few jobs wait (the first visit to an area: a thousand pipelines, 70 s of
// compiling), and sleep otherwise. OOT_PIPELINE_THREADS=<n>: the usual threads; OOT_PIPELINE_SURGE=<n>: the extra.
class WorkerPool {
public:
    WorkerPool(unsigned count,unsigned surge){for(unsigned i=0;i<count+surge;++i)threads.emplace_back([this,i,count]{Loop(i>=count);});}
    void Run(std::function<void()> job,const void* key=nullptr){
        size_t waiting;
        {std::lock_guard lock{mutex};normal.push_back({std::move(job),key});if(key)queued[key]=std::prev(normal.end());waiting=normal.size()+urgent.size();}
        ready.notify_one();if(waiting>SurgeFrom)ready.notify_all();
    }
    void Urgent(const void* key){
        std::lock_guard lock{mutex};
        const auto it=queued.find(key);if(it==queued.end())return; // (started, done, or already urgent)
        urgent.splice(urgent.end(),normal,it->second);queued.erase(it);
    }
private:
    static constexpr size_t SurgeFrom=8;
    struct Job{std::function<void()> run;const void* key;};
    void Loop(bool surge){for(;;){Job job;
        {std::unique_lock lock{mutex};ready.wait(lock,[&]{return !urgent.empty()||(surge?normal.size()>SurgeFrom:!normal.empty());});
         auto& from=urgent.empty()?normal:urgent;job=std::move(from.front());
         if(&from==&normal&&job.key)queued.erase(job.key);
         from.pop_front();}
        job.run();}}
    std::mutex mutex;std::condition_variable ready;std::list<Job> normal,urgent;std::unordered_map<const void*,std::list<Job>::iterator> queued;std::vector<std::thread> threads;
};
WorkerPool& PipelineWorkers(){
    static WorkerPool pool{[]{const unsigned hw=std::thread::hardware_concurrency();
            return std::getenv("OOT_PIPELINE_THREADS")?(unsigned)std::max(1,std::atoi(std::getenv("OOT_PIPELINE_THREADS"))):std::max(2u,std::min(6u,hw/3));}(),
        []{const unsigned hw=std::thread::hardware_concurrency(),usual=std::max(2u,std::min(6u,hw/3));
            return std::getenv("OOT_PIPELINE_SURGE")?(unsigned)std::max(0,std::atoi(std::getenv("OOT_PIPELINE_SURGE"))):(hw*3/4>usual+2?hw*3/4-2-usual:0u);}()};
    return pool;}
std::mutex compile_time_mutex;
std::atomic<u64> quick_wait_frame{0},quick_wait_us{0};
bool QuickWaitAllowed(){ // (see where pipelines are queued)
    if(quick_wait_frame.load(std::memory_order_relaxed)!=frame_number){quick_wait_frame.store(frame_number,std::memory_order_relaxed);quick_wait_us.store(0,std::memory_order_relaxed);}
    return quick_wait_us.load(std::memory_order_relaxed)<12000;
}
struct Pipeline {
    std::atomic<VkPipeline> pipeline{VK_NULL_HANDLE}; // null while a worker compiles it
    VkPipelineLayout layout=VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptors=VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool=VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set=VK_NULL_HANDLE;
    u64 descriptor_hash=0,descriptor_flushes=~0ull; // contents and batch of descriptor_set (reused while unchanged)
    bool push=false; // push-descriptor layout: draws push their descriptors (no set)
    // Mesh fallback: the task and mesh programs' compute pipelines (set before `pipeline`, which is then the generated
    // vertex stage's and the fragment program's), the task program's threads a workgroup, its payload words.
    bool fallback=false;VkPipeline task_compute=VK_NULL_HANDLE,mesh_compute=VK_NULL_HANDLE;u32 fallback_task_threads=0,fallback_payload_words=0,fallback_record_words=0;
    bool waited=false; // a draw has waited for this pipeline once (the executor's: see Draw, draws shading over a depth pre-pass)
    std::vector<ShaderVariant*> shaders;
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    u64 record=0;       // its pipeline list record (0: none)
    bool touched=true;  // false for a pipeline made before the game, until a draw first asks for it (the executor's)
};
// (by a hash of the key: a std::map compared keys of ~400 bytes several times per draw, 5% of the executor's time)
struct PipelineKeyHash{size_t operator()(const std::string& k)const{u64 h=0xcbf29ce484222325ull;size_t i=0;
    for(;i+8<=k.size();i+=8){u64 w;std::memcpy(&w,k.data()+i,8);h=(h^w)*0x9E3779B97F4A7C15ull;h^=h>>29;}
    for(;i<k.size();++i)h=(h^(u8)k[i])*0x100000001b3ull;return h;}};
std::unordered_map<std::string,std::unique_ptr<Pipeline>,PipelineKeyHash> pipelines;

VkDescriptorType DescriptorType(nvshader::ResourceKind kind) {
    using K=nvshader::ResourceKind;
    switch(kind) {
    case K::UniformBuffer:return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    case K::StorageBuffer:return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case K::Texture:return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    case K::Image:return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case K::TextureBuffer:return VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    case K::ImageBuffer:return VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    } Fatal("unknown shader resource");
}
VkBlendFactor BlendFactor(u32 value) {
    switch(value) {
    case 1:return VK_BLEND_FACTOR_ZERO;case 2:return VK_BLEND_FACTOR_ONE;
    case 3:return VK_BLEND_FACTOR_SRC_COLOR;case 4:return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case 5:return VK_BLEND_FACTOR_SRC_ALPHA;case 6:return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case 7:return VK_BLEND_FACTOR_DST_ALPHA;case 8:return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case 9:return VK_BLEND_FACTOR_DST_COLOR;case 10:return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case 11:return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case 16:return VK_BLEND_FACTOR_SRC1_COLOR;case 17:return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
    case 18:return VK_BLEND_FACTOR_SRC1_ALPHA;case 19:return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
    case 0x61:return VK_BLEND_FACTOR_CONSTANT_COLOR;case 0x62:return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case 0x63:return VK_BLEND_FACTOR_CONSTANT_ALPHA;case 0x64:return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    default:Fatal("unknown NVN blend factor %u",value);
    }
}
// A compute stage held to the console GPU's 32-lane subgroups where the device's own are larger (vk.cpp pin_subgroup: AMD
// runs 64-lane waves, and the game's programs take a subgroup for a 32-thread warp). `threads`: the workgroup's x size
// when known - a multiple of 32 also asks for whole subgroups. `pin` must live until the pipeline is created.
// A graphics pipeline's stages held to 32 lanes where the GPU can (vk.pin_stages: on AMD's RDNA 2 the fragment, task and
// mesh stages; a stage it cannot hold was translated for the wider subgroups).
struct StagePins{std::array<VkPipelineShaderStageRequiredSubgroupSizeCreateInfo,8> pins;std::vector<VkPipelineShaderStageCreateInfo> stages;
    const std::vector<VkPipelineShaderStageCreateInfo>& Apply(const std::vector<VkPipelineShaderStageCreateInfo>& from){
        if(!Vk().pin_stages)return from;
        stages=from;
        for(size_t i=0;i<stages.size()&&i<pins.size();++i)if(Vk().pin_stages&stages[i].stage){
            pins[i]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};pins[i].requiredSubgroupSize=32;pins[i].pNext=const_cast<void*>(stages[i].pNext);stages[i].pNext=&pins[i];}
        return stages;}
};
void PinSubgroup(VkPipelineShaderStageCreateInfo& stage,VkPipelineShaderStageRequiredSubgroupSizeCreateInfo& pin,u32 threads=0){
    if(!Vk().pin_subgroup)return;
    pin={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};pin.requiredSubgroupSize=32;pin.pNext=const_cast<void*>(stage.pNext);stage.pNext=&pin;
    if(Vk().full_subgroups&&threads&&threads%32==0)stage.flags|=VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
}
VkPrimitiveTopology Topology(u32 primitive) {
    switch(primitive){
    case 0:return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;case 1:return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case 3:return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;case 4:return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    case 5:return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;case 6:return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    case 10:return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;case 11:return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
    case 12:return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY;case 13:return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
    case 14:return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    default:Fatal("NVN primitive %u requires index conversion",primitive);
    }
}
// Everything pipeline creation reads of the draw state, normalized (bits it does not read are zero) so that equal pipelines
// look equal; also the key of the pipeline tables and, as raw bytes, the state a pipeline list record holds (pipeline_list.h).
struct PipelineDesc {
    u64 blends[8];
    u64 depth;
    u32 depth_bounds_range[2]; // (float bits)
    u32 attributes[32];
    u32 streams[16][2];
    u32 color_count,color_blend,channel,polygon,primitive,sample_mask,samples;
    u32 colors[8];             // VkFormat
    u32 depth_format,stencil_format;
    u32 restart,depth_clamp,discard,depth_bounds,lower_left,mesh;
    u32 reserved;
};
static_assert(std::has_unique_object_representations_v<PipelineDesc>,"pipeline descriptions are compared and stored as bytes");
PipelineDesc MakeDesc(bool mesh,const std::array<VkFormat,8>& colors,VkFormat depth,VkFormat stencil,u32 samples){
    PipelineDesc d{};
    for(u32 i=0;i<32;++i){const u32 a=(*tls_state).attributes[i];d.attributes[i]=a&0x40?a&(31u|0x40u|0x3fffu<<7|0x3ffu<<21):0u;}
    {std::array<bool,16> used{};for(u32 i=0;i<32;++i)if((*tls_state).attributes[i]&0x40)used[(*tls_state).attributes[i]&31]=true;
     for(u32 i=0;i<16;++i){d.streams[i][0]=used[i]?(*tls_state).streams[i][0]:0u;d.streams[i][1]=used[i]?(*tls_state).streams[i][1]:0u;}}
    d.color_count=(*tls_state).color_count;
    constexpr u64 DefaultBlend=0x01020102110c0000ULL&~0xffffffull; // (what an unblended attachment carries: never read)
    for(u32 i=0;i<(*tls_state).color_count;++i){const bool blended=((*tls_state).color>>(8+i))&1;
        if(blended)d.color_blend|=1u<<i;
        d.blends[i]=blended?(*tls_state).blends[i]&~0xffffffull:DefaultBlend;d.channel|=((*tls_state).channel>>(i*4)&15)<<(i*4);}
    d.polygon=(*tls_state).polygon&(0x1fu|0xfu<<15);d.lower_left=WindowOriginLowerLeft();
    d.depth=((*tls_state).depth&0xf7)|((((*tls_state).depth>>2)&1?(*tls_state).depth:0x1118111800000022ULL)&0xffffffff00000000ull);
    d.depth_bounds=(*tls_state).depth_bounds;
    if((*tls_state).depth_bounds)for(u32 i=0;i<2;++i)std::memcpy(&d.depth_bounds_range[i],&(*tls_state).depth_bounds_range[i],4);
    d.primitive=(*tls_state).primitive;d.restart=(*tls_state).restart;d.depth_clamp=(*tls_state).depth_clamp;d.discard=(*tls_state).discard;d.sample_mask=(*tls_state).sample_mask;d.mesh=mesh;
    for(u32 i=0;i<8;++i)d.colors[i]=colors[i];d.depth_format=depth;d.stencil_format=stencil;d.samples=samples;
    return d;
}
std::string PipelineKey(const PipelineDesc& d,const std::vector<u64>& identities){
    std::string key(reinterpret_cast<const char*>(&d),sizeof d);
    for(u64 id:identities)key.append(reinterpret_cast<const char*>(&id),sizeof id);
    return key;
}
// Descriptor set layout and pipeline layout from the resources of the pipeline's shaders.
using StageResources=std::pair<VkShaderStageFlagBits,const std::vector<nvshader::Resource>*>;
void BuildPipelineLayout(Pipeline& pipeline,const std::vector<StageResources>& stages){
    auto& vk=Vk();
    for(const auto& [stage,resources]:stages)for(const auto& r:*resources){
        const bool buffer=r.kind==nvshader::ResourceKind::UniformBuffer||r.kind==nvshader::ResourceKind::StorageBuffer;
#ifdef OOT_SHADER_LONGSHOT
        const bool table=r.kind==nvshader::ResourceKind::StorageBuffer&&r.count>1; // (a storage buffer table: an array of buffers)
        // (the translator's views of one buffer with different element types share its binding: listed once)
        bool listed=false;for(auto& b:pipeline.bindings)if(b.binding==r.binding){b.stageFlags|=stage;listed=true;}
        if(listed)continue;
        pipeline.bindings.push_back({r.binding,DescriptorType(r.kind),buffer&&!table?1:r.count,stage,nullptr});
#else
        pipeline.bindings.push_back({r.binding,DescriptorType(r.kind),buffer?1:r.count,stage,nullptr});
#endif
    }
    VkDescriptorSetLayoutCreateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};di.bindingCount=pipeline.bindings.size();di.pBindings=pipeline.bindings.data();
    // (graphics only: dispatches bind a descriptor set. A compute pipeline with a push-descriptor layout got a null set bound:
    // its passes ran without resources, and the 3D image and movies went black, now and then)
    bool compute=false;for(const auto& [stage,resources]:stages)compute|=stage==VK_SHADER_STAGE_COMPUTE_BIT;
    {u32 total=0;for(const auto& b:pipeline.bindings)total+=b.descriptorCount;
     if(!compute&&vk.push_descriptor_set&&total<=vk.max_push_descriptors){di.flags=VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;pipeline.push=true;}}
    VK_CHECK(vkCreateDescriptorSetLayout(vk.device,&di,nullptr,&pipeline.descriptors));
    VkPushConstantRange push{VK_SHADER_STAGE_ALL,0,128};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};li.setLayoutCount=1;li.pSetLayouts=&pipeline.descriptors;li.pushConstantRangeCount=1;li.pPushConstantRanges=&push;
    VK_CHECK(vkCreatePipelineLayout(vk.device,&li,nullptr,&pipeline.layout));
}
// The Vulkan pipeline for a description (a worker, or a prewarm thread: those pass fatal=false and get null for a failure).
VkPipeline CreateGraphicsPipeline(const PipelineDesc& d,VkPipelineLayout layout,const std::vector<VkPipelineShaderStageCreateInfo>& stages,u64 fragment_address,bool fatal){
    auto& vk=Vk();const bool mesh=d.mesh==1; // (2: the mesh fallback's pipeline, a vertex stage with no vertex input)
    std::vector<VkVertexInputAttributeDescription> attributes;
    std::vector<VkVertexInputBindingDescription> vertex_bindings;
    std::array<bool,16> used{};
    for(u32 i=0;i<32;++i){const u32 a=d.attributes[i];if(!(a&0x40))continue;const u32 stream=a&31;if(stream>=16)Fatal("vertex stream exceeds native limit");
        attributes.push_back({i,stream,GetFormat(a>>21&0x3ff).vk,(a>>7)&0x3fff});used[stream]=true;}
    for(u32 i=0;i<16;++i)if(used[i]){if(d.streams[i][1]>1)Fatal("vertex instance divisor %u not enabled",d.streams[i][1]);vertex_bindings.push_back({i,d.streams[i][0],d.streams[i][1]?VK_VERTEX_INPUT_RATE_INSTANCE:VK_VERTEX_INPUT_RATE_VERTEX});}
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};vi.vertexAttributeDescriptionCount=attributes.size();vi.pVertexAttributeDescriptions=attributes.data();vi.vertexBindingDescriptionCount=vertex_bindings.size();vi.pVertexBindingDescriptions=vertex_bindings.data();
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};ia.topology=Topology(d.primitive);ia.primitiveRestartEnable=d.restart;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};vp.viewportCount=1;vp.scissorCount=1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};rs.cullMode=d.polygon&3;{static const char* nocull=std::getenv("OOT_CULL_NONE");if(nocull&&fragment_address){char a[24];std::snprintf(a,sizeof a,"%llx",(unsigned long long)fragment_address);if(std::strstr(nocull,a))rs.cullMode=VK_CULL_MODE_NONE;}}
    // Lower-left origin keeps NVN's memory layout with an unflipped Vulkan viewport, which mirrors
    // the window y axis and with it the winding: OpenGL's counter-clockwise is Vulkan's clockwise.
    const bool ccw=(d.polygon&4)!=0;rs.frontFace=ccw!=(d.lower_left!=0)?VK_FRONT_FACE_COUNTER_CLOCKWISE:VK_FRONT_FACE_CLOCKWISE;
    const VkPolygonMode modes[]={VK_POLYGON_MODE_POINT,VK_POLYGON_MODE_LINE,VK_POLYGON_MODE_FILL};u32 mode=d.polygon>>3&3;if(mode>2)Fatal("invalid polygon mode");rs.polygonMode=modes[mode];rs.lineWidth=1;rs.depthBiasEnable=(d.polygon>>15&15)!=0;rs.depthClampEnable=d.depth_clamp;rs.rasterizerDiscardEnable=d.discard;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};ms.rasterizationSamples=(VkSampleCountFlagBits)d.samples;const VkSampleMask mask=d.sample_mask;ms.pSampleMask=&mask;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};ds.depthTestEnable=d.depth&1;ds.depthWriteEnable=(d.depth>>1)&1;ds.depthCompareOp=(VkCompareOp)(((d.depth>>4)&15)-1);{static const u64 nowrite=std::getenv("OOT_FS_NO_DEPTH_WRITE")?std::strtoull(std::getenv("OOT_FS_NO_DEPTH_WRITE"),nullptr,16):0;if(nowrite&&fragment_address==nowrite)ds.depthWriteEnable=VK_FALSE;} // (diagnostic: that fragment program's draws leave depth alone)
    {static const bool relax=std::getenv("OOT_DEPTH_EQUAL_RELAX")!=nullptr;if(relax&&ds.depthCompareOp==VK_COMPARE_OP_EQUAL)ds.depthCompareOp=VK_COMPARE_OP_LESS_OR_EQUAL;}ds.stencilTestEnable=(d.depth>>2)&1;{static const char* always=std::getenv("OOT_DEPTH_ALWAYS");if(always&&fragment_address){char a[24];std::snprintf(a,sizeof a,"%llx",(unsigned long long)fragment_address);if(std::strstr(always,a))ds.depthTestEnable=VK_FALSE;}}ds.depthBoundsTestEnable=d.depth_bounds;std::memcpy(&ds.minDepthBounds,&d.depth_bounds_range[0],4);std::memcpy(&ds.maxDepthBounds,&d.depth_bounds_range[1],4);
    for(u32 i=0;i<2;++i){const u32 p=(u32)(d.depth>>32)>>(i*16);auto& st=i?ds.back:ds.front;st.compareOp=(VkCompareOp)((p&15)-1);st.failOp=(VkStencilOp)(((p>>4)&15)-1);st.depthFailOp=(VkStencilOp)(((p>>8)&15)-1);st.passOp=(VkStencilOp)(((p>>12)&15)-1);}
    std::vector<VkPipelineColorBlendAttachmentState> blend(d.color_count);
    for(u32 i=0;i<blend.size();++i){const u64 b=d.blends[i];auto& out=blend[i];static const bool no_blend=std::getenv("OOT_DEBUG_NOBLEND")!=nullptr;out.blendEnable=!no_blend&&((d.color_blend>>i)&1);out.colorWriteMask=d.channel>>(i*4)&15;
        out.srcColorBlendFactor=BlendFactor(b>>32&127);out.dstColorBlendFactor=BlendFactor(b>>40&127);out.srcAlphaBlendFactor=BlendFactor(b>>48&127);out.dstAlphaBlendFactor=BlendFactor(b>>56&127);
        const VkBlendOp ops[]={VK_BLEND_OP_ADD,VK_BLEND_OP_SUBTRACT,VK_BLEND_OP_REVERSE_SUBTRACT,VK_BLEND_OP_MIN,VK_BLEND_OP_MAX};
        const u32 rgb=b>>24&7,alpha=b>>28&7;if(rgb<1||rgb>5||alpha<1||alpha>5)Fatal("invalid blend equation");out.colorBlendOp=ops[rgb-1];out.alphaBlendOp=ops[alpha-1];}
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};cb.attachmentCount=blend.size();cb.pAttachments=blend.data();
    const VkDynamicState dynamics[]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR,VK_DYNAMIC_STATE_BLEND_CONSTANTS,VK_DYNAMIC_STATE_DEPTH_BIAS,VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,VK_DYNAMIC_STATE_STENCIL_REFERENCE};
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};dy.dynamicStateCount=std::size(dynamics);dy.pDynamicStates=dynamics;
    VkFormat colors[8];for(u32 i=0;i<8;++i)colors[i]=(VkFormat)d.colors[i];
    VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};ri.colorAttachmentCount=d.color_count;ri.pColorAttachmentFormats=colors;ri.depthAttachmentFormat=(VkFormat)d.depth_format;ri.stencilAttachmentFormat=(VkFormat)d.stencil_format;
    VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};pi.pNext=&ri;StagePins pinned;const auto& pinned_stages=pinned.Apply(stages);pi.stageCount=pinned_stages.size();pi.pStages=pinned_stages.data();pi.pVertexInputState=mesh?nullptr:&vi;pi.pInputAssemblyState=mesh?nullptr:&ia;pi.pViewportState=&vp;pi.pRasterizationState=&rs;pi.pMultisampleState=&ms;pi.pDepthStencilState=&ds;pi.pColorBlendState=&cb;pi.pDynamicState=&dy;pi.layout=layout;
    VkPipeline created=VK_NULL_HANDLE;
    const VkResult result=vkCreateGraphicsPipelines(vk.device,PipelineCache(),1,&pi,nullptr,&created);
    if(fatal)VK_CHECK(result);else if(result!=VK_SUCCESS)return VK_NULL_HANDLE;
    return created;
}
// ---- mesh fallback ---------------------------------------------------------------------------------------------------
// A GPU that cannot run the game's task and mesh programs as mesh shader stages (vk.cpp mesh_fallback) runs them as
// compute programs: the task program's workgroups write their payloads and the list of mesh workgroups they start into
// the payload buffer, the mesh program's workgroups (an indirect dispatch of that list) write one record per primitive
// corner into the records buffer, and an ordinary draw with a generated vertex stage reads the records back
// (the translator's emit_spirv_mesh.cpp; the scheme is Longshot's). A mesh draw is then: dispatches, then a draw whose
// vertex count the mesh workgroups counted into the records' header.
struct MeshFallbackStages{VkShaderModule task=VK_NULL_HANDLE,mesh=VK_NULL_HANDLE,vertex=VK_NULL_HANDLE,fragment=VK_NULL_HANDLE;u32 task_threads=0,mesh_threads=0;};
// out: the graphics pipeline, the task program's compute pipeline (null without one), the mesh program's.
bool CreateMeshFallbackPipelines(const PipelineDesc& d,VkPipelineLayout layout,const MeshFallbackStages& m,u64 fragment_address,bool fatal,VkPipeline out[3]){
    auto& vk=Vk();out[0]=out[1]=out[2]=VK_NULL_HANDLE;
    const auto compute=[&](VkShaderModule module,u32 threads)->VkPipeline{
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};ci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=module;ci.stage.pName="main";ci.layout=layout;
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo pin;PinSubgroup(ci.stage,pin,threads);
        VkPipeline created=VK_NULL_HANDLE;const VkResult result=vkCreateComputePipelines(vk.device,PipelineCache(),1,&ci,nullptr,&created);
        if(fatal)VK_CHECK(result);else if(result!=VK_SUCCESS)return VK_NULL_HANDLE;
        return created;};
    const auto discard=[&]{for(u32 i=0;i<3;++i)if(out[i]){vkDestroyPipeline(vk.device,out[i],nullptr);out[i]=VK_NULL_HANDLE;}return false;};
    if(!m.mesh||!m.vertex)return false;
    if(m.task&&!(out[1]=compute(m.task,m.task_threads)))return discard();
    if(!(out[2]=compute(m.mesh,m.mesh_threads)))return discard();
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    {VkPipelineShaderStageCreateInfo si{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};si.stage=VK_SHADER_STAGE_VERTEX_BIT;si.module=m.vertex;si.pName="main";stages.push_back(si);
     if(m.fragment){si.stage=VK_SHADER_STAGE_FRAGMENT_BIT;si.module=m.fragment;stages.push_back(si);}}
    if(!(out[0]=CreateGraphicsPipeline(d,layout,stages,fragment_address,fatal)))return discard();
    return true;
}
// The stage a program's resources are bound for: the fallback's task and mesh programs are compute stages.
VkShaderStageFlagBits BindingStage(u32 stage,bool fallback){return fallback&&(stage==6||stage==7)?VK_SHADER_STAGE_COMPUTE_BIT:ShaderVkStage[stage];}
// A fallback pipeline's description: the mesh program's topology, no vertex input.
void MeshFallbackDesc(PipelineDesc& d,u32 mesh_topology){
    d.mesh=2;d.primitive=mesh_topology;d.restart=0; // (GL's numbers, which are NVN's: 0 points, 1 lines, 4 triangles)
    std::memset(d.attributes,0,sizeof d.attributes);std::memset(d.streams,0,sizeof d.streams);
}
struct MeshFallbackBuffers{VkBuffer records=VK_NULL_HANDLE,payload=VK_NULL_HANDLE,commands=VK_NULL_HANDLE;VkDeviceAddress records_address=0,payload_address=0;u64 payload_size=0;
    const u32* commands_map=nullptr;u32 next_slot=0;VkPipeline advance=VK_NULL_HANDLE;VkPipelineLayout advance_layout=VK_NULL_HANDLE;} mesh_fallback;
constexpr u32 MeshFallbackSlots=16384; // (the draws' indirect commands, one each: a ring far longer than the batches in flight draw)
constexpr u64 MeshFallbackRecordsBytes=u64(nvshader::MeshFallbackHeaderWords+nvshader::MeshFallbackRecordWords)*4;
constexpr u64 MeshFallbackPayloadBytes=48ull<<20; // (the slot map and 2,048 task workgroups' payloads of up to ~5,800 words)
void InitializeMeshFallback(){
    auto& vk=Vk();auto& f=mesh_fallback;if(f.records)return;
    const auto make=[&](u64 size,VkBuffer& buffer,VkDeviceAddress& address){
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=size;
        bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        VK_CHECK(vkCreateBuffer(vk.device,&bi,nullptr,&buffer));
        VkMemoryRequirements req;vkGetBufferMemoryRequirements(vk.device,buffer,&req);
        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};flags.flags=VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.pNext=&flags;ai.allocationSize=req.size;ai.memoryTypeIndex=FindMemoryType(req.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory memory;VK_CHECK(vkAllocateMemory(vk.device,&ai,nullptr,&memory));VK_CHECK(vkBindBufferMemory(vk.device,buffer,memory,0));
        VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};info.buffer=buffer;address=vkGetBufferDeviceAddress(vk.device,&info);};
    make(MeshFallbackRecordsBytes,f.records,f.records_address);
    make(MeshFallbackPayloadBytes,f.payload,f.payload_address);f.payload_size=MeshFallbackPayloadBytes;
    {   // (mesh_advance.comp: the header made ready between two draws that share the records)
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT,0,16};
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};li.pushConstantRangeCount=1;li.pPushConstantRanges=&push;
        VK_CHECK(vkCreatePipelineLayout(vk.device,&li,nullptr,&f.advance_layout));
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};mi.codeSize=sizeof mesh_advance_spv;mi.pCode=mesh_advance_spv;
        VkShaderModule module;VK_CHECK(vkCreateShaderModule(vk.device,&mi,nullptr,&module));
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};ci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=module;ci.stage.pName="main";ci.layout=f.advance_layout;
        VK_CHECK(vkCreateComputePipelines(vk.device,PipelineCache(),1,&ci,nullptr,&f.advance));
        vkDestroyShaderModule(vk.device,module,nullptr);
    }
    {   // (the draws' commands: copied from the records' header on the GPU, read by the draws, and by the CPU afterwards
        // for how many records each draw took)
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=u64(MeshFallbackSlots)*16;
        bi.usage=VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VK_CHECK(vkCreateBuffer(vk.device,&bi,nullptr,&f.commands));
        VkMemoryRequirements req;vkGetBufferMemoryRequirements(vk.device,f.commands,&req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=req.size;
        ai.memoryTypeIndex=FindMemoryType(req.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory memory;VK_CHECK(vkAllocateMemory(vk.device,&ai,nullptr,&memory));VK_CHECK(vkBindBufferMemory(vk.device,f.commands,memory,0));
        void* map=nullptr;VK_CHECK(vkMapMemory(vk.device,memory,0,VK_WHOLE_SIZE,0,&map));std::memset(map,0,bi.size);f.commands_map=static_cast<const u32*>(map);
    }
    RT_LOG("mesh fallback buffers: records %llu MiB, payloads %llu MiB",(unsigned long long)(MeshFallbackRecordsBytes>>20),(unsigned long long)(MeshFallbackPayloadBytes>>20));
}
// The block every pipeline's push constants start with (the translator's rescaling words, the guest page table), and for
// the mesh fallback its buffers: the payloads, the records, a task program's first workgroup or the guest address its
// indirect arguments are read from.
void FillPushConstants(float push[32],u32 mesh_first=0,u64 indirect_input=0){
    std::memset(push,0,32*sizeof(float));push[8]=1;
    {const u64 pages=GuestPageTableAddress();std::memcpy((char*)push+GuestPagesPushOffset,&pages,8);}
    if(mesh_fallback.records){
        std::memcpy((char*)push+nvshader::MeshFirstPushOffset,&mesh_first,4);
        std::memcpy((char*)push+nvshader::MeshFallbackPayloadPushOffset,&mesh_fallback.payload_address,8);
        std::memcpy((char*)push+nvshader::MeshFallbackRecordsPushOffset,&mesh_fallback.records_address,8);
        std::memcpy((char*)push+nvshader::MeshFallbackIndirectPushOffset,&indirect_input,8);}
}
std::shared_mutex pipeline_tables_mutex; // pipelines and compute_pipelines: the executor and the prewarm threads (a hit is a lookup under it)
std::atomic<u32> prewarm_hits{0};  // pipelines made before the game that play needed (since the last report)
Pipeline& GetPipelineOld(bool mesh) {
    std::vector<ShaderVariant*> active;
    ShaderVariant* previous=nullptr;u32 binding=0;
    const std::initializer_list<u32> order=mesh?std::initializer_list<u32>{6u,7u,1u}:std::initializer_list<u32>{0u,3u,4u,2u,1u};
    for(u32 i:order) if((*tls_state).stages[i].shader) {
        auto& v=CompileShader(*(*tls_state).stages[i].shader,binding,previous);previous=&v;active.push_back(&v);
        for(auto& r:v.compiled.resources)binding=std::max(binding,r.binding+(r.kind==nvshader::ResourceKind::UniformBuffer||r.kind==nvshader::ResourceKind::StorageBuffer?r.count:1));
    }
    static Pipeline untranslatable;
    for(auto* v:active)if(!v->module)return untranslatable;
    if(!mesh&&!(*tls_state).stages[0].shader)Fatal("draw has no vertex shader");
    if(mesh&&!(*tls_state).stages[7].shader)Fatal("mesh draw has no mesh shader");
    static thread_local std::string key;key.clear();
    const auto add=[&](const auto& value){key.append(reinterpret_cast<const char*>(&value),sizeof value);};
    // Only the state bits pipeline creation reads below: the raw NVN words also carry unrelated
    // (often uninitialized) bits, which would make every draw look like a new pipeline.
    for(u32 i=0;i<32;++i){const u32 a=(*tls_state).attributes[i];add(a&0x40?a&(31u|0x40u|0x3fffu<<7|0x3ffu<<21):0u);}
    {std::array<bool,16> used{};for(u32 i=0;i<32;++i)if((*tls_state).attributes[i]&0x40)used[(*tls_state).attributes[i]&31]=true;
     for(u32 i=0;i<16;++i){add(used[i]?(*tls_state).streams[i][0]:0u);add(used[i]?(*tls_state).streams[i][1]:0u);}}
    add((*tls_state).color_count);
    for(u32 i=0;i<(*tls_state).color_count;++i){const bool blended=((*tls_state).color>>(8+i))&1;add(blended);add(blended?(*tls_state).blends[i]>>24:0ull);add((*tls_state).channel>>(i*4)&15);}
    add((*tls_state).polygon&(0x1fu|0xfu<<15));add(WindowOriginLowerLeft());
    add((*tls_state).depth&0xf7);add(((*tls_state).depth>>2)&1?(*tls_state).depth>>32:0ull);
    add((*tls_state).depth_bounds);if((*tls_state).depth_bounds)add((*tls_state).depth_bounds_range);
    add((*tls_state).primitive);add((*tls_state).restart);add((*tls_state).depth_clamp);add((*tls_state).discard);add((*tls_state).sample_mask);add(mesh);
    std::array<VkFormat,8> colors{};VkFormat depth=VK_FORMAT_UNDEFINED,stencil=VK_FORMAT_UNDEFINED;u32 samples=1;
    for(u32 i=0;i<(*tls_state).color_count;++i)if((*tls_state).targets.colors[i]){auto* t=HostTexture((*tls_state).targets.colors[i]);colors[i]=(*tls_state).targets.color_views[i].format?GetFormat((*tls_state).targets.color_views[i].format).vk:t->info.vk;samples=t->samples;}
    if((*tls_state).targets.depth){auto* t=HostTexture((*tls_state).targets.depth);if(t->info.depth)depth=t->info.vk;if(t->info.stencil)stencil=t->info.vk;samples=t->samples;}
    add(colors);add(depth);add(stencil);add(samples);for(auto* shader:active)add(shader->module);
    auto found=pipelines.find(key);if(found!=pipelines.end())return *found->second;
    auto pipeline=std::make_unique<Pipeline>();pipeline->shaders=active;auto& vk=Vk();
    for(auto* shader:active)for(const auto& r:shader->compiled.resources){
        const bool buffer=r.kind==nvshader::ResourceKind::UniformBuffer||r.kind==nvshader::ResourceKind::StorageBuffer;
        pipeline->bindings.push_back({r.binding,DescriptorType(r.kind),buffer?1:r.count,ShaderVkStage[shader->source->stage],nullptr});
    }
    VkDescriptorSetLayoutCreateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};di.bindingCount=pipeline->bindings.size();di.pBindings=pipeline->bindings.data();
    {u32 total=0;for(const auto& b:pipeline->bindings)total+=b.descriptorCount;
     if(vk.push_descriptor_set&&total<=vk.max_push_descriptors){di.flags=VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;pipeline->push=true;}}
    VK_CHECK(vkCreateDescriptorSetLayout(vk.device,&di,nullptr,&pipeline->descriptors));
    VkPushConstantRange push{VK_SHADER_STAGE_ALL,0,128};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};li.setLayoutCount=1;li.pSetLayouts=&pipeline->descriptors;li.pushConstantRangeCount=1;li.pPushConstantRanges=&push;
    VK_CHECK(vkCreatePipelineLayout(vk.device,&li,nullptr,&pipeline->layout));
    // Pipeline compilation runs on a worker from a snapshot of the state; the draw waits briefly
    // and is skipped (this frame only) if the pipeline is not ready, instead of stalling the frame.
    {
    auto snapshot=std::make_shared<State>((*tls_state));
    const bool lower_left=WindowOriginLowerLeft();
    Pipeline* target=pipeline.get();
    auto job=[snapshot,lower_left,target,active,colors,depth,stencil,samples,mesh]{
        const State& st=*snapshot;auto& vk=Vk();
        std::vector<VkPipelineShaderStageCreateInfo> stages;
        for(auto* shader:active){VkPipelineShaderStageCreateInfo si{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};si.stage=ShaderVkStage[shader->source->stage];si.module=shader->module;si.pName="main";stages.push_back(si);}
        std::vector<VkVertexInputAttributeDescription> attributes;
        std::vector<VkVertexInputBindingDescription> vertex_bindings;
        std::array<bool,16> used{};
        for(u32 i=0;i<32;++i){const u32 a=st.attributes[i];if(!(a&0x40))continue;const u32 stream=a&31;if(stream>=16)Fatal("vertex stream exceeds native limit");
            attributes.push_back({i,stream,GetFormat(a>>21&0x3ff).vk,(a>>7)&0x3fff});used[stream]=true;}
        for(u32 i=0;i<16;++i)if(used[i]){if(st.streams[i][1]>1)Fatal("vertex instance divisor %u not enabled",st.streams[i][1]);vertex_bindings.push_back({i,st.streams[i][0],st.streams[i][1]?VK_VERTEX_INPUT_RATE_INSTANCE:VK_VERTEX_INPUT_RATE_VERTEX});}
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};vi.vertexAttributeDescriptionCount=attributes.size();vi.pVertexAttributeDescriptions=attributes.data();vi.vertexBindingDescriptionCount=vertex_bindings.size();vi.pVertexBindingDescriptions=vertex_bindings.data();
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};ia.topology=Topology(st.primitive);ia.primitiveRestartEnable=st.restart;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};vp.viewportCount=1;vp.scissorCount=1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};rs.cullMode=st.polygon&3;{static const char* nocull=std::getenv("OOT_CULL_NONE");if(nocull&&st.stages[1].shader){char a[24];std::snprintf(a,sizeof a,"%llx",(unsigned long long)st.stages[1].shader->address);if(std::strstr(nocull,a))rs.cullMode=VK_CULL_MODE_NONE;}}
        // Lower-left origin keeps NVN's memory layout with an unflipped Vulkan viewport, which mirrors
        // the window y axis and with it the winding: OpenGL's counter-clockwise is Vulkan's clockwise.
        const bool ccw=(st.polygon&4)!=0;rs.frontFace=ccw!=lower_left?VK_FRONT_FACE_COUNTER_CLOCKWISE:VK_FRONT_FACE_CLOCKWISE;
        const VkPolygonMode modes[]={VK_POLYGON_MODE_POINT,VK_POLYGON_MODE_LINE,VK_POLYGON_MODE_FILL};u32 mode=st.polygon>>3&3;if(mode>2)Fatal("invalid polygon mode");rs.polygonMode=modes[mode];rs.lineWidth=1;rs.depthBiasEnable=(st.polygon>>15&15)!=0;rs.depthClampEnable=st.depth_clamp;rs.rasterizerDiscardEnable=st.discard;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};ms.rasterizationSamples=(VkSampleCountFlagBits)samples;const VkSampleMask mask=st.sample_mask;ms.pSampleMask=&mask;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};ds.depthTestEnable=st.depth&1;ds.depthWriteEnable=(st.depth>>1)&1;ds.depthCompareOp=(VkCompareOp)(((st.depth>>4)&15)-1);{static const bool relax=std::getenv("OOT_DEPTH_EQUAL_RELAX")!=nullptr;if(relax&&ds.depthCompareOp==VK_COMPARE_OP_EQUAL)ds.depthCompareOp=VK_COMPARE_OP_LESS_OR_EQUAL;}ds.stencilTestEnable=(st.depth>>2)&1;{static const char* always=std::getenv("OOT_DEPTH_ALWAYS");if(always&&st.stages[1].shader){char a[24];std::snprintf(a,sizeof a,"%llx",(unsigned long long)st.stages[1].shader->address);if(std::strstr(always,a))ds.depthTestEnable=VK_FALSE;}}ds.depthBoundsTestEnable=st.depth_bounds;ds.minDepthBounds=st.depth_bounds_range[0];ds.maxDepthBounds=st.depth_bounds_range[1];
        for(u32 i=0;i<2;++i){const u32 p=(u32)(st.depth>>32)>>(i*16);auto& st=i?ds.back:ds.front;st.compareOp=(VkCompareOp)((p&15)-1);st.failOp=(VkStencilOp)(((p>>4)&15)-1);st.depthFailOp=(VkStencilOp)(((p>>8)&15)-1);st.passOp=(VkStencilOp)(((p>>12)&15)-1);}
        std::vector<VkPipelineColorBlendAttachmentState> blend(st.color_count);
        for(u32 i=0;i<blend.size();++i){const u64 b=st.blends[i];auto& out=blend[i];static const bool no_blend=std::getenv("OOT_DEBUG_NOBLEND")!=nullptr;out.blendEnable=!no_blend&&((st.color>>(8+i))&1);out.colorWriteMask=st.channel>>(i*4)&15;
            out.srcColorBlendFactor=BlendFactor(b>>32&127);out.dstColorBlendFactor=BlendFactor(b>>40&127);out.srcAlphaBlendFactor=BlendFactor(b>>48&127);out.dstAlphaBlendFactor=BlendFactor(b>>56&127);
            const VkBlendOp ops[]={VK_BLEND_OP_ADD,VK_BLEND_OP_SUBTRACT,VK_BLEND_OP_REVERSE_SUBTRACT,VK_BLEND_OP_MIN,VK_BLEND_OP_MAX};
            const u32 rgb=b>>24&7,alpha=b>>28&7;if(rgb<1||rgb>5||alpha<1||alpha>5)Fatal("invalid blend equation");out.colorBlendOp=ops[rgb-1];out.alphaBlendOp=ops[alpha-1];}
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};cb.attachmentCount=blend.size();cb.pAttachments=blend.data();
        const VkDynamicState dynamics[]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR,VK_DYNAMIC_STATE_BLEND_CONSTANTS,VK_DYNAMIC_STATE_DEPTH_BIAS,VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,VK_DYNAMIC_STATE_STENCIL_REFERENCE};
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};dy.dynamicStateCount=std::size(dynamics);dy.pDynamicStates=dynamics;
        VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};ri.colorAttachmentCount=colors.size();ri.pColorAttachmentFormats=colors.data();ri.depthAttachmentFormat=depth;ri.stencilAttachmentFormat=stencil;ri.colorAttachmentCount=st.color_count;
        VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};pi.pNext=&ri;StagePins pinned;const auto& pinned_stages=pinned.Apply(stages);pi.stageCount=pinned_stages.size();pi.pStages=pinned_stages.data();pi.pVertexInputState=mesh?nullptr:&vi;pi.pInputAssemblyState=mesh?nullptr:&ia;pi.pViewportState=&vp;pi.pRasterizationState=&rs;pi.pMultisampleState=&ms;pi.pDepthStencilState=&ds;pi.pColorBlendState=&cb;pi.pDynamicState=&dy;pi.layout=target->layout;
        {const auto t0=std::chrono::steady_clock::now();
        VkPipeline created=VK_NULL_HANDLE;
        VK_CHECK(vkCreateGraphicsPipelines(vk.device,PipelineCache(),1,&pi,nullptr,&created));
        target->pipeline.store(created,std::memory_order_release);
        std::lock_guard timing{compile_time_mutex};
        compile_time.pipelines+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();++compile_time.created;}
    };
    static const bool synchronous=std::getenv("OOT_SYNC_PIPELINES")!=nullptr;
    if(synchronous)job();
    else{
        PipelineWorkers().Run(std::move(job),target);
        const auto waited_from=std::chrono::steady_clock::now();
        // (a quick one is there in a moment and its draw need not be left out: up to 3 ms, but no more than 12 ms a
        // frame for all of them - a new area's thousand pipelines were 3 s of these waits)
        if(QuickWaitAllowed()){
            const auto deadline=waited_from+std::chrono::microseconds(3000);
            while(!target->pipeline.load(std::memory_order_acquire)&&std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
            const double waited=std::chrono::duration<double>(std::chrono::steady_clock::now()-waited_from).count();
            hitch.pipeline_wait+=waited;quick_wait_us.fetch_add((u64)(waited*1e6),std::memory_order_relaxed);
        }
    }
    }
    auto* result=pipeline.get();pipelines.emplace(std::move(key),std::move(pipeline));return *result;
}
// Whether two variants name the same resources in the same places (their translations can be the same code while
// their texture handles sit at other offsets): a draw may then resolve its descriptors from either.
u64 ResourceHash(const ShaderVariant& v){
    u64 h=v.resource_hash.load(std::memory_order_relaxed);
    if(h)return h;
    h=0x9E3779B97F4A7C15ull^v.source->stage;
    const auto mix=[&](u64 x){h=(h^x)*0xff51afd7ed558ccdull;h^=h>>32;};
    for(const auto& r:v.compiled.resources){
        for(u32 x:{(u32)r.kind,r.binding,r.count,r.bank,r.offset,r.secondary_bank,r.secondary_offset,r.shift,r.secondary_shift,r.size_shift,r.texture_type,r.image_format,r.indirect_offset,r.storage_index,r.storage_offset,r.storage_stride,
                   (u32)r.sized_storage,(u32)r.secondary,(u32)r.depth,(u32)r.multisample,(u32)r.integer,(u32)r.written})mix(x);
        for(u32 w:r.handle_program)mix(w);mix(r.handle_program.size());}
    mix(v.compiled.global_loads);mix(v.compiled.global_stores);
    h|=1;v.resource_hash.store(h,std::memory_order_relaxed);
    return h;
}
bool SameResources(const std::vector<ShaderVariant*>& a,const std::vector<ShaderVariant*>& b){
    if(a.size()!=b.size())return false;
    for(size_t i=0;i<a.size();++i)if(a[i]!=b[i]&&ResourceHash(*a[i])!=ResourceHash(*b[i]))return false;
    return true;
}
Pipeline& GetPipeline(bool mesh);
Pipeline& GetPipeline(bool mesh) {
    {static const bool old=std::getenv("OOT_OLD_PIPELINE")!=nullptr;if(old)return GetPipelineOld(mesh);}
    std::vector<ShaderVariant*> active;
    ShaderVariant* previous=nullptr;u32 binding=0;
    const std::initializer_list<u32> order=mesh?std::initializer_list<u32>{6u,7u,1u}:std::initializer_list<u32>{0u,3u,4u,2u,1u};
    for(u32 i:order) if((*tls_state).stages[i].shader) {
        auto& v=CompileShader(*(*tls_state).stages[i].shader,binding,previous);previous=&v;active.push_back(&v);
#ifdef OOT_SHADER_LONGSHOT
        for(auto& r:v.compiled.resources)binding=std::max(binding,r.binding+1); // (every resource one binding: arrays and storage tables too)
#else
        for(auto& r:v.compiled.resources)binding=std::max(binding,r.binding+(r.kind==nvshader::ResourceKind::UniformBuffer||r.kind==nvshader::ResourceKind::StorageBuffer?r.count:1));
#endif
    }
    static Pipeline untranslatable;
    if(resolve_bail)return untranslatable;
    for(auto* v:active)if(!v->module)return untranslatable;
    if(!mesh&&!(*tls_state).stages[0].shader)Fatal("draw has no vertex shader");
    if(mesh&&!(*tls_state).stages[7].shader)Fatal("mesh draw has no mesh shader");
    std::array<VkFormat,8> colors{};VkFormat depth=VK_FORMAT_UNDEFINED,stencil=VK_FORMAT_UNDEFINED;u32 samples=1;
    for(u32 i=0;i<(*tls_state).color_count;++i)if((*tls_state).targets.colors[i]){auto* t=HostTexture((*tls_state).targets.colors[i]);colors[i]=(*tls_state).targets.color_views[i].format?GetFormat((*tls_state).targets.color_views[i].format).vk:t->info.vk;samples=t->samples;}
    if((*tls_state).targets.depth){auto* t=HostTexture((*tls_state).targets.depth);if(t->info.depth)depth=t->info.vk;if(t->info.stencil)stencil=t->info.vk;samples=t->samples;}
    // Only the state bits pipeline creation reads: the raw NVN words also carry unrelated (often uninitialized) bits,
    // which would make every draw look like a new pipeline. Shaders count by what they are, not by this run's handles.
    const bool fallback=mesh&&Vk().mesh_fallback;
    MeshFallbackStages fallback_stages;
    const PipelineDesc desc=[&]{PipelineDesc d=MakeDesc(mesh,colors,depth,stencil,samples);
        if(fallback)for(auto* shader:active){const u32 st=shader->source->stage;
            if(st==6){fallback_stages.task=shader->module;fallback_stages.task_threads=shader->compiled.mesh_threads;}
            if(st==7){fallback_stages.mesh=shader->module;fallback_stages.vertex=shader->fallback_vertex_module;fallback_stages.mesh_threads=shader->compiled.mesh_threads;MeshFallbackDesc(d,shader->compiled.mesh_topology);}
            if(st==1)fallback_stages.fragment=shader->module;}
        return d;}();
    if(fallback&&!fallback_stages.vertex)return untranslatable;
    std::vector<u64> identities;for(auto* shader:active)identities.push_back(shader->identity);
    std::string key=PipelineKey(desc,identities);
    {static const bool by_module=std::getenv("OOT_PIPELINE_KEY_MODULE")!=nullptr;if(by_module)for(auto* shader:active)key.append(reinterpret_cast<const char*>(&shader->module),sizeof shader->module);} // (diagnostic: one pipeline per module, as before)
    if(resolve_worker){ // (threads resolving draws only look up, together)
        // (each thread remembers the pipelines it met: a reader-writer lock taken by every draw of eight threads was
        // slower than the lookups it guarded; pipelines stay for the whole run)
        static thread_local std::unordered_map<std::string,Pipeline*,PipelineKeyHash> met;
        Pipeline* known=nullptr;
        if(const auto it=met.find(key);it!=met.end())known=it->second;
        else{std::shared_lock reading{pipeline_tables_mutex};
            const auto in_table=pipelines.find(key);
            if(in_table!=pipelines.end()&&in_table->second->touched){known=in_table->second.get();met.emplace(key,known);}}
        if(known&&(known->shaders==active||SameResources(known->shaders,active)))return *known;
        RESOLVE_BAIL();return untranslatable; // (a new pipeline, or one made before the game and not used yet: the executor's)
    }
    std::unique_lock tables{pipeline_tables_mutex};
    auto found=pipelines.find(key);
    if(found!=pipelines.end()){
        Pipeline& hit=*found->second;
        if(!hit.touched){hit.touched=true;++prewarm_hits;plist::Use(hit.record);} // (made before the game: now it is in use)
        if(hit.shaders!=active){if(!resolve_worker)hit.shaders=active;else if(!SameResources(hit.shaders,active)){RESOLVE_BAIL();return untranslatable;}} // (the variants this draw uses: another run's, or another program with the same code)
        return hit;
    }
    if(resolve_worker){RESOLVE_BAIL();return untranslatable;} // (a new pipeline: made by the executor)
    auto pipeline=std::make_unique<Pipeline>();pipeline->shaders=active;
    {std::vector<StageResources> stages;for(auto* shader:active)stages.emplace_back(BindingStage(shader->source->stage,fallback),&shader->compiled.resources);BuildPipelineLayout(*pipeline,stages);}
    if(fallback){pipeline->fallback=true;pipeline->fallback_task_threads=fallback_stages.task?std::max(fallback_stages.task_threads,1u):0;
        for(auto* shader:active)if(shader->source->stage==7){pipeline->fallback_payload_words=shader->compiled.mesh_payload_words;pipeline->fallback_record_words=shader->compiled.mesh_record_words;}}
    // Pipeline compilation runs on a worker; the draw waits briefly and is skipped (this frame only) if the
    // pipeline is not ready, instead of stalling the frame.
    {
    plist::Record record; // (what the next run needs to make this one before the game: pipeline_list.h)
    bool listed=plist::Enabled();for(auto* shader:active)listed&=shader->cache_key!=0;
    if(listed){
        record.kind=mesh?plist::Kind::Mesh:plist::Kind::Graphics;
        for(auto* shader:active)record.shaders.push_back({shader->source->stage,shader->cache_key,shader->identity});
        record.state.assign(reinterpret_cast<const u8*>(&desc),reinterpret_cast<const u8*>(&desc)+sizeof desc);
        record.id=plist::MakeId(record.kind,record.shaders,record.state);pipeline->record=record.id;
    }
    const u64 fragment_address=(*tls_state).stages[1].shader?(*tls_state).stages[1].shader->address:0;
    Pipeline* target=pipeline.get();
    auto job=[desc,target,active,fragment_address,listed,fallback,fallback_stages,record=std::move(record)]{
        std::vector<VkPipelineShaderStageCreateInfo> stages;
        for(auto* shader:active){VkPipelineShaderStageCreateInfo si{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};si.stage=ShaderVkStage[shader->source->stage];si.module=shader->module;si.pName="main";stages.push_back(si);}
        const auto t0=std::chrono::steady_clock::now();
        VkPipeline created=VK_NULL_HANDLE;
        if(fallback){VkPipeline made[3];CreateMeshFallbackPipelines(desc,target->layout,fallback_stages,fragment_address,true,made);
            target->task_compute=made[1];target->mesh_compute=made[2];created=made[0];}
        else created=CreateGraphicsPipeline(desc,target->layout,stages,fragment_address,true);
        target->pipeline.store(created,std::memory_order_release);
        {std::lock_guard timing{compile_time_mutex};
        compile_time.pipelines+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();++compile_time.created;}
        if(listed&&!plist::Add(record))plist::Use(record.id);
    };
    static const bool synchronous=std::getenv("OOT_SYNC_PIPELINES")!=nullptr;
    if(synchronous)job();
    else{
        PipelineWorkers().Run(std::move(job),target);
        const auto waited_from=std::chrono::steady_clock::now();
        // (a quick one is there in a moment and its draw need not be left out: up to 3 ms, but no more than 12 ms a
        // frame for all of them - a new area's thousand pipelines were 3 s of these waits)
        if(QuickWaitAllowed()){
            const auto deadline=waited_from+std::chrono::microseconds(3000);
            while(!target->pipeline.load(std::memory_order_acquire)&&std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
            const double waited=std::chrono::duration<double>(std::chrono::steady_clock::now()-waited_from).count();
            hitch.pipeline_wait+=waited;quick_wait_us.fetch_add((u64)(waited*1e6),std::memory_order_relaxed);
        }
    }
    }
    auto* result=pipeline.get();pipelines.emplace(std::move(key),std::move(pipeline));return *result;
}
u64 ShaderStorageAddress(const ShaderVariant& shader,u32 index) {
    auto& stage=(*tls_state).stages[ShaderPublicStage[shader.source->stage]];
    for(const auto& resource:shader.compiled.resources)if(resource.kind==nvshader::ResourceKind::StorageBuffer) {
        if(index--!=0)continue;
        u64 address=Cbuf(&stage,resource.bank,resource.offset)|(u64)Cbuf(&stage,resource.bank,resource.offset+4)<<32;
        if(resource.indirect_offset!=~0u)address=ReadPointer(nullptr,address+resource.indirect_offset);
        return address;
    }
    Fatal("texture handle expression uses an unknown storage descriptor");
}
u32 ResourceHandle(const nvshader::Resource& r,ShaderVariant& shader,u32 element) {
    auto& stage=(*tls_state).stages[ShaderPublicStage[shader.source->stage]];
    if(r.storage_index!=~0u) {
        const u64 address=ShaderStorageAddress(shader,r.storage_index);
        const auto range=LookupAddress(address);
        const u64 offset=u64(r.storage_offset)+u64(element)*r.storage_stride;
        return range.pool&&offset+4<=range.available?ReadWord(address+offset):0;
    }
    if(!r.handle_program.empty()) {
        struct Context { StageState* stage; const ShaderVariant* shader; } context{&stage,&shader};
        return nvshader::EvaluateHandle(r.handle_program,&context,
            [](void* p,u32 bank,u32 offset){return Cbuf(static_cast<Context*>(p)->stage,bank,offset);},
            [](void* p,u32 index,u32 offset){
                auto& c=*static_cast<Context*>(p);
                // IR storage indices name the compact shader descriptor list, not
                // the application's NVN storage slots.
                const u64 address=ShaderStorageAddress(*c.shader,index);
                const u64 alignment=Vk().properties.limits.minStorageBufferOffsetAlignment;
                return ReadWord((address&~(alignment-1))+offset);
#ifdef OOT_SHADER_LONGSHOT
            },element); // (a handle table's element is computed by the program itself: HandleOp::Index)
#else
            })+element;
#endif
    }
    // A handle word where a storage buffer of the same program keeps its address is no handle (from Hookshot
    // 3858629 / Fritchii d603d98bb): OoT's water loads each light's handle per pixel from the light buffer, and the
    // translation falls back to that buffer's address, whose low bits named a different texture each frame (the
    // water flickered). No texture there. OOT_HANDLE_POINTER=1: the old handle.
    static const bool handle_pointer=std::getenv("OOT_HANDLE_POINTER")!=nullptr;
    if(!handle_pointer&&!r.secondary){
        const u32 at=r.offset+(element<<r.size_shift);
        for(const auto& sb:shader.compiled.resources)
            if(sb.kind==nvshader::ResourceKind::StorageBuffer&&sb.indirect_offset==~0u&&sb.storage_index==~0u&&sb.bank==r.bank&&at>=sb.offset&&at<sb.offset+8)return 0;
    }
    u32 value=Cbuf(&stage,r.bank,r.offset+(element<<r.size_shift))<<r.shift;
    if(r.secondary)value|=Cbuf(&stage,r.secondary_bank,r.secondary_offset+(element<<r.size_shift))<<r.secondary_shift;
    return value;
}
// Writes a color texture (any blittable format, mip 0 / layer 0) scaled to 640x360 as a PPM.
void DumpTexture(Texture& t, const std::string& path) {
    auto& vk=Vk();constexpr u32 W=640,H=360;
    VkFormatProperties fp;vkGetPhysicalDeviceFormatProperties(vk.physical,t.decoded_astc?VK_FORMAT_R8G8B8A8_UNORM:t.info.vk,&fp);
    if(!(fp.optimalTilingFeatures&VK_FORMAT_FEATURE_BLIT_SRC_BIT)||t.info.depth||t.info.stencil){RT_LOG("  (cannot dump format %u)",t.format);return;}
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};ii.imageType=VK_IMAGE_TYPE_2D;ii.format=VK_FORMAT_R8G8B8A8_UNORM;ii.extent={W,H,1};ii.mipLevels=1;ii.arrayLayers=1;
    ii.samples=VK_SAMPLE_COUNT_1_BIT;ii.tiling=VK_IMAGE_TILING_OPTIMAL;ii.usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VkImage image;VK_CHECK(vkCreateImage(vk.device,&ii,nullptr,&image));VkMemoryRequirements req;vkGetImageMemoryRequirements(vk.device,image,&req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=req.size;ai.memoryTypeIndex=FindMemoryType(req.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory memory;VK_CHECK(vkAllocateMemory(vk.device,&ai,nullptr,&memory));VK_CHECK(vkBindImageMemory(vk.device,image,memory,0));
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=W*H*4;bi.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT;VkBuffer buffer;VK_CHECK(TracedCreateBuffer(&bi,&buffer,__FILE__,__LINE__));
    vkGetBufferMemoryRequirements(vk.device,buffer,&req);ai.allocationSize=req.size;ai.memoryTypeIndex=FindMemoryType(req.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory host;VK_CHECK(vkAllocateMemory(vk.device,&ai,nullptr,&host));VK_CHECK(vkBindBufferMemory(vk.device,buffer,host,0));
    Immediate([&](VkCommandBuffer cmd){
        Transition(cmd,t,VK_IMAGE_LAYOUT_GENERAL);
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};b.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED;b.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;b.image=image;b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};b.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&b);
        VkImageBlit blit{};blit.srcSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};blit.srcOffsets[1]={(s32)t.width,(s32)t.height,1};blit.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};blit.dstOffsets[1]={(s32)W,(s32)H,1};
        vkCmdBlitImage(cmd,t.image,VK_IMAGE_LAYOUT_GENERAL,image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&blit,VK_FILTER_LINEAR);
        b.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;b.newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;b.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;b.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&b);
        VkBufferImageCopy copy{};copy.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};copy.imageExtent={W,H,1};vkCmdCopyImageToBuffer(cmd,image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,buffer,1,&copy);});
    Flush();
    u8* px;VK_CHECK(vkMapMemory(vk.device,host,0,W*H*4,0,reinterpret_cast<void**>(&px)));
    if(FILE* f=std::fopen(path.c_str(),"wb")){std::fprintf(f,"P6\n%u %u\n255\n",W,H);for(u32 i=0;i<W*H;++i)std::fwrite(px+i*4,1,3,f);std::fclose(f);}
    vkUnmapMemory(vk.device,host);vkDestroyBuffer(vk.device,buffer,nullptr);vkFreeMemory(vk.device,host,nullptr);vkDestroyImage(vk.device,image,nullptr);vkFreeMemory(vk.device,memory,nullptr);
}
// OOT_DUMP_RAW=<target hex>:<x>,<y>,<w>,<h> (with OOT_DRAW_DUMP, diagnostic): after each logged draw into that colour
// target, the rectangle's texels exactly as stored (raw-<n>.bin: a "w h format bytes" line, then the rows).
void DumpRaw(Texture& t,u32 n,const char* dir){
    static u64 target=0;static u32 rx=0,ry=0,rw=0,rh=0;static bool parsed=false;
    if(!parsed){parsed=true;if(const char* e=std::getenv("OOT_DUMP_RAW"))std::sscanf(e,"%llx:%u,%u,%u,%u",reinterpret_cast<unsigned long long*>(&target),&rx,&ry,&rw,&rh);}
    if(!target||t.object!=target||!t.image||t.info.depth||t.info.stencil||t.info.block_w!=1)return;
    const u32 w=std::min(rw,t.width>rx?t.width-rx:0),h=std::min(rh,t.height>ry?t.height-ry:0);if(!w||!h)return;
    auto& vk=Vk();const u64 size=(u64)w*h*t.info.bytes;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=size;bi.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT;VkBuffer buffer;VK_CHECK(TracedCreateBuffer(&bi,&buffer,__FILE__,__LINE__));
    VkMemoryRequirements req;vkGetBufferMemoryRequirements(vk.device,buffer,&req);VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=req.size;
    ai.memoryTypeIndex=FindMemoryType(req.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);VkDeviceMemory host;VK_CHECK(vkAllocateMemory(vk.device,&ai,nullptr,&host));VK_CHECK(vkBindBufferMemory(vk.device,buffer,host,0));
    Immediate([&](VkCommandBuffer cmd){Transition(cmd,t,VK_IMAGE_LAYOUT_GENERAL);
        VkBufferImageCopy copy{};copy.imageSubresource={AspectOf(t),0,0,1};copy.imageOffset={(s32)rx,(s32)ry,0};copy.imageExtent={w,h,1};
        vkCmdCopyImageToBuffer(cmd,t.image,VK_IMAGE_LAYOUT_GENERAL,buffer,1,&copy);});
    Flush();
    u8* px;VK_CHECK(vkMapMemory(vk.device,host,0,size,0,reinterpret_cast<void**>(&px)));
    char name[64];std::snprintf(name,sizeof name,"/raw-%04u.bin",n);
    if(FILE* f=std::fopen((std::string(dir)+name).c_str(),"wb")){std::fprintf(f,"%u %u %u %u\n",w,h,t.format,t.info.bytes);std::fwrite(px,1,size,f);std::fclose(f);}
    vkUnmapMemory(vk.device,host);vkDestroyBuffer(vk.device,buffer,nullptr);vkFreeMemory(vk.device,host,nullptr);
}
void DumpTargets(const char* what) {
    static const char* dir=std::getenv("OOT_DRAW_DUMP");static u32 n=0;if(!dir)return;
    for(u32 i=0;i<(*tls_state).color_count;++i)if((*tls_state).targets.colors[i])DumpRaw(*HostTexture((*tls_state).targets.colors[i]),n,dir);
    for(u32 i=0;i<(*tls_state).color_count;++i)if((*tls_state).targets.colors[i]){char name[96];std::snprintf(name,sizeof name,"/%03u-%s-c%u-%llx.ppm",n,what,i,(unsigned long long)(*tls_state).targets.colors[i]);DumpTexture(*HostTexture((*tls_state).targets.colors[i]),dir+std::string(name));}
    ++n;
}
void LogDraw(const Command& command) {
    if((*tls_state).stages[6].shader){auto& st=(*tls_state).stages[6];const auto word=[&](u32 u,u32 off)->long long{const auto& b=st.uniforms[u];return b.address&&off+4<=b.size?(long long)ReadWord(b.address+off):-1;};
        RT_LOG("  task constants: c3[0x420]=%lld c3[0x428]=%lld c4[0x5c]=%#llx (c3 @%#llx c4 @%#llx)",word(0,0x420),word(0,0x428),word(1,0x5c),(unsigned long long)st.uniforms[0].address,(unsigned long long)st.uniforms[1].address);}
    if((*tls_state).stages[7].shader)for(u32 i=0;i<4;++i){const auto& u=(*tls_state).stages[7].uniforms[i];if(!u.address)continue;const auto range=LookupAddress(u.address);if(!range.pool)continue; // (the mesh stage's blocks: hashed whole, first words shown)
        const u64 n=std::min<u64>(std::min<u64>(u.size,range.available),0x4000);u64 h=1469598103934665603ull;const u8* b=reinterpret_cast<const u8*>(range.cpu_address);for(u64 k=0;k<n;++k)h=(h^b[k])*1099511628211ull;
        std::string words;for(u32 k=0;k<24&&k*4+4<=n;++k){float f;std::memcpy(&f,b+k*4,4);char t[24];std::snprintf(t,sizeof t," %g",f);words+=t;}
        RT_LOG("  mesh uniform %u @%#llx size %#llx hash %016llx:%s",i,(unsigned long long)u.address,(unsigned long long)u.size,(unsigned long long)h,words.c_str());}
    if((*tls_state).stages[7].shader)for(u32 i=0;i<16;++i){const auto& b=(*tls_state).stages[7].storage[i];if(b.address)RT_LOG("  mesh storage %u @%#llx size %#llx",i,(unsigned long long)b.address,(unsigned long long)b.size);}
    const auto* x=command.x;
    for(u32 st=0;st<2;++st)for(u32 i=0;i<16;++i){const auto& b=(*tls_state).stages[st].storage[i];if(!b.address)continue;const auto range=LookupAddress(b.address);
        std::string words;if(range.pool){const float* f=static_cast<const float*>(ResolveAddress(b.address,0x40));for(u32 k=0;k<16;++k){char t[24];std::snprintf(t,sizeof t," %g",f[k]);words+=t;}}
        RT_LOG("  stage %u storage %u @%#llx size %#llx mapped %d:%s",st,i,(unsigned long long)b.address,(unsigned long long)b.size,range.pool!=nullptr,words.c_str());}
    for(u32 st=0;st<2;++st)for(u32 i=0;i<8;++i){const auto& u=(*tls_state).stages[st].uniforms[i];if(!u.address)continue;const auto range=LookupAddress(u.address);
        std::string words;if(range.pool){const float* f=static_cast<const float*>(ResolveAddress(u.address,std::min<u64>(u.size,0x60)));for(u32 k=0;k<std::min<u64>(u.size,0x60)/4;++k){char b[24];std::snprintf(b,sizeof b," %g",f[k]);words+=b;}}
        RT_LOG("  stage %u uniform %u @%#llx size %#llx:%s",st,i,(unsigned long long)u.address,(unsigned long long)u.size,words.c_str());
        if(range.pool&&u.size<=0x100){const u64 a=u.address,n=std::min<u64>(u.size,0x40);bool zero=true;const u8* p=static_cast<const u8*>(ResolveAddress(a,n));for(u64 k=0;k<n;++k)zero&=p[k]==0;
            if(zero)AfterFlush([a,n,st,i,pool_flags=range.pool->flags,mirrored=range.pool->device_buffer!=VK_NULL_HANDLE]{const float* f=static_cast<const float*>(ResolveAddress(a,n));
                RT_LOG("  (stage %u uniform %u @%#llx pool flags %#x%s, after its batch: %g %g %g %g %g %g %g %g)",st,i,(unsigned long long)a,pool_flags,mirrored?" device copy":"",f[0],f[1],f[2],f[3],f[4],f[5],f[6],f[7]);});}}
    RT_LOG("  device modes: origin lower-left %d, depth near -w %d",WindowOriginLowerLeft(),DepthNearIsMinusW());
    RT_LOG("  raster: discard %d sample_mask %#x depth_clamp %d depth_bounds %d restart %d samples-target; depth bias factor %g units %g clamp %g",(*tls_state).discard,(*tls_state).sample_mask,(*tls_state).depth_clamp,(*tls_state).depth_bounds,(*tls_state).restart,(*tls_state).depth_bias[0],(*tls_state).depth_bias[1],(*tls_state).depth_bias[2]);
    for(u32 i=0;i<32;++i){const u32 a=(*tls_state).attributes[i];if(a&0x40)RT_LOG("  attribute %u: raw %#x stream %u offset %u format %u",i,a,a&31,(a>>7)&0x3fff,a>>21&0x3ff);}
    for(u32 i=0;i<2;++i){const auto& v=(*tls_state).vertices[i];if(!v.address)continue;const float* f=static_cast<const float*>(ResolveAddress(v.address,48));
        RT_LOG("  vertex stream %u @%#llx stride %u: %g %g %g %g %g %g",i,(unsigned long long)v.address,(*tls_state).streams[i][0],f[0],f[1],f[2],f[3],f[4],f[5]);
        const u8* raw=static_cast<const u8*>(ResolveAddress(v.address,96));std::string hex;for(u32 k=0;k<96;++k){char b[4];std::snprintf(b,sizeof b,"%02x",raw[k]);hex+=b;if(k%16==15)hex+=' ';}
        RT_LOG("  vertex bytes: %s",hex.c_str());}
    std::string shaders;for(u32 i=0;i<std::size((*tls_state).stages);++i)if(i!=5&&(*tls_state).stages[i].shader){char b[48];std::snprintf(b,sizeof b," s%u=%#llx",i,(unsigned long long)(*tls_state).stages[i].shader->address);shaders+=b;}
    std::string targets;for(u32 i=0;i<(*tls_state).color_count;++i)if((*tls_state).targets.colors[i]){auto* t=HostTexture((*tls_state).targets.colors[i]);char b[64];std::snprintf(b,sizeof b," c%u=%#llx(%ux%u f%u)",i,(unsigned long long)(*tls_state).targets.colors[i],t->width,t->height,t->format);targets+=b;}
    RT_LOG("draw op %u args %llu %llu %llu %llu %llu %llu %llu |%s |%s depth=%#llx vp(%g %g %g %g) sc(%d %d %u %u) blend0=%#llx color=%#x chan=%#x poly=%#x ds=%#llx",
        command.op,(unsigned long long)x[0],(unsigned long long)x[1],(unsigned long long)x[2],(unsigned long long)x[3],(unsigned long long)x[4],(unsigned long long)x[5],(unsigned long long)x[6],
        shaders.c_str(),targets.c_str(),(unsigned long long)(*tls_state).targets.depth,(*tls_state).viewport.x,(*tls_state).viewport.y,(*tls_state).viewport.width,(*tls_state).viewport.height,
        (*tls_state).scissor.offset.x,(*tls_state).scissor.offset.y,(*tls_state).scissor.extent.width,(*tls_state).scissor.extent.height,(unsigned long long)(*tls_state).blends[0],(*tls_state).color,(*tls_state).channel,(*tls_state).polygon,(unsigned long long)(*tls_state).depth);
}
// A 64 KiB zero-filled buffer bound in place of unbound uniform/storage slots.
VkDescriptorBufferInfo ZeroBuffer(){
    // (made once, by whichever thread asks first: draws are resolved on several)
    static const VkBuffer buffer=[]{VkBuffer buffer=VK_NULL_HANDLE;auto& vk=Vk();
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=0x10000;bi.usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        VK_CHECK(TracedCreateBuffer(&bi,&buffer,__FILE__,__LINE__));VkMemoryRequirements req;vkGetBufferMemoryRequirements(vk.device,buffer,&req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=req.size;ai.memoryTypeIndex=FindMemoryType(req.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory memory;VK_CHECK(vkAllocateMemory(vk.device,&ai,nullptr,&memory));VK_CHECK(vkBindBufferMemory(vk.device,buffer,memory,0));
        void* p;VK_CHECK(vkMapMemory(vk.device,memory,0,0x10000,0,&p));std::memset(p,0,0x10000);return buffer;}();
    return {buffer,0,0x10000};
}
// Whether CPU-side evaluation of this pipeline's memory-resident handles and pointers could read
// guest memory still being written by work in the open batch.
void ReportHandleWait(const char* what,u64 shader,u64 address,u64 size){
    static const bool trace=std::getenv("OOT_WAIT_TRACE")!=nullptr;if(!trace)return;
    static std::map<std::pair<u64,const char*>,bool> reported;if(reported[{shader,what}])return;reported[{shader,what}]=true;
    u64 a=0,b=0;PendingWriteRange(address,size,&a,&b);
    RT_LOG("%s of shader %#llx at %#llx+%#llx waits for GPU writes to %#llx-%#llx",what,(unsigned long long)shader,(unsigned long long)address,(unsigned long long)size,(unsigned long long)a,(unsigned long long)b);
}
bool HandleMemoryPending(const Pipeline& pipeline){
    for(const auto* shader:pipeline.shaders){
        auto& stage=(*tls_state).stages[ShaderPublicStage[shader->source->stage]];
        bool program=false,reads=false;
        for(const auto& r:shader->compiled.resources){program|=!r.handle_program.empty();reads|=r.storage_index!=~0u||r.indirect_offset!=~0u;}
        if(!program&&!reads)continue;
        // Pointers to storage buffers are read on the CPU (their contents are not).
        for(const auto& r:shader->compiled.resources)if(r.kind==nvshader::ResourceKind::StorageBuffer&&r.indirect_offset!=~0u){
            const u64 base=Cbuf(&stage,r.bank,r.offset)|(u64)Cbuf(&stage,r.bank,r.offset+4)<<32;
            if(HasPendingWrite(base+r.indirect_offset,8)){
                // The pointer usually sits in a structure whose other fields GPU work updates:
                // use its current value, and check once the batch completed that it did not change.
                static const bool exact=std::getenv("OOT_EXACT_STORAGE_POINTERS")!=nullptr;
                if(exact){ReportHandleWait("storage pointer",shader->source->address,base+r.indirect_offset,8);return true;}
                const u64 at=base+r.indirect_offset,value=ReadPointer(nullptr,at);
                AfterFlush([at,value,shader=shader->source->address]{
                    const u64 now=ReadPointer(nullptr,at);
                    static u32 changed=0;if(now!=value&&changed++<20)RT_LOG("storage pointer of shader %#llx at %#llx changed on the GPU (%#llx -> %#llx): one draw used the old one",
                        (unsigned long long)shader,(unsigned long long)at,(unsigned long long)value,(unsigned long long)now);});
            }}
        // Storage-indexed handle arrays: exactly the entries read.
        for(const auto& r:shader->compiled.resources)if(r.storage_index!=~0u){
            const u64 address=ShaderStorageAddress(*shader,r.storage_index);
            const u64 extent=(u64)std::max(r.count,1u)*std::max(r.storage_stride,4u);
            if(address&&HasPendingWrite(address+r.storage_offset,extent)){
                static std::map<u64,bool> reported;u64 a=0,b=0;
                if(std::getenv("OOT_WAIT_TRACE")&&!reported[shader->source->address]&&PendingWriteRange(address+r.storage_offset,extent,&a,&b)){reported[shader->source->address]=true;
                    RT_LOG("handle array of shader %#llx at %#llx+%#llx waits for GPU writes to %#llx-%#llx",(unsigned long long)shader->source->address,(unsigned long long)(address+r.storage_offset),(unsigned long long)extent,(unsigned long long)a,(unsigned long long)b);}
                return true;}}
        if(!program)continue;
        // Handle programs: dry-run each one, recording the guest words it reads, and check exactly
        // those (a program reads a few words; the buffers around them are often GPU-written).
        // (the words are gathered and asked about together: a handle table's elements read a word each, and
        // asking one by one was a sixth of the executor's time with Longshot's translator)
        static thread_local std::vector<std::pair<u64,u64>> words;words.clear();
        struct Probe { StageState* stage; const ShaderVariant* shader; std::vector<std::pair<u64,u64>>* words; } probe{&stage,shader,&words};
        for(const auto& r:shader->compiled.resources){
            if(r.handle_program.empty())continue;
            for(u32 element=0;element<std::max(r.count,1u);++element)
                nvshader::EvaluateHandle(r.handle_program,&probe,
                    [](void* p,u32 bank,u32 offset){auto& q=*static_cast<Probe*>(p);
                        if(bank>=UniformBankBase&&bank-UniformBankBase<q.stage->uniforms.size()){const auto& u=q.stage->uniforms[bank-UniformBankBase];
                            if(u.address)q.words->push_back({u.address+offset,4});}
                        return Cbuf(q.stage,bank,offset);},
                    [](void* p,u32 index,u32 offset){auto& q=*static_cast<Probe*>(p);
                        const u64 alignment=Vk().properties.limits.minStorageBufferOffsetAlignment;
                        const u64 address=(ShaderStorageAddress(*q.shader,index)&~(alignment-1))+offset;
                        q.words->push_back({address,4});
                        return ReadWord(address);}
#ifdef OOT_SHADER_LONGSHOT
                    ,element
#endif
                    );
        }
        if(words.size()>1){std::sort(words.begin(),words.end());words.erase(std::unique(words.begin(),words.end()),words.end());}
        if(HasPendingWriteAny(words)){ReportHandleWait("handle program",shader->source->address,0,4);return true;}
    }
    return false;
}
// Resources of the draw/dispatch whose descriptors were written last: textures it reads and
// whether it writes storage (buffers or images). Draw uses them to keep a render pass open.
thread_local std::vector<Texture*> descriptor_reads;
thread_local std::vector<Texture*> resolve_to_general; // textures a draw being resolved samples in another layout
thread_local bool descriptor_writes=false;
// The descriptor writes of the last WriteDescriptors (pushed by a push-descriptor pipeline's draw).
thread_local std::vector<VkWriteDescriptorSet> descriptor_write_list;
// Writes of descriptor_write_list that come from a uniform binding (the draw fast path patches
// them), and whether the last resolution depends only on bindings the fast path tracks.
struct UniformSlot { u32 write; StageState* stage; u32 bank; };
thread_local std::vector<UniformSlot> uniform_write_slots;
// Writes that are one texture whose handle a small program computes from the draw's constants (most of the game's
// material textures): the fast path evaluates the handle again and patches the write when it names another texture.
// OOT_FAST_TEXTURES=1 turns it on (off by default).
struct TextureSlot { u32 write, read; ShaderVariant* shader; const nvshader::Resource* resource; u32 handle; VkImage image; };
thread_local std::vector<TextureSlot> texture_write_slots;
// State that only chooses the pipeline (vertex format, blending, depth and stencil, polygon state): the fast path
// looks the pipeline up again and keeps the descriptors where its programs are the same. OOT_FAST_STATE=1 turns this on;
// by default such a change ends the fast path.
bool FastState(){static const bool on=std::getenv("OOT_FAST_STATE")!=nullptr;return on;} // (off unless asked for, as FastTextures)
inline void PipelineStateChanged(){if(FastState())fast_pipeline_stale=true;else fast_draw_allowed=false;}
// Writes that are a driver bank (what the shader reads of the draw's own parameters and bindings).
struct DriverSlot { u32 write; StageState* stage; u32 bank; };
thread_local std::vector<DriverSlot> driver_write_slots;
// Writes that are one storage buffer the draw only reads, bound the plain way (its address and size in the stage's
// own bank): kept with the words it was resolved from and the memory it reads.
struct StorageSlot { u32 write; StageState* stage; const nvshader::Resource* resource; u32 words[3]; u64 cpu_address, extent; };
thread_local std::vector<StorageSlot> storage_write_slots;
// Whether the last resolution holds nothing but uniform buffers, driver banks and single textures: what a later draw
// of the same pipeline can take over whole (ResolveCache below).
thread_local bool descriptor_cacheable = false;
bool FastTextures(){static const bool on=std::getenv("OOT_FAST_TEXTURES")!=nullptr;return on;} // (off unless asked for: no frame rate gained on an 8-core CPU, ~2% with two helper threads)
thread_local bool descriptor_fast_eligible = false;
// Guest ranges [start, end) of the buffers bound by the last WriteDescriptors, and the images it
// writes: compute dispatches use them to skip barriers between independent dispatches.
thread_local std::vector<std::pair<u64,u64>> descriptor_buffer_reads,descriptor_buffer_writes;
thread_local std::vector<Texture*> descriptor_image_writes;
std::shared_ptr<Texture> SnapshotAttachment(Texture& source){
    auto copy=std::shared_ptr<Texture>(new Texture,RetireTexture);
    copy->format=source.format;copy->target=source.target;
    copy->width=source.width;copy->height=source.height;copy->depth=source.depth;
    copy->levels=source.levels;copy->layers=source.layers;copy->samples=source.samples;
    copy->flags=source.flags;copy->info=source.info;
    std::memcpy(copy->swizzle,source.swizzle,sizeof copy->swizzle);
    copy->depth_stencil_mode=source.depth_stencil_mode;
    // Private storage identity: never adopt/steal an image belonging to guest memory.
    copy->storage=reinterpret_cast<u64>(copy.get());
    EnsureImage(*copy);
    Immediate([&](VkCommandBuffer cmd){
        Transition(cmd,source,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        Transition(cmd,*copy,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        std::vector<VkImageCopy> regions;
        for(u32 level=0;level<source.levels;++level){
            VkImageCopy region{};
            region.srcSubresource=region.dstSubresource={AspectOf(source),level,0,source.layers};
            region.extent={std::max(1u,source.width>>level),std::max(1u,source.height>>level),source.target==Target::T3D?std::max(1u,source.depth>>level):1};
            regions.push_back(region);
        }
        vkCmdCopyImage(cmd,source.image,source.layout,copy->image,copy->layout,regions.size(),regions.data());
        Transition(cmd,source,VK_IMAGE_LAYOUT_GENERAL);
        Transition(cmd,*copy,VK_IMAGE_LAYOUT_GENERAL);
    });
    copy->uploaded=true;
    AfterBatch([copy]{}); // descriptor views survive through this draw's GPU submission
    return copy;
}
// Fills the pipeline's (reused) descriptor set from the current state.
bool WriteDescriptors(Pipeline& pipeline){
    auto& vk=Vk();static const bool gpu_trace=std::getenv("OOT_GPU_TRACE")!=nullptr;
    // Texture handles and buffer pointers read from guest memory on the CPU must see GPU writes
    // recorded earlier in the open batch: submit first only when such a write is pending.
    if(HandleMemoryPending(pipeline)){if(resolve_worker){RESOLVE_BAIL();return false;}Flush();}
    if(resolve_bail)return false;
    if(LogThisFrame())if(const char* dir=std::getenv("OOT_INPUT_DUMP")){
        const u64 address=std::getenv("OOT_INPUT_DUMP_SHADER")?std::strtoull(std::getenv("OOT_INPUT_DUMP_SHADER"),nullptr,16):0;
        for(auto* shader:pipeline.shaders)if(shader->source->address==address){
            for(const auto& r:shader->compiled.resources)if(r.kind==nvshader::ResourceKind::Texture)
                for(u32 element=0;element<r.count;++element){
                    const u32 handle=ResourceHandle(r,*shader,element);if(!HandleRegistered(handle))continue;
                    auto& texture=*Binding(handle).host;EnsureUploaded(texture);
                    const auto bytes=ReadTexels(texture);
                    char path[512];std::snprintf(path,sizeof path,"%s/%llx-%x-%x-%u.bin",dir,(unsigned long long)address,r.bank,r.offset,element);
                    if(FILE* f=std::fopen(path,"wb")){
                        std::fprintf(f,"%u %u %u %u %u\n",texture.width,texture.height,texture.depth,texture.format,texture.info.bytes);
                        std::fwrite(bytes.data(),1,bytes.size(),f);std::fclose(f);
                    }
                }
            auto& stage=(*tls_state).stages[ShaderPublicStage[shader->source->stage]];
            for(u32 bank=3;bank<=4;++bank){
                char path[512];std::snprintf(path,sizeof path,"%s/%llx-cbuf%u.bin",dir,(unsigned long long)address,bank);
                if(FILE* f=std::fopen(path,"wb")){
                    const u32 size=std::min<u64>(stage.uniforms[bank-UniformBankBase].size,0x10000);
                    for(u32 offset=0;offset+4<=size;offset+=4){const u32 word=Cbuf(&stage,bank,offset);std::fwrite(&word,4,1,f);}
                    std::fclose(f);
                }
            }
        }
    }
    descriptor_reads.clear();descriptor_writes=false;resolve_to_general.clear();
    // Readback flushes the batch and resets its descriptor pools. Finish debug
    // captures before allocating this draw's set, so it survives until submission.
    if(LogThisFrame())if(const char* dir=std::getenv("OOT_DRAW_DUMP")) {
        for(auto* shader:pipeline.shaders){
            for(const auto& r:shader->compiled.resources)if(r.kind==nvshader::ResourceKind::Texture||r.kind==nvshader::ResourceKind::Image)
                for(u32 element=0;element<r.count;++element){const u32 handle=ResourceHandle(r,*shader,element);if(!HandleRegistered(handle))continue;
                    auto& binding=Binding(handle);auto& t=*binding.host.get();EnsureUploaded(t);
                    char name[96];std::snprintf(name,sizeof name,"/tex-%llx.ppm",(unsigned long long)binding.texture);DumpTexture(t,dir+std::string(name));
                }
        }
    }
    // Driver banks for this draw go to the ring; recycle it (after the GPU is done) when low.
    if(resolve_worker&&(!ring.buffer||ring.head+32*DriverBankSize+(4u<<20)>ring.size)){RESOLVE_BAIL();return false;}
    if(ring.buffer&&ring.head+32*DriverBankSize+(4u<<20)>ring.size){Flush();ring.head=0;++ring_generation;ring_banks.clear();uniform_overlay_cache.clear();}
    // Uploads below may flush, retiring this batch's descriptor pools: then write a fresh set.
    for(u64 flushes=~0ull;flushes!=FlushCount();){
    flushes=FlushCount();
    std::map<Texture*,std::shared_ptr<Texture>> attachment_snapshots;
    descriptor_reads.clear();descriptor_writes=false;
    descriptor_buffer_reads.clear();descriptor_buffer_writes.clear();descriptor_image_writes.clear();
    uniform_write_slots.clear();texture_write_slots.clear();driver_write_slots.clear();storage_write_slots.clear();descriptor_fast_eligible=true;descriptor_cacheable=true;
    // The set is allocated once its contents are known: a draw binding the same resources as the
    // pipeline's previous draw in this batch reuses that set.
    const VkDescriptorSet set=VK_NULL_HANDLE;
    if(gpu_trace)RT_LOG("GPU descriptor update");
    // Descriptor writes of this set, applied together (their arrays live until then).
    auto& writes=descriptor_write_list;
    static thread_local std::deque<std::vector<VkDescriptorBufferInfo>> buffer_lists;
    static thread_local std::deque<std::vector<VkDescriptorImageInfo>> image_lists;
    writes.clear();buffer_lists.clear();image_lists.clear();
    const auto queue_write=[&](const nvshader::Resource& r,std::vector<VkDescriptorBufferInfo>&& b,std::vector<VkDescriptorImageInfo>&& i){
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};write.dstSet=set;write.dstBinding=r.binding;write.descriptorType=DescriptorType(r.kind);
        if(!b.empty()){buffer_lists.push_back(std::move(b));write.descriptorCount=buffer_lists.back().size();write.pBufferInfo=buffer_lists.back().data();}
        else{image_lists.push_back(std::move(i));write.descriptorCount=image_lists.back().size();write.pImageInfo=image_lists.back().data();}
        if(write.descriptorCount)writes.push_back(write);};
    for(auto* shader:pipeline.shaders){auto& stage=(*tls_state).stages[ShaderPublicStage[shader->source->stage]];
        for(const auto& r:shader->compiled.resources){
            std::vector<VkDescriptorBufferInfo> buffers;std::vector<VkDescriptorImageInfo> images;
            // (Fast path: handles or pointers read through memory, and root-bank uniforms, are not tracked.)
            const bool texture_slot=FastTextures()&&r.kind==nvshader::ResourceKind::Texture&&r.count==1&&r.storage_index==~0u&&r.indirect_offset==~0u&&r.bank!=RootBank;
            u32 slot_handle=0,slot_read=~0u; // (set where the texture resolved the ordinary way)
            if(r.kind==nvshader::ResourceKind::Texture&&r.count!=1&&FastTextures())descriptor_fast_eligible=false; // (an array's handles are not followed)
            const bool storage_slot=r.kind==nvshader::ResourceKind::StorageBuffer&&!r.written&&r.count==1&&r.indirect_offset==~0u&&r.storage_index==~0u&&r.bank<UniformBankBase;
            if(r.kind!=nvshader::ResourceKind::UniformBuffer&&!texture_slot&&!storage_slot)descriptor_cacheable=false;
            bool driver_slot=false,storage_resolved=false;u64 storage_cpu=0,storage_extent=0;
            if((!r.handle_program.empty()&&!texture_slot)||r.storage_index!=~0u||r.indirect_offset!=~0u||r.bank==RootBank||
               (r.kind==nvshader::ResourceKind::StorageBuffer&&r.bank>=UniformBankBase))descriptor_fast_eligible=false;
            {static const bool why=std::getenv("OOT_SET_STATS")!=nullptr;if(why&&!resolve_worker){static u64 n[6]{},at=0;
                if(!r.handle_program.empty())++n[0];if(r.storage_index!=~0u)++n[1];if(r.indirect_offset!=~0u)++n[2];if(r.bank==RootBank)++n[3];
                if(r.kind==nvshader::ResourceKind::StorageBuffer&&r.bank>=UniformBankBase)++n[4];if(r.kind==nvshader::ResourceKind::StorageBuffer&&r.count>1)++n[5];
                if(at!=frame_number&&frame_number%300==0){at=frame_number;RT_LOG("ineligible resources a frame: handle program %.0f, storage index %.0f, indirect %.0f, root bank %.0f, storage in uniform bank %.0f, storage table %.0f",n[0]/300.0,n[1]/300.0,n[2]/300.0,n[3]/300.0,n[4]/300.0,n[5]/300.0);std::memset(n,0,sizeof n);}}}
            const bool uniform_slot=r.kind==nvshader::ResourceKind::UniformBuffer&&r.bank>=UniformBankBase&&r.bank!=RootBank;
                if(r.kind==nvshader::ResourceKind::UniformBuffer&&(r.bank<UniformBankBase||(r.bank==RootBank&&!stage.uniforms[RootBank-UniformBankBase].address))){buffers.push_back(DriverBankDescriptor(stage,r.bank));driver_slot=true;}
                else if(r.kind==nvshader::ResourceKind::StorageBuffer&&r.count>1){
                    // A storage buffer table (Longshot's translator): NVN descriptors {address, size} at offset + 16k of
                    // the constant buffer, which the program's global accesses choose between at run time.
                    descriptor_fast_eligible=false;
                    const u64 alignment=vk.properties.limits.minStorageBufferOffsetAlignment;
                    for(u32 k=0;k<r.count;++k){
                        // (the table may be shorter than the translator's 16 entries: words past the bound buffer are 0)
                        const auto word=[&](u32 at)->u32{
                            if(r.bank>=UniformBankBase){const u32 u=r.bank-UniformBankBase;
                                if(u>=stage.uniforms.size()||!stage.uniforms[u].address||at+4>stage.uniforms[u].size)return 0;}
                            return Cbuf(&stage,r.bank,at);};
                        const u64 address=word(r.offset+16*k)|(u64)word(r.offset+16*k+4)<<32;
                        const u64 size=word(r.offset+16*k+8);
                        const auto range=address?GpuLookupAddress(address):BufferRange{};
                        if(resolve_worker&&(resolve_bail||(address&&!range.pool)||r.written)){RESOLVE_BAIL();return false;}
                        if(!range.pool){buffers.push_back(ZeroBuffer());continue;} // (an empty entry of the table)
                        if(size>(256ull<<20)){buffers.push_back(ZeroBuffer());continue;} // (past the table's end: whatever the constants hold)
                        // An entry without a size is a bare pointer into a pool the program reads anywhere in: bound as far
                        // as a binding reaches (1 MiB if written). What the GPU writes through a bare one is only known to
                        // start there: its first 64 bytes go back to guest memory, the rest is a guess and stays the GPU's
                        // (copied back whole it would overwrite the CPU's data beside it; Longshot vk_draw.cpp does the same).
                        const u64 delta=range.offset&(alignment-1);
                        const u64 extent=std::min<u64>(size?size:r.written?(1ull<<20):(1ull<<30),range.available);
                        u64 touched=size?extent:std::min<u64>(extent,64);
#ifdef OOT_SHADER_LONGSHOT
                        if(r.bank==0&&r.offset==PointerTableOffset)touched=0; // (the pointer table's regions are guesses: the GPU's alone)
#endif
                        if(r.written&&!touched){descriptor_writes=true;descriptor_buffer_writes.push_back({address,address+extent});}
                        else if(r.written){NotePendingWrite(address,touched,true);descriptor_writes=true;descriptor_buffer_writes.push_back({address,address+touched});}
                        else descriptor_buffer_reads.push_back({address,address+extent});
                        if(UniformOverlayEnabled())WriteUniformUpdatesOverlapping(range.cpu_address,extent);
                        buffers.push_back({range.buffer,range.offset-delta,extent+delta});}
                    queue_write(r,std::move(buffers),{});
                    continue;
                }
                else if(r.kind==nvshader::ResourceKind::UniformBuffer||r.kind==nvshader::ResourceKind::StorageBuffer){
                Buffer b;
                if(r.kind==nvshader::ResourceKind::UniformBuffer){b=stage.uniforms.at(r.bank-UniformBankBase);}
                else {b.address=Cbuf(&stage,r.bank,r.offset)|(u64)Cbuf(&stage,r.bank,r.offset+4)<<32;
                    if(r.indirect_offset!=~0u)b.address=ReadPointer(nullptr,b.address+r.indirect_offset);
                    if(r.sized_storage)b.size=Cbuf(&stage,r.bank,r.offset+8);
                    static const bool exact=std::getenv("OOT_EXACT_STORAGE_SIZE")==nullptr?false:true;
                    if(!exact)b.size=0;} // (bound to the end of its memory: diagnostic default, see below)
                auto range=GpuLookupAddress(b.address);
                if(resolve_worker&&resolve_bail)return false; // (a device copy to make: the executor's; the site that asked is kept)
                if(resolve_worker&&!range.pool){RESOLVE_BAIL();return false;} // (a zero buffer: the executor's)
                if(resolve_worker&&r.kind==nvshader::ResourceKind::StorageBuffer&&r.written){RESOLVE_BAIL();return false;} // (a GPU write to note: the executor's)
                if(!range.pool){
                    // Unbound slots read as zeros (shaders guard such accesses on the console).
                    static std::map<std::tuple<u64,u32,u32>,bool> reported;
                    if(!reported[{shader->source->address,r.bank,r.offset}]){reported[{shader->source->address,r.bank,r.offset}]=true;
                        RT_LOG("shader %#llx stage %u: %s bank %u+%#x unbound (%#llx); using a zero buffer",(unsigned long long)shader->source->address,shader->source->stage,r.kind==nvshader::ResourceKind::StorageBuffer?"storage":"uniform",r.bank,r.offset,(unsigned long long)b.address);}
                    descriptor_fast_eligible=false;
                    buffers.push_back(ZeroBuffer());
                    queue_write(r,std::move(buffers),{});
                    continue;
                }
                // Bound to the end of its memory: written ones up to 64 MiB, read-only ones as far as a storage binding
                // reaches (1 GiB at most). OoT's froxel fog finds its light probes up to ~130 MB past the pointer; past
                // the binding they read 0 and the fog went overbright (from Hookshot 3858629 / Lichii #351).
                // OOT_STORAGE_READ_MB=<MiB> caps the read-only ones.
                static const u64 read_cap=[&]{const u64 reach=std::min<u64>(vk.properties.limits.maxStorageBufferRange,1ull<<30)&~0xffffull;
                    return std::max<u64>(64ull<<20,std::getenv("OOT_STORAGE_READ_MB")?std::min<u64>(reach,(u64)std::atoll(std::getenv("OOT_STORAGE_READ_MB"))<<20):reach);}();
                if(!b.size)b.size=std::min<u64>(range.available,r.kind==nvshader::ResourceKind::StorageBuffer&&!r.written?read_cap:(64u<<20));
                // Extent the shader accesses: an NVN storage descriptor in the driver bank holds its
                // size at +8 (shaders keep to it; the binding itself stays wider).
                u64 extent=std::min(b.size,range.available);
                if(r.kind==nvshader::ResourceKind::StorageBuffer&&r.indirect_offset==~0u&&r.bank==0){
                    static const bool whole=std::getenv("OOT_WHOLE_STORAGE_WRITES")!=nullptr;
                    const u64 nvn_size=Cbuf(&stage,r.bank,r.offset+8);if(nvn_size&&!whole)extent=std::min(extent,nvn_size);}
#ifdef OOT_SHADER_LONGSHOT
                // Longshot's translator binds every candidate pointer of an access it cannot resolve: 8-byte arguments in
                // the program's own constant buffers. Written ones were each 64 MiB of pending GPU writes, which the game's
                // threads then waited for and copied back (8 extra submissions a frame). The word after such a pointer is
                // its size when it looks like one; else the pointer is bare and only its first 64 bytes count as written
                // (as the table entries above).
                if(r.kind==nvshader::ResourceKind::StorageBuffer&&r.written&&r.indirect_offset==~0u&&r.bank>=UniformBankBase){
                    const u32 u=r.bank-UniformBankBase;
                    const bool words=u<stage.uniforms.size()&&stage.uniforms[u].address&&r.offset+16<=stage.uniforms[u].size;
                    const u64 size=words?Cbuf(&stage,r.bank,r.offset+8):0,high=words?Cbuf(&stage,r.bank,r.offset+12):1;
                    extent=size&&!high&&size<=range.available?size:std::min<u64>(extent,64);}
#endif
                {static const u64 info=std::getenv("OOT_SHADER_INFO")?std::strtoull(std::getenv("OOT_SHADER_INFO"),nullptr,16):0;
                 if(info==1&&!range.pool->device_buffer){static std::set<std::tuple<u64,u32,u32>> seen;if(seen.insert({shader->source->address,r.bank,r.offset}).second&&seen.size()<400)RT_LOG("host-memory binding: shader %#llx stage %u %s bank %u+%#x: %#llx+%#llx pool %#llx+%#llx flags %#x",
                    (unsigned long long)shader->source->address,shader->source->stage,r.kind==nvshader::ResourceKind::StorageBuffer?(r.written?"storage (written)":"storage"):"uniform",r.bank,r.offset,(unsigned long long)b.address,(unsigned long long)extent,
                    (unsigned long long)range.pool->address,(unsigned long long)range.pool->size,range.pool->flags);}
                 if(info&&shader->source->address==info){static int shown=0;if(shown++<40)RT_LOG("shader %#llx %s bank %u+%#x: %#llx+%#llx pool %#llx flags %#x %s",(unsigned long long)info,
                    r.kind==nvshader::ResourceKind::StorageBuffer?(r.written?"storage (written)":"storage"):"uniform",r.bank,r.offset,(unsigned long long)b.address,(unsigned long long)extent,
                    (unsigned long long)range.pool->address,range.pool->flags,range.pool->device_buffer?"device copy":"host memory");}}
                if(r.kind==nvshader::ResourceKind::StorageBuffer&&r.written){static const bool write_back=std::getenv("OOT_NO_STORAGE_WRITEBACK")==nullptr;NotePendingWrite(b.address,extent,write_back);descriptor_writes=true;}
                (r.kind==nvshader::ResourceKind::StorageBuffer&&r.written?descriptor_buffer_writes:descriptor_buffer_reads).push_back({b.address,b.address+extent});
                const u64 alignment=r.kind==nvshader::ResourceKind::StorageBuffer?vk.properties.limits.minStorageBufferOffsetAlignment:vk.properties.limits.minUniformBufferOffsetAlignment;
                const u64 delta=range.offset&(alignment-1);
                if(r.kind==nvshader::ResourceKind::UniformBuffer){
                    const u64 bytes=std::min<u64>(std::min(b.size,range.available)+delta,65536+delta);
                    if(const auto o=UniformOverlay(range.cpu_address-delta,bytes);o.buffer){buffers.push_back(o);queue_write(r,std::move(buffers),{});
                        if(uniform_slot)uniform_write_slots.push_back({(u32)writes.size()-1,&stage,r.bank});continue;}
                }else if(UniformOverlayEnabled()){if(resolve_worker&&UnwrittenUniformOverlap(range.cpu_address,extent)){RESOLVE_BAIL();return false;}WriteUniformUpdatesOverlapping(range.cpu_address,extent);} // (read as storage: memory must have them)
                buffers.push_back({range.buffer,range.offset-delta,std::min(b.size,range.available)+delta});
                storage_resolved=true;storage_cpu=range.cpu_address;storage_extent=extent;
            }else if(r.kind==nvshader::ResourceKind::Texture||r.kind==nvshader::ResourceKind::Image){
                u32 array_valid=0;
                for(u32 element=0;element<r.count;++element){const u32 handle=ResourceHandle(r,*shader,element);
                    bool array_entry_valid=true;
                    if(r.storage_index!=~0u){
                        array_entry_valid=HandleRegistered(handle)&&(r.kind!=nvshader::ResourceKind::Texture||SamplerRegistered(handle));
                        if(array_entry_valid){const auto& b=Binding(handle);const auto* t=b.host.get();
                            const u32 type=TextureType(nullptr,handle);
                            array_entry_valid=(type==r.texture_type||(type==8&&r.texture_type==2))&&(!r.depth||t->info.depth);
                        }
                    }
                    if(handle==0||!array_entry_valid){
                        // A declared resource may be unused on this material's path.
                        // A null image returns zero and discards writes, preserving
                        // the rest of the draw instead of dropping its geometry.
                        static const VkSampler null_sampler=[]{VkSampler made=VK_NULL_HANDLE;VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};si.maxLod=VK_LOD_CLAMP_NONE;VK_CHECK(vkCreateSampler(Vk().device,&si,nullptr,&made));return made;}(); // (made once, whichever thread comes first)
                        images.push_back({r.kind==nvshader::ResourceKind::Texture?null_sampler:VK_NULL_HANDLE,VK_NULL_HANDLE,VK_IMAGE_LAYOUT_GENERAL});
                        continue;
                    }
                    if(!HandleRegistered(handle)&&resolve_worker&&!std::getenv("OOT_SKIP_UNREGISTERED")){ // (as below, without the report)
                        static const VkSampler null_sampler3=[]{VkSampler made=VK_NULL_HANDLE;VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};si.maxLod=VK_LOD_CLAMP_NONE;VK_CHECK(vkCreateSampler(Vk().device,&si,nullptr,&made));return made;}();
                        images.push_back({r.kind==nvshader::ResourceKind::Texture?null_sampler3:VK_NULL_HANDLE,VK_NULL_HANDLE,VK_IMAGE_LAYOUT_GENERAL});
                        continue;}
                    if(!HandleRegistered(handle)){
                        if(resolve_worker){RESOLVE_BAIL();return false;}
                        static std::map<u64,bool> reported;
                        if(!reported[shader->source->address]){reported[shader->source->address]=true;
                            RT_LOG("shader %#llx stage %u: texture resource bank %u+%#x element %u has unregistered handle %#x; bound as a null texture",(unsigned long long)shader->source->address,shader->source->stage,r.bank,r.offset,element,handle);
                            std::string bound;for(u32 i=0;i<32;++i){char b[64];if(stage.textures[i]&&(u32)stage.textures[i]!=0x10200102){const u32 h=(u32)stage.textures[i];std::snprintf(b,sizeof b," t%u=%#x(type%d)",i,h,HandleRegistered(h)?(int)TextureType(nullptr,h):-1);bound+=b;}if(stage.images[i]){std::snprintf(b,sizeof b," i%u=%#llx",i,(unsigned long long)stage.images[i]);bound+=b;}}
                            for(u32 i=0;i<16;++i)if(stage.storage[i].address){char b[64];std::snprintf(b,sizeof b," s%u=%#llx/%#llx",i,(unsigned long long)stage.storage[i].address,(unsigned long long)stage.storage[i].size);bound+=b;}
                            for(u32 i=0;i<18;++i)if(stage.uniforms[i].address){char b[64];std::snprintf(b,sizeof b," u%u=%#llx/%#llx",i,(unsigned long long)stage.uniforms[i].address,(unsigned long long)stage.uniforms[i].size);bound+=b;}
                            RT_LOG("  bound:%s",bound.c_str());
                            for(const auto& q:shader->compiled.resources)RT_LOG("  resource kind %d bank %u offset %#x count %u",(int)q.kind,q.bank,q.offset,q.count);}
                        // Like an unused slot above: the material path that would sample it is not
                        // taken (the slot holds other data, e.g. a float), so the draw goes ahead with a
                        // null texture. Skipping it dropped text, UI images and effects.
                        static const bool skip=std::getenv("OOT_SKIP_UNREGISTERED")!=nullptr;
                        if(skip)return false;
                        static VkSampler null_sampler2=VK_NULL_HANDLE;
                        if(!null_sampler2){VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};si.maxLod=VK_LOD_CLAMP_NONE;VK_CHECK(vkCreateSampler(vk.device,&si,nullptr,&null_sampler2));}
                        images.push_back({r.kind==nvshader::ResourceKind::Texture?null_sampler2:VK_NULL_HANDLE,VK_NULL_HANDLE,VK_IMAGE_LAYOUT_GENERAL});
                        continue;}
                    ++array_valid;
                    auto& binding=Binding(handle);auto& t=*binding.host.get();EnsureUploaded(t);
                    if(resolve_bail)return false;
                    if(resolve_worker){if(resolve_texture_unready||t.layout!=VK_IMAGE_LAYOUT_GENERAL)resolve_to_general.push_back(&t);resolve_texture_unready=false;} // (uploaded and brought to the layout when the draw is committed)
                    else if(t.layout!=VK_IMAGE_LAYOUT_GENERAL)
                        Immediate([&](VkCommandBuffer cmd){Transition(cmd,t,VK_IMAGE_LAYOUT_GENERAL);});
                    descriptor_reads.push_back(&t);slot_handle=handle;slot_read=(u32)descriptor_reads.size()-1;if(r.kind==nvshader::ResourceKind::Image&&r.written){descriptor_writes=true;descriptor_image_writes.push_back(&t);}
                    if(LogThisFrame()&&t.format==1&&t.storage&&t.storage_size){std::vector<u8> bytes(t.storage_size);std::memcpy(bytes.data(),ResolveAddress(t.storage,bytes.size()),bytes.size());u64 nz=0;for(u8 b:bytes)nz+=b!=0;RT_LOG("  (R8 texture %#llx: %llu/%zu nonzero guest bytes, uploaded frame %llu, upload seq %u)",(unsigned long long)binding.texture,(unsigned long long)nz,bytes.size(),(unsigned long long)t.upload_frame,t.upload_sequence);}
                    if(LogThisFrame()){const u32* sw=t.swizzle;RT_LOG("  stage %u texture resource bank %u offset %#x element %u: handle %#x -> texture %#llx %ux%u fmt %u written %d storage %#llx swizzle %u %u %u %u view %d (%u %u %u %u)",shader->source->stage,r.bank,r.offset,element,handle,(unsigned long long)binding.texture,t.width,t.height,t.format,(int)t.gpu_written,(unsigned long long)t.storage,sw[0],sw[1],sw[2],sw[3],binding.view.present,binding.view.swizzle[0],binding.view.swizzle[1],binding.view.swizzle[2],binding.view.swizzle[3]);}
                    if(r.kind==nvshader::ResourceKind::Image&&!resolve_worker){static std::map<std::pair<u64,u32>,bool> seen;if(!seen[{shader->source->address,t.format}]){seen[{shader->source->address,t.format}]=true;
                        RT_LOG("image binding: shader %#llx bank %u+%#x texture fmt %u %ux%u, shader image format %u written %d",(unsigned long long)shader->source->address,r.bank,r.offset,t.format,t.width,t.height,r.image_format,(int)r.written);}}
                    TextureView view=binding.view;view.target=ShaderViewTarget(r.texture_type,t)+1;
                    if(r.kind==nvshader::ResourceKind::Image){view.swizzle[0]=2;view.swizzle[1]=3;view.swizzle[2]=4;view.swizzle[3]=5;} // storage views: identity
                    Texture* sampled=&t;
                    if(r.kind==nvshader::ResourceKind::Texture&&shader->source->stage!=5){
                        for(u32 color=0;color<(*tls_state).color_count;++color)if((*tls_state).targets.colors[color]&&HostTexture((*tls_state).targets.colors[color])==&t&&(((*tls_state).channel>>(color*4))&15)){
                            if(resolve_worker){RESOLVE_BAIL();return false;} // (a copy of the target: recorded by the executor)
                            auto& snapshot=attachment_snapshots[&t];
                            if(!snapshot)snapshot=SnapshotAttachment(t);
                            sampled=snapshot.get();descriptor_fast_eligible=false;break;
                        }
                    }
                    images.push_back({r.kind==nvshader::ResourceKind::Texture?Sampler(handle):VK_NULL_HANDLE,ImageView(*sampled,view),VK_IMAGE_LAYOUT_GENERAL});
                    if(resolve_bail)return false;}
                if(r.storage_index!=~0u&&!resolve_worker){static std::map<std::pair<u64,u32>,bool> seen;
                    if(!seen[{shader->source->address,r.binding}]){seen[{shader->source->address,r.binding}]=true;
                        RT_LOG("storage texture array shader %#llx binding %u: %u/%u valid, storage %u+%#x stride %u",(unsigned long long)shader->source->address,r.binding,array_valid,r.count,r.storage_index,r.storage_offset,r.storage_stride);}}
            }else Fatal("texel buffer descriptor pending");
            queue_write(r,std::move(buffers),std::move(images));
            if(uniform_slot)uniform_write_slots.push_back({(u32)writes.size()-1,&stage,r.bank});
            if(driver_slot)driver_write_slots.push_back({(u32)writes.size()-1,&stage,r.bank});
            if(storage_slot){if(!storage_resolved)descriptor_cacheable=false;
                else storage_write_slots.push_back({(u32)writes.size()-1,&stage,&r,{Cbuf(&stage,r.bank,r.offset),Cbuf(&stage,r.bank,r.offset+4),Cbuf(&stage,r.bank,r.offset+8)},storage_cpu,storage_extent});}
            if(texture_slot){if(slot_read==~0u)descriptor_fast_eligible=false; // (a null texture: the full path's)
                else texture_write_slots.push_back({(u32)writes.size()-1,slot_read,shader,&r,slot_handle,descriptor_reads[slot_read]->image});}
        }
    }
    // One update for the whole set (unless uploads above flushed: then the loop writes a new set).
    if(flushes!=FlushCount())continue; // (uploads above flushed: resolve again for the new batch)
    if(pipeline.push){pipeline.descriptor_set=VK_NULL_HANDLE;break;} // (the draw pushes descriptor_write_list)
    if(resolve_worker)break; // (its set is made from these writes when the draw is committed: WriteBatchSet)
    {
        u64 hash=0xcbf29ce484222325ull;
        // (8 bytes at a time: descriptor infos are multiples of 8 bytes; a byte-wise hash of large
        // texture arrays cost more than the set it saved.)
        const auto mix=[&](const void* data,size_t size){const u8* b=static_cast<const u8*>(data);size_t i=0;
            for(;i+8<=size;i+=8){u64 w;std::memcpy(&w,b+i,8);hash=(hash^w)*0x9E3779B97F4A7C15ull;hash^=hash>>29;}
            for(;i<size;++i)hash=(hash^b[i])*0x100000001b3ull;};
        for(const auto& w:writes){mix(&w.dstBinding,4);mix(&w.descriptorType,4);mix(&w.descriptorCount,4);
            if(w.pBufferInfo)mix(w.pBufferInfo,sizeof(VkDescriptorBufferInfo)*w.descriptorCount);
            if(w.pImageInfo)mix(w.pImageInfo,sizeof(VkDescriptorImageInfo)*w.descriptorCount);}
        static const bool no_reuse=std::getenv("OOT_NO_SET_REUSE")!=nullptr;
        static u64 hits=0,misses=0;if(std::getenv("OOT_SET_STATS")&&(hits+misses)%100000==0&&hits+misses)RT_LOG("descriptor sets: %llu reused, %llu written",(unsigned long long)hits,(unsigned long long)misses);
        if(!no_reuse&&pipeline.descriptor_set&&pipeline.descriptor_hash==hash&&pipeline.descriptor_flushes==flushes){++hits;break;} // unchanged
        ++misses;
        pipeline.descriptor_set=AllocateBatchSet(pipeline.descriptors);
        if(flushes!=FlushCount())continue;
        pipeline.descriptor_hash=hash;pipeline.descriptor_flushes=flushes;
        for(auto& w:writes)w.dstSet=pipeline.descriptor_set;
        if(!writes.empty())vkUpdateDescriptorSets(vk.device,(u32)writes.size(),writes.data(),0,nullptr);
    }
    } // (descriptor set retry)
    return true;
}
// The descriptor set of a pipeline without pushed descriptors for these writes (as the end of WriteDescriptors:
// the pipeline's last set of this batch is reused while its contents are the same).
VkDescriptorSet WriteBatchSet(Pipeline& pipeline,std::vector<VkWriteDescriptorSet>& writes){
    u64 hash=0xcbf29ce484222325ull;
    const auto mix=[&](const void* data,size_t size){const u8* b=static_cast<const u8*>(data);size_t i=0;
        for(;i+8<=size;i+=8){u64 w;std::memcpy(&w,b+i,8);hash=(hash^w)*0x9E3779B97F4A7C15ull;hash^=hash>>29;}
        for(;i<size;++i)hash=(hash^b[i])*0x100000001b3ull;};
    for(const auto& w:writes){mix(&w.dstBinding,4);mix(&w.descriptorType,4);mix(&w.descriptorCount,4);
        if(w.pBufferInfo)mix(w.pBufferInfo,sizeof(VkDescriptorBufferInfo)*w.descriptorCount);
        if(w.pImageInfo)mix(w.pImageInfo,sizeof(VkDescriptorImageInfo)*w.descriptorCount);}
    for(;;){
        const u64 flushes=FlushCount();
        if(pipeline.descriptor_set&&pipeline.descriptor_hash==hash&&pipeline.descriptor_flushes==flushes)return pipeline.descriptor_set;
        pipeline.descriptor_set=AllocateBatchSet(pipeline.descriptors);
        if(flushes!=FlushCount())continue; // (the allocation flushed: a set of the new batch)
        pipeline.descriptor_hash=hash;pipeline.descriptor_flushes=flushes;
        for(auto& w:writes)w.dstSet=pipeline.descriptor_set;
        if(!writes.empty())vkUpdateDescriptorSets(Vk().device,(u32)writes.size(),writes.data(),0,nullptr);
        return pipeline.descriptor_set;
    }
}
// Indirect dispatch arguments, validated on the GPU (indirect_clamp.comp) into a slot of a
// device buffer: dispatched from there, and copied to the compute driver bank's grid words.
struct IndirectClamp{VkPipeline pipeline=VK_NULL_HANDLE;VkPipelineLayout layout=VK_NULL_HANDLE;VkBuffer buffer=VK_NULL_HANDLE;VkDeviceMemory memory=VK_NULL_HANDLE;
    VkDeviceAddress address=0;u32 next=0;static constexpr u32 Slots=4096;} indirect_clamp;
// The Foliage Density setting: the share of the GPU-culled mesh draws' tasks (grass and small plants) drawn, in 1/256ths.
u64 FoliageKeep(){
    if(const char* e=std::getenv("OOT_FOLIAGE_DENSITY"))return (u64)std::clamp(std::atoi(e)*256/100,1,256);
    return 256;
}
void InitializeIndirectClamp(){
    auto& vk=Vk();auto& c=indirect_clamp;if(c.pipeline)return;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=IndirectClamp::Slots*16;
    bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VK_CHECK(TracedCreateBuffer(&bi,&c.buffer,__FILE__,__LINE__));
    VkMemoryRequirements req;vkGetBufferMemoryRequirements(vk.device,c.buffer,&req);
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};flags.flags=VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.pNext=&flags;ai.allocationSize=req.size;ai.memoryTypeIndex=FindMemoryType(req.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(vk.device,&ai,nullptr,&c.memory));VK_CHECK(vkBindBufferMemory(vk.device,c.buffer,c.memory,0));
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};info.buffer=c.buffer;c.address=vkGetBufferDeviceAddress(vk.device,&info);
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT,0,24};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};li.pushConstantRangeCount=1;li.pPushConstantRanges=&push;
    VK_CHECK(vkCreatePipelineLayout(vk.device,&li,nullptr,&c.layout));
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};mi.codeSize=sizeof indirect_clamp_spv;mi.pCode=indirect_clamp_spv;
    VkShaderModule module;VK_CHECK(vkCreateShaderModule(vk.device,&mi,nullptr,&module));
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};ci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=module;ci.stage.pName="main";ci.layout=c.layout;
    VK_CHECK(vkCreateComputePipelines(vk.device,PipelineCache(),1,&ci,nullptr,&c.pipeline));
    vkDestroyShaderModule(vk.device,module,nullptr);
}
// ---- frame skip (on by default; OOT_FRAME_SKIP=0 disables) -------------------------------------------------------------
// The game advances its world one fixed step per frame, so below 60 fps everything slows down. With
// frame skip on, frames run their game logic as usual but only some are rendered and shown: draws
// of skipped frames are dropped when they are recurring per-frame work (a render target drawn in
// each of the last 8 frames; one-off renders such as menu icon captures always run). Compute runs. The skip level is chosen so frames average 60 per second.
bool frame_skipping=false; u32 skip_level=0; u64 frame_skip_dropped=0;
bool FrameSkipOn(){static const bool on=[]{const char* e=std::getenv("OOT_FRAME_SKIP");if(e)return std::strcmp(e,"0")!=0;const char* r=std::getenv("OOT_REALTIME");return r&&std::atoi(r)==0;}();return on;} // (default off: the real-time step keeps game speed (present.cpp); OOT_FRAME_SKIP=1 enables)
std::unordered_map<u64,std::pair<u64,u32>> recurring_work; // key -> (last frame it was requested, consecutive frames)
bool RecurringWork(u64 key){
    auto& r=recurring_work[key];
    if(r.first!=frame_number||r.second==0){r.second=(r.first+1==frame_number&&r.second)?r.second+1:1;r.first=frame_number;}
    return r.second>=8;
}
// Whether this draw (current render targets) may be dropped in a skipped frame.
bool SkipDraw(){
    if(!FrameSkipOn())return false;
    u64 key=0;
    for(u32 i=0;i<(*tls_state).color_count&&!key;++i)if((*tls_state).targets.colors[i])if(auto* t=HostTexture((*tls_state).targets.colors[i]))key=t->storage;
    if(!key&&(*tls_state).targets.depth)if(auto* t=HostTexture((*tls_state).targets.depth))key=t->storage;
    if(!key)return false;
    const bool recurring=RecurringWork(key);
    if(frame_skipping&&recurring){++frame_skip_dropped;return true;}
    return false;
}
std::unordered_map<u64,u64> mesh_preclamped; // guest address of a mesh draw's arguments -> their converted slot (within a group of resolved draws)
thread_local u64 mesh_indirect_slot=0; // nonzero while Draw records a DrawMeshTasksIndirect (slot + 1)
// Indirect mesh draws whose arguments were converted for the whole group are resolved on the helper threads like
// other draws (they were ~100 draws a frame run by the executor itself, 1.1 ms in a heavy view).
// OOT_NO_MESH_AHEAD=1: the executor runs each in its place, as before.
std::atomic<u64> mesh_ahead_stats[4]{}; // (resolved ahead, tried and left to the executor, direct mesh draws, indirect ones not tried)
bool MeshAhead(){static const bool on=std::getenv("OOT_NO_MESH_AHEAD")==nullptr;return on;}
struct FastDraw {
    bool valid=false; Pipeline* pipeline=nullptr; std::array<Texture*,8> colors{}; Texture* depth=nullptr; VkExtent2D extent{};
    std::array<VkRenderingAttachmentInfo,8> attachments{}; VkRenderingAttachmentInfo da{}; std::vector<u64> key; u64 flushes=0, pools=0; u32 texture_writes=0;
    bool feedback=false; // (the draw samples one of its own attachments: what the pass does then depends on the depth state)
};
thread_local FastDraw fast_draw;
u64 fast_draw_count=0;std::atomic<u64> fast_texture_patches{0},fast_pipeline_swaps{0};
// Brings the current resolution (descriptor_write_list and its slots, made for `pipeline` by an earlier draw of
// this batch) up to the state of the draw at hand: the uniform buffers bound now, the textures its handles name
// now and, where `all` (the resolution comes from the cache, with other draws in between), its driver banks and a
// check of every texture. False: the draw resolves the usual way.
bool PatchResolved(Pipeline& pipeline,const std::array<Texture*,8>& colors,Texture* depth,bool all){
    // The variants still apply (their specialization constants may live in the rebound buffers).
    for(auto* shader:pipeline.shaders){auto& stage=(*tls_state).stages[ShaderPublicStage[shader->source->stage]];
        if(stage.shader!=shader->source){++fast_draw_fails[3];return false;}
        for(const auto& d:shader->compiled.dependencies)if(CurrentDependencyValue(stage,d)!=d.value){++fast_draw_fails[4];return false;}}
    bool banks=all;
    if(!texture_write_slots.empty()){ // as WriteDescriptors resolves a texture, where nothing is left to bring or record
        if(HandleMemoryPending(pipeline)){++fast_draw_fails[5];return false;}
        for(auto& slot:texture_write_slots){
            const auto& r=*slot.resource;const u32 handle=ResourceHandle(r,*slot.shader,0);
            if(handle==slot.handle&&!all)continue;
            if(!handle||!HandleRegistered(handle)){++fast_draw_fails[5];return false;}
            auto& binding=Binding(handle);auto& t=*binding.host.get();
            // (one of the draw's own targets: a copy to make, or a pass to end. Between draws that follow each other
            // the depth target counts too: what the pass does about it was decided for the first of them.)
            if(!TextureReadyNow(t)||t.layout!=VK_IMAGE_LAYOUT_GENERAL||(!all&&&t==depth)){++fast_draw_fails[6];return false;}
            for(auto* c:colors)if(c==&t){++fast_draw_fails[6];return false;}
            descriptor_reads[slot.read]=&t;
            if(handle==slot.handle&&t.image==slot.image)continue; // (the same texture with the image it had)
            TextureView view=binding.view;view.target=ShaderViewTarget(r.texture_type,t)+1;
            const VkDescriptorImageInfo info{Sampler(handle),ImageView(t,view),VK_IMAGE_LAYOUT_GENERAL};
            if(resolve_bail)return false;
            *const_cast<VkDescriptorImageInfo*>(descriptor_write_list[slot.write].pImageInfo)=info;
            descriptor_reads[slot.read]=&t;slot.handle=handle;slot.image=t.image;++fast_texture_patches;
            banks|=r.bank<UniformBankBase; // (a texture bound directly: its handle is in a driver bank)
        }
    }
    if(all)for(const auto& slot:storage_write_slots){ // (between draws that follow each other a storage binding that changes ends the fast path)
        const auto& r=*slot.resource;
        if(Cbuf(slot.stage,r.bank,r.offset)!=slot.words[0]||Cbuf(slot.stage,r.bank,r.offset+4)!=slot.words[1]||Cbuf(slot.stage,r.bank,r.offset+8)!=slot.words[2]){++fast_draw_fails[7];return false;}
        const auto range=GpuLookupAddress(slot.words[0]|(u64)slot.words[1]<<32);if(resolve_bail)return false;
        const auto& info=*descriptor_write_list[slot.write].pBufferInfo;
        const u64 delta=range.offset&(Vk().properties.limits.minStorageBufferOffsetAlignment-1);
        if(!range.pool||range.buffer!=info.buffer||range.offset-delta!=info.offset||range.cpu_address!=slot.cpu_address){++fast_draw_fails[7];return false;}
        if(UniformOverlayEnabled()){if(resolve_worker&&UnwrittenUniformOverlap(slot.cpu_address,slot.extent))return false;WriteUniformUpdatesOverlapping(slot.cpu_address,slot.extent);} // (read as storage: memory must have them)
    }
    if(banks&&!driver_write_slots.empty()){
        if(!ring.buffer||ring.head+32*DriverBankSize+(4u<<20)>ring.size)return false; // (the ring starts over: the full path's)
        for(const auto& slot:driver_write_slots)*const_cast<VkDescriptorBufferInfo*>(descriptor_write_list[slot.write].pBufferInfo)=DriverBankDescriptor(*slot.stage,slot.bank);
    }
    const u64 alignment=Vk().properties.limits.minUniformBufferOffsetAlignment;
    for(const auto& slot:uniform_write_slots){ // as WriteDescriptors resolves a uniform binding
        Buffer b=slot.stage->uniforms.at(slot.bank-UniformBankBase);
        const auto range=GpuLookupAddress(b.address);if(!range.pool)return false;
        if(!b.size)b.size=std::min<u64>(range.available,64u<<20);
        const u64 delta=range.offset&(alignment-1);
        if(UnwrittenUniformOverlap(range.cpu_address-delta,std::min(b.size,range.available)+delta)){
            if(ring.head+(2u<<20)>ring.size)return false;
            const auto o=UniformOverlay(range.cpu_address-delta,std::min<u64>(std::min(b.size,range.available)+delta,65536+delta));
            if(!o.buffer)return false;
            *const_cast<VkDescriptorBufferInfo*>(descriptor_write_list[slot.write].pBufferInfo)=o;continue;}
        *const_cast<VkDescriptorBufferInfo*>(descriptor_write_list[slot.write].pBufferInfo)={range.buffer,range.offset-delta,std::min(b.size,range.available)+delta};
    }
    return true;
}
bool TryFastDraw(FastDraw& fd){
    if(FlushCount()!=fd.flushes){++fast_draw_fails[0];return false;}
    if(PoolGeneration()!=fd.pools||TextureWriteSequence()!=fd.texture_writes){++fast_draw_fails[1];return false;} // (texture writes: re-upload via the full path)
    if(!PassOpenWithKey(fd.key)){++fast_draw_fails[2];return false;}
    return PatchResolved(*fd.pipeline,fd.colors,fd.depth,false);
}
// ---- resolutions kept per pipeline ---------------------------------------------------------------------------
// The game draws its objects material by material, so most draws follow a draw of another program and resolved
// all their descriptors anew (~6 us each, against under 1 us for a draw that follows one of its own program). The
// resolution a pipeline's last draw made is kept for the batch, on the thread that made it: a later draw of that
// pipeline takes it back and patches it as the fast path does, with its driver banks made again and every texture
// checked. Only resolutions of uniform buffers, driver banks and single textures are kept.
// OOT_RESOLVE_CACHE=1 turns it on (off by default: every such draw resolves anew).
struct ResolveCache {
    u64 pools=~0ull;
    std::vector<VkWriteDescriptorSet> writes;std::vector<VkDescriptorBufferInfo> buffers;std::vector<VkDescriptorImageInfo> images;
    struct Slot{u32 write,stage,bank;};std::vector<Slot> uniforms,drivers;std::vector<TextureSlot> textures;std::vector<StorageSlot> storage;std::vector<u32> storage_stage;
    std::vector<Texture*> reads;std::vector<std::pair<u64,u64>> buffer_reads;
};
thread_local std::unordered_map<const Pipeline*,ResolveCache> resolve_cache;
std::atomic<u64> resolve_cache_hits{0},resolve_cache_misses{0},resolve_cache_why[6]{};
bool ResolveCacheOn(){static const bool on=std::getenv("OOT_RESOLVE_CACHE")!=nullptr;return on;} // (off unless asked for, as FastTextures)
// (the writes' arrays are the cache's own from here on: the lists WriteDescriptors filled are reused by the next draw)
void KeepResolved(const Pipeline& pipeline){
    auto& c=resolve_cache[&pipeline];
    c.pools=PoolGeneration();
    c.writes=descriptor_write_list;c.buffers.clear();c.images.clear();
    for(const auto& w:c.writes){if(w.pBufferInfo)c.buffers.insert(c.buffers.end(),w.pBufferInfo,w.pBufferInfo+w.descriptorCount);
        else c.images.insert(c.images.end(),w.pImageInfo,w.pImageInfo+w.descriptorCount);}
    State& st=*tls_state;const auto index=[&](const StageState* stage){return (u32)(stage-&st.stages[0]);};
    c.uniforms.clear();for(const auto& u:uniform_write_slots)c.uniforms.push_back({u.write,index(u.stage),u.bank});
    c.drivers.clear();for(const auto& d:driver_write_slots)c.drivers.push_back({d.write,index(d.stage),d.bank});
    c.textures=texture_write_slots;c.reads=descriptor_reads;c.buffer_reads.assign(descriptor_buffer_reads.begin(),descriptor_buffer_reads.end());
    c.storage=storage_write_slots;c.storage_stage.clear();for(const auto& q:storage_write_slots)c.storage_stage.push_back(index(q.stage));
}
// The pipeline's kept resolution as the current one, patched for this draw; false: none that holds.
bool TakeResolved(Pipeline& pipeline,const std::array<Texture*,8>& colors,Texture* depth){
    const auto it=resolve_cache.find(&pipeline);if(it==resolve_cache.end()){++resolve_cache_why[1];return false;}
    auto& c=it->second;
    // (kept across batches and frames: every texture is looked up and checked again, the buffers are resolved again.
    // A pool that changed may have another texture or sampler behind a handle.)
    if(c.pools!=PoolGeneration()){++resolve_cache_why[2];return false;}
    // (held in lists of this thread's own: the draw's packet reads them after this returns)
    static thread_local std::vector<VkDescriptorBufferInfo> buffers;static thread_local std::vector<VkDescriptorImageInfo> images;
    buffers=c.buffers;images=c.images;descriptor_write_list=c.writes;
    {size_t b=0,i=0;for(auto& w:descriptor_write_list){if(w.pBufferInfo){w.pBufferInfo=buffers.data()+b;b+=w.descriptorCount;}else{w.pImageInfo=images.data()+i;i+=w.descriptorCount;}}}
    State& st=*tls_state;
    uniform_write_slots.clear();for(const auto& u:c.uniforms)uniform_write_slots.push_back({u.write,&st.stages[u.stage],u.bank});
    driver_write_slots.clear();for(const auto& d:c.drivers)driver_write_slots.push_back({d.write,&st.stages[d.stage],d.bank});
    texture_write_slots=c.textures;descriptor_reads=c.reads;
    storage_write_slots=c.storage;for(size_t k=0;k<storage_write_slots.size();++k)storage_write_slots[k].stage=&st.stages[c.storage_stage[k]];
    descriptor_buffer_reads.assign(c.buffer_reads.begin(),c.buffer_reads.end());descriptor_buffer_writes.clear();descriptor_image_writes.clear();
    descriptor_writes=false;resolve_to_general.clear();descriptor_fast_eligible=true;descriptor_cacheable=true;
    if(!PatchResolved(pipeline,colors,depth,true)){++resolve_cache_why[3];descriptor_fast_eligible=false;descriptor_cacheable=false;return false;}
    pipeline.descriptor_set=VK_NULL_HANDLE; // (a pipeline that pushes its descriptors: as WriteDescriptors leaves it)
    // (what the patch changed is what the next draw of the pipeline starts from)
    c.textures=texture_write_slots;c.reads=descriptor_reads;
    {size_t n=0;for(const auto& w:descriptor_write_list)if(w.pImageInfo)for(u32 k=0;k<w.descriptorCount;++k)c.images[n++]=w.pImageInfo[k];}
    return true;
}
// OOT_SKIP_SHADER_FILE=<path> (diagnostic): draws and dispatches using the programs listed there (hex
// addresses) are dropped; the file is re-read every 30 frames (bisecting an artifact while running).
bool SkippedByFile(u64 address){
    static const char* skip_file=std::getenv("OOT_SKIP_SHADER_FILE");
    if(!skip_file)return false;
    static std::vector<u64> live;static u64 read_at=~0ull;
    if(frame_number/30!=read_at&&!resolve_worker){read_at=frame_number/30;live.clear(); // (re-read by the executor alone: threads resolving draws read the list)
        if(FILE* f=std::fopen(skip_file,"r")){char word[64];while(std::fscanf(f,"%63s",word)==1)live.push_back(std::strtoull(word,nullptr,16));std::fclose(f);}}
    return !live.empty()&&std::find(live.begin(),live.end(),address)!=live.end();
}
// One draw as the command buffer needs it (see Draw).
struct DrawPacket {
    VkPipeline pipeline;VkPipelineLayout layout;bool push;VkDescriptorSet set;
    const VkWriteDescriptorSet* write_data;u32 write_count; // (pushed descriptors: the draw's list, or the copy below)
    std::vector<VkWriteDescriptorSet> writes;std::vector<VkDescriptorBufferInfo> buffers;std::vector<VkDescriptorImageInfo> images;
    VkViewport viewport;VkRect2D scissor;float blend[4];float bias[3];u32 stencil[6];
    u32 stream_mask;VkBuffer vb[16];VkDeviceSize vo[16];
    VkBuffer index_buffer;VkDeviceSize index_offset;
    u32 op;u64 x[8];
    u64 ls_camera=0,ls_camera_size=0;
    mkwvk::SkinnedDraw ls_skin;bool ls_skinned=false; // (a skinned model's draw: the upscalers' actor motion)
    u32 predicate;bool inverted;u8 mesh;/* (1: mesh shader stages; 2: the fallback's draw of the records) */u32 mesh_slot;/* (with 2: its command's slot) */bool colour_writes;u64 indirect_slot;VkQueryPool stats_pool;u32 sample_query;u64 checkpoint;
    // (a draw resolved ahead of its turn, see SegmentOutput: its render pass, for the executor to commit)
    std::array<Texture*,8> colors;Texture* depth_target;VkExtent2D extent;std::array<VkRenderingAttachmentInfo,8> attachments;VkRenderingAttachmentInfo da;u32 color_count;std::vector<u64> key;u64 profile_key;Pipeline* set_pipeline;std::vector<Texture*> to_general;bool end_pass_after;
    void OwnWrites(){
        size_t nb=0,ni=0;for(u32 i=0;i<write_count;++i){if(write_data[i].pBufferInfo)nb+=write_data[i].descriptorCount;else ni+=write_data[i].descriptorCount;}
        writes.assign(write_data,write_data+write_count);buffers.resize(nb);images.resize(ni);nb=ni=0;
        for(auto& w:writes){
            if(w.pBufferInfo){std::memcpy(buffers.data()+nb,w.pBufferInfo,w.descriptorCount*sizeof(VkDescriptorBufferInfo));w.pBufferInfo=buffers.data()+nb;nb+=w.descriptorCount;}
            else{std::memcpy(images.data()+ni,w.pImageInfo,w.descriptorCount*sizeof(VkDescriptorImageInfo));w.pImageInfo=images.data()+ni;ni+=w.descriptorCount;}}
        write_data=writes.data();
    }
};
// (more than the batch's queue of unrecorded draws holds: a packet is not reused before it was recorded)
constexpr u32 DrawPacketRing=2048;
static_assert(DrawPacketRing>=2*AsyncDrawQueue);
// Records a draw into the open render pass (this thread, or the recorder thread: nothing but the packet and what
// only recording touches is read). `other_serial`: OtherWorkSerial() where the draw stands in the recording order.
void RecordDrawPacket(VkCommandBuffer cmd,void* packet,u64 other_serial){
    const DrawPacket& p=*static_cast<const DrawPacket*>(packet);auto& vk=Vk();
    // State already set on this command buffer by the previous draw is not set again (unless
    // other work was recorded in between). OOT_NO_DRAW_STATE_CACHE=1 sets everything per draw.
    static const bool no_cache=std::getenv("OOT_NO_DRAW_STATE_CACHE")!=nullptr;
    struct DrawStateCache{VkCommandBuffer cmd=VK_NULL_HANDLE;u64 serial=~0ull;VkPipeline pipeline=VK_NULL_HANDLE;VkPipelineLayout layout=VK_NULL_HANDLE;VkDescriptorSet set=VK_NULL_HANDLE;
        VkViewport viewport{};VkRect2D scissor{};float blend[4]{};float bias[3]{};u32 stencil[6]{};VkBuffer vb[16]{};VkDeviceSize vo[16]{};bool valid=false;};
    static DrawStateCache cache;
    if(no_cache||cache.cmd!=cmd||cache.serial!=other_serial){cache=DrawStateCache{};cache.cmd=cmd;}
    if(cache.pipeline!=p.pipeline){vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,p.pipeline);cache.pipeline=p.pipeline;}
    if(cache.layout!=p.layout){ // (push constants are the same for every draw; set again for a new layout)
        float push[32];FillPushConstants(push);vkCmdPushConstants(cmd,p.layout,VK_SHADER_STAGE_ALL,0,sizeof push,push);cache.layout=p.layout;cache.set=VK_NULL_HANDLE;}
    if(p.push){if(p.write_count)vk.push_descriptor_set(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,p.layout,0,p.write_count,p.write_data);cache.set=VK_NULL_HANDLE;}
    else if(cache.set!=p.set){vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,p.layout,0,1,&p.set,0,nullptr);cache.set=p.set;}
    if(!cache.valid||std::memcmp(&p.viewport,&cache.viewport,sizeof p.viewport)){vkCmdSetViewport(cmd,0,1,&p.viewport);cache.viewport=p.viewport;}
    if(!cache.valid||std::memcmp(&p.scissor,&cache.scissor,sizeof p.scissor)){vkCmdSetScissor(cmd,0,1,&p.scissor);cache.scissor=p.scissor;}
    if(!cache.valid||std::memcmp(p.blend,cache.blend,sizeof cache.blend)){vkCmdSetBlendConstants(cmd,p.blend);std::memcpy(cache.blend,p.blend,sizeof cache.blend);}
    if(!cache.valid||std::memcmp(p.bias,cache.bias,sizeof cache.bias)){vkCmdSetDepthBias(cmd,p.bias[1],p.bias[2],p.bias[0]);std::memcpy(cache.bias,p.bias,sizeof cache.bias);}
    if(!cache.valid||std::memcmp(p.stencil,cache.stencil,sizeof p.stencil)){
        for(u32 i=0;i<2;++i){VkStencilFaceFlags face=i?VK_STENCIL_FACE_BACK_BIT:VK_STENCIL_FACE_FRONT_BIT;vkCmdSetStencilCompareMask(cmd,face,p.stencil[i*3]);vkCmdSetStencilWriteMask(cmd,face,p.stencil[i*3+1]);vkCmdSetStencilReference(cmd,face,p.stencil[i*3+2]);}
        std::memcpy(cache.stencil,p.stencil,sizeof p.stencil);}
    for(u32 i=0;i<16;++i)if(p.stream_mask>>i&1){
        if(!cache.valid||cache.vb[i]!=p.vb[i]||cache.vo[i]!=p.vo[i]){vkCmdBindVertexBuffers(cmd,i,1,&p.vb[i],&p.vo[i]);cache.vb[i]=p.vb[i];cache.vo[i]=p.vo[i];}}
    cache.valid=true;cache.serial=other_serial;
    // DrawElements*(mode, index type, count, index buffer[, base vertex[, base instance, instances]])
    const auto* x=p.x;const Op op=(Op)p.op;
    // (OOT_MESH_PREDICATES=0: draws of mesh shader stages are not made conditional - they are drawn whether or not the
    // occlusion test before them passed. On the Steam Deck (RADV, Mesa 26.0) the GPU stopped answering at the first
    // area with mesh stages and conditional rendering, gfx and compute ring timeouts; with the predicates evaluated on
    // the CPU it ran - 2026-11-17.)
    static const bool mesh_predicates=[]{const char* e=std::getenv("OOT_MESH_PREDICATES");return e?std::atoi(e)!=0:true;}();
    const bool conditional=p.predicate!=~0u&&(p.mesh!=1||mesh_predicates);
    if(conditional){VkConditionalRenderingBeginInfoEXT ci{VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT};ci.buffer=gpu_counters.predicates;ci.offset=p.predicate*4ull;if(p.inverted)ci.flags=VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT;gpu_counters.begin(cmd,&ci);}
    if(p.stats_pool)vkCmdBeginQuery(cmd,p.stats_pool,0,0);
    if(p.sample_query!=~0u)vkCmdBeginQuery(cmd,samples.pool,p.sample_query,VK_QUERY_CONTROL_PRECISE_BIT);
    const bool indexed=op==Op::DrawElements||op==Op::DrawElementsInstanced||op==Op::DrawElementsBaseVertex;
    const bool instanced=op==Op::DrawElementsInstanced||op==Op::DrawArraysInstanced;
    if(p.checkpoint)vk.checkpoint(cmd,reinterpret_cast<void*>(p.checkpoint));
    if(p.mesh==2)vkCmdDrawIndirect(cmd,mesh_fallback.commands,VkDeviceSize(p.mesh_slot)*16,1,16); // (the records the mesh workgroups took, and the first of them: MeshFallbackDispatch)
    else if(p.mesh){
        if(x[0])Fatal("nonzero first mesh task %llu requires workgroup base support",(unsigned long long)x[0]);
        static const auto draw_mesh=reinterpret_cast<PFN_vkCmdDrawMeshTasksEXT>(vkGetDeviceProcAddr(vk.device,"vkCmdDrawMeshTasksEXT"));
        static const auto draw_mesh_indirect=reinterpret_cast<PFN_vkCmdDrawMeshTasksIndirectEXT>(vkGetDeviceProcAddr(vk.device,"vkCmdDrawMeshTasksIndirectEXT"));
        if(!draw_mesh||!draw_mesh_indirect)Fatal("Vulkan mesh draw entry point unavailable");
        if(p.indirect_slot)draw_mesh_indirect(cmd,indirect_clamp.buffer,p.indirect_slot-1,1,16);
        else draw_mesh(cmd,(u32)x[1],1,1);
    }
    else if(indexed){const VkIndexType types[]={VK_INDEX_TYPE_UINT8_KHR,VK_INDEX_TYPE_UINT16,VK_INDEX_TYPE_UINT32};if(x[1]>2)Fatal("invalid index type %llu",(unsigned long long)x[1]);vkCmdBindIndexBuffer(cmd,p.index_buffer,p.index_offset,types[x[1]]);
        const s32 base_vertex=op==Op::DrawElements?0:(s32)x[4];vkCmdDrawIndexed(cmd,(u32)x[2],instanced?(u32)x[6]:1,0,base_vertex,instanced?(u32)x[5]:0);}
    else vkCmdDraw(cmd,x[2],instanced?x[4]:1,x[1],instanced?x[3]:0);
    if(p.stats_pool)vkCmdEndQuery(cmd,p.stats_pool,0);
    if(p.sample_query!=~0u)vkCmdEndQuery(cmd,samples.pool,p.sample_query);
    if(conditional)gpu_counters.end(cmd);
}
// ---- draws resolved on several threads (ExecuteSegmentsParallel) ----
// What resolving one command-list segment gives: its draws as packets and its counter commands, in command order, for
// the executor to commit; `state` is the segment's state as it goes, and where a command needed the executor (`bail`),
// the state that command starts from.
struct ResolvedItem { u32 kind; u32 index; DrawPacket* packet; bool deferred; u64 condition; bool if_equal; u64 depth; u64 depth_target; }; // kind 0 draw, 1 counter command at `index`, 2 a command the executor runs from snapshot `condition`
struct SegmentOutput {
    State state;
    std::vector<std::unique_ptr<DrawPacket>> packets; size_t used=0;
    std::vector<ResolvedItem> items;
    size_t bail=~size_t{0}; const char* bail_file=""; int bail_line=0;
    std::vector<std::pair<u64,u64>> checked; // guest ranges whose pending GPU writes this segment's resolution asked about (address, size)
    std::vector<std::pair<const char*,int>> reasons; // (where draws needed the executor: statistics)
    std::vector<State> snapshots; size_t snapshots_used=0; // the state at each command the executor runs itself (item kind 2)
    const std::vector<u64>* group_reports=nullptr; // addresses the group's reports write (sorted)
    DrawPacket& NewPacket(){if(used==packets.size())packets.push_back(std::make_unique<DrawPacket>());return *packets[used];}
};
thread_local SegmentOutput* resolve_out=nullptr;
// The game's light probe: a 128x128 colour target (R11G11B10F) it draws the surroundings into again every frame, to light
// the scene by - some 250 draws a frame in Castle Town, a tenth of all. The settings window's Light Probe row draws it
// every 2nd or 4th frame instead: its clears and draws are left out in the frames between, so it keeps the last one's
// picture (the passes that filter it run as before). OOT_PROBE_RATE=<n> overrides.
bool ProbeFrameSkipped(){
    static const int env=std::getenv("OOT_PROBE_RATE")?std::atoi(std::getenv("OOT_PROBE_RATE")):0;
    const int every=env?env:gfx::Current().probe_rate;
    if(every<=1||(*tls_state).color_count!=1||!(*tls_state).targets.colors[0])return false;
    const Texture* t=HostTexture((*tls_state).targets.colors[0]);
    if(!t||t->width!=128||t->height!=128||t->format!=63)return false;
    return CurrentFrameNumber()%(u64)every!=0;
}
thread_local u64 mesh_indirect_source=0; // (with mesh_indirect_slot) the device address of the guest's {task count, first task}
std::atomic<u64> mesh_fallback_draws{0};
// The fallback's part of a mesh draw before its render pass: the records' header is reset, the task program's workgroups
// run (their count the draw's, or the converted indirect arguments), then the mesh workgroups they started. `set`: the
// draw's descriptor set (one layout for the compute stages and the draw). False: the draw is left out.
// The mesh fallback's dispatches for one draw: the task program, the mesh program for the workgroups it started, and the
// draw's command (how many records the mesh workgroups took, and the first of them) copied into the draw's slot.
//
// The records buffer has one allocator. The first fallback draw of a render pass resets it; this ends the pass, as
// every dispatch must come before the pass that draws its records. The mesh draws that follow in the same pass do not
// end it again: their dispatches go into a command buffer that executes before the one the pass is in
// (batch.h SplitForHoist) and keep allocating where the draw before stopped, so each draw's records are one run and its
// command starts at their first (the header's firstVertex is set to the allocator between draws). That holds while
// nothing but draws is recorded (OtherWorkSerial): the dispatches read what the batch's uploads and earlier work left,
// which draws do not change. What a draw took last time (read from its slot once its batch is done) is what it is
// expected to take now: a pass whose draws would fill more than half of the buffer starts over with a reset, as does
// a draw never seen. A draw that takes more than the buffer has left draws what fitted (the translator's check).
// OOT_MESH_HOIST=0: every draw resets and ends the pass (as before).
struct MeshHoist{u64 expected=0;u32 draws=0,words=1;} mesh_hoist; // (words: a record's size in the program that drew last)
std::mutex mesh_taken_mutex;std::unordered_map<u64,u32> mesh_taken; // a draw's key -> the records it took last time
std::atomic<u64> mesh_hoisted{0},mesh_resets{0};
// Work a mesh draw needs before its dispatches (an indirect draw's arguments read and bounded): beside them while the
// pass's dispatches are being put before it, else as light work, which ends the pass.
void MeshHoistOrLight(const std::function<void(VkCommandBuffer)>& record){
    if(const VkCommandBuffer hoist=Vk().mesh_fallback?HoistBuffer():VK_NULL_HANDLE)record(hoist);
    else ImmediateLight(record);
}
bool MeshFallbackDispatch(Pipeline& pipeline,VkDescriptorSet set,const Command& command,u64 indirect_slot,u32& slot_out){
    InitializeMeshFallback();
    auto& f=mesh_fallback;const bool task=pipeline.task_compute!=VK_NULL_HANDLE;const bool indirect=indirect_slot!=0;
    const u32 first=(u32)command.x[0],count=(u32)std::min<u64>(command.x[1],nvshader::MeshFallbackCapacity);
    if(!indirect&&!count)return false;
    if(!task&&first){static int told=0;if(told++<4)RT_LOG("mesh fallback: a mesh program without a task program drawn from workgroup %u is left out",first);return false;}
    if(task){ // (the payload buffer holds the slot map and a payload for each task workgroup that can start mesh workgroups)
        const u64 groups=(nvshader::MeshFallbackCapacity+pipeline.fallback_task_threads-1)/std::max(pipeline.fallback_task_threads,1u);
        const u64 need=u64(nvshader::MeshFallbackCapacity)*4+groups*pipeline.fallback_payload_words*4;
        if(need>f.payload_size){static int told=0;if(told++<4)RT_LOG("mesh fallback: a task program's payloads need %llu MiB (the buffer has %llu): its draws are left out",(unsigned long long)(need>>20),(unsigned long long)(f.payload_size>>20));return false;}}
    const VkPipeline task_pipeline=pipeline.task_compute,mesh_pipeline=pipeline.mesh_compute;const VkPipelineLayout layout=pipeline.layout;
    const u32 task_threads=std::max(pipeline.fallback_task_threads,1u);const u64 source=indirect?mesh_indirect_source:0;
    ++mesh_fallback_draws;
    const u32 slot=f.next_slot++%MeshFallbackSlots;slot_out=slot;
    const u32 words=std::max(pipeline.fallback_record_words,1u);
    const u64 key=(reinterpret_cast<u64>(&pipeline)*0x9e3779b97f4a7c15ull)^(u64(first)<<32|count)^(indirect?0x8000000000000000ull:0);
    const u32 previous_words=mesh_hoist.words;mesh_hoist.words=words;
    const auto record=[&](VkCommandBuffer cmd,bool reset){
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        if(reset){
            // (the header: the draw's command with no vertices, one instance and the first record, the allocator at 0, the
            // mesh dispatch with no workgroups and y, z 1. Nothing else needs a reset: a draw reaches only the records taken
            // for it, the mesh dispatch only the map entries and payloads its task program wrote.)
            vkCmdFillBuffer(cmd,f.records,0,4,0);vkCmdFillBuffer(cmd,f.records,4,4,1);vkCmdFillBuffer(cmd,f.records,8,16,0);vkCmdFillBuffer(cmd,f.records,24,8,1);
        }
        barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
        if(!reset){ // (the header for this draw: its first record past what the draws before took, mesh_advance.comp)
            struct{u64 header;u32 before,now;} advance{f.records_address,previous_words,words};
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,f.advance);
            vkCmdPushConstants(cmd,f.advance_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof advance,&advance);
            vkCmdDispatch(cmd,1,1,1);
            barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,0,1,&barrier,0,nullptr,0,nullptr);
        }
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&set,0,nullptr);
        float push[32];FillPushConstants(push,first,source);vkCmdPushConstants(cmd,layout,VK_SHADER_STAGE_ALL,0,sizeof push,push);
        const auto run=[&](u32 groups){if(indirect)vkCmdDispatchIndirect(cmd,indirect_clamp.buffer,indirect_slot-1);else vkCmdDispatch(cmd,groups,1,1);};
        if(task){
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,task_pipeline);
            run(std::min(count,std::max(1u,nvshader::MeshFallbackCapacity/task_threads)));
            barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,0,1,&barrier,0,nullptr,0,nullptr);
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,mesh_pipeline);
            vkCmdDispatchIndirect(cmd,f.records,VkDeviceSize(nvshader::MeshFallbackDispatchWord)*4); // (the workgroups the task threads started)
        }else{
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,mesh_pipeline);
            run(count);
        }
        // The draw's command (the records taken for it, and their first) into its slot.
        barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
        const VkBufferCopy command_copy{0,VkDeviceSize(slot)*16,16};vkCmdCopyBuffer(cmd,f.records,f.commands,1,&command_copy);
        barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_VERTEX_SHADER_BIT|VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,0,1,&barrier,0,nullptr,0,nullptr);
    };
    static const bool hoisting=[]{const char* e=std::getenv("OOT_MESH_HOIST");return e?std::atoi(e)!=0:true;}();
    constexpr u64 Budget=u64(nvshader::MeshFallbackRecordWords)*4/2; // (half of the buffer by what the draws took last time: twice that fits)
    u64 expected=0;bool known=false;
    {std::lock_guard lock{mesh_taken_mutex};const auto it=mesh_taken.find(key);if(it!=mesh_taken.end()){known=true;expected=u64(it->second)*words*4*3/2+(1u<<20);}}
    const VkCommandBuffer hoist=hoisting?HoistBuffer():VK_NULL_HANDLE;
    if(hoist&&known&&mesh_hoist.expected+expected<=Budget&&mesh_hoist.draws<1024){
        record(hoist,false);mesh_hoist.expected+=expected;++mesh_hoist.draws;++mesh_hoisted;
    }else{
        Immediate([&](VkCommandBuffer cmd){record(cmd,true);});++mesh_resets;
        if(hoisting)SplitForHoist();
        mesh_hoist.draws=0;mesh_hoist.expected=known?expected:Budget; // (a draw never seen may take anything: the next one starts over too)
    }
    // (what it took, for the next time: the slot is the GPU's copy of the draw's command)
    AfterBatch([key,slot,words]{
        const u32 taken=mesh_fallback.commands_map[slot*4];
        if(u64(taken)*words>nvshader::MeshFallbackRecordWords)return; // (not this draw's any more: the ring came round)
        std::lock_guard lock{mesh_taken_mutex};
        if(mesh_taken.size()>(1u<<16))mesh_taken.clear();
        u32& last=mesh_taken[key];last=std::max(taken,last-last/16); // (grows at once, shrinks slowly)
    });
    if(mesh_fallback_draws%20000==0)RT_LOG("mesh fallback: of the last draws %llu kept their render pass open, %llu ended it",(unsigned long long)mesh_hoisted.exchange(0),(unsigned long long)mesh_resets.exchange(0));
    return true;
}
// ---- draws whose pipeline is not made yet ---------------------------------------------------------------------------
// The game draws most objects twice: depth only first (a pre-pass), then shaded where the depth is EQUAL. The shaded
// draw's pipeline is the slow one to make (hundreds of ms on a first sight, and a new area asks for a thousand at
// once). Left out while it compiles, the pre-pass alone left the object as a hole showing the sky; waited for, the
// game froze - frames of 9, 20 and 29 s on a first visit to Kokiri Forest with nothing cached. Now neither: the
// shaded draw names its mesh here, and until its pipeline is there every depth-writing draw of that mesh is left
// out as well. The object appears whole a moment later, and the game keeps running. The frame it is first seen can
// still show the hole once, when its pre-pass pipeline already existed.
// OOT_EQUAL_PASS_WAIT_MS=<ms>: wait for such pipelines instead, as before (2000 was the default).
u32 EqualPassWaitMs(){static const u32 ms=std::getenv("OOT_EQUAL_PASS_WAIT_MS")?(u32)std::atoi(std::getenv("OOT_EQUAL_PASS_WAIT_MS")):0;return ms;}
constexpr u32 PendingMeshSlots=4096;
std::atomic<u64> pending_mesh_key[PendingMeshSlots],pending_mesh_until[PendingMeshSlots];
std::atomic<u64> pending_mesh_latest{0},pending_mesh_marks{0};
u64 MeshKey(const Command& command,bool mesh){
    u64 k=0xcbf29ce484222325ull;const auto mix=[&](u64 v){k=(k^v)*0x100000001b3ull;};
    if(mesh){mix(1);mix(mesh_indirect_source);mix(command.x[0]);mix(command.x[1]);mix(command.x[2]);}
    else{mix(command.op);mix(command.x[2]);mix(command.x[3]);mix(command.x[4]);mix((*tls_state).vertices[0].address);}
    return k|1; // (never 0: an empty slot)
}
void MarkMeshPending(u64 key){
    const u64 until=frame_number+1; // (this frame's later draws and the next frame's)
    pending_mesh_latest.store(until,std::memory_order_relaxed);
    for(u32 probe=0;probe<8;++probe){
        const u32 slot=(u32)(key+probe)&(PendingMeshSlots-1);
        const u64 held=pending_mesh_key[slot].load(std::memory_order_relaxed);
        if(held!=key&&pending_mesh_until[slot].load(std::memory_order_relaxed)>=frame_number)continue; // (another mesh's, still current)
        if(held!=key){pending_mesh_key[slot].store(key,std::memory_order_relaxed);pending_mesh_marks.fetch_add(1,std::memory_order_relaxed);}
        pending_mesh_until[slot].store(until,std::memory_order_relaxed);
        return;
    }
}
bool MeshPending(u64 key){
    for(u32 probe=0;probe<8;++probe){
        const u32 slot=(u32)(key+probe)&(PendingMeshSlots-1);
        if(pending_mesh_key[slot].load(std::memory_order_relaxed)==key)return pending_mesh_until[slot].load(std::memory_order_relaxed)>=frame_number;
    }
    return false;
}
void Draw(const Command& command) {
    const bool profile_cpu = DrawCpuProfiling();
    const auto draw_cpu_start = profile_cpu ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    struct DrawCpuScope {
        bool on; std::chrono::steady_clock::time_point start; u64 shader;
        ~DrawCpuScope() {
            if (!on) return;
            const u64 ns = (u64)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
            draw_cpu_profile.draws.fetch_add(1, std::memory_order_relaxed);
            draw_cpu_profile.total_ns.fetch_add(ns, std::memory_order_relaxed);
            if (draw_was_fast) { draw_cpu_profile.fast_draws.fetch_add(1, std::memory_order_relaxed); draw_cpu_profile.fast_ns.fetch_add(ns, std::memory_order_relaxed); }
            draw_was_fast = false;
            auto& stat = draw_cpu_profile.shaders[shader]; stat.first += ns; ++stat.second;
        }
    } draw_cpu_scope{profile_cpu, draw_cpu_start, (*tls_state).stages[1].shader ? (*tls_state).stages[1].shader->address : 0};
    // OOT_SKIP_SHADER=<hex address>[,...]: drop draws using these programs (finding which draws
    // produce an artifact).
    {static const int skip_kind=std::getenv("OOT_SKIP_DRAWS")?std::atoi(std::getenv("OOT_SKIP_DRAWS")):0; // (diagnostic: 1 mesh draws dropped, 2 all others)
     const bool is_mesh=(Op)command.op==Op::DrawMeshTasks||(Op)command.op==Op::DrawMeshTasksIndirect;if(skip_kind&&(skip_kind==1)==is_mesh)return;}
    // OOT_FS_TWEAK=<fragment shader hex>:<what> (diagnostic): that program's draws with depth writes off (nowrite), the depth
    // test off too (notest), or every colour channel written (colour).
    struct DepthTweak{u64 saved_depth;u32 saved_channel,saved_polygon;float saved_bias[3];bool on;~DepthTweak(){if(on){(*tls_state).depth=saved_depth;(*tls_state).channel=saved_channel;(*tls_state).polygon=saved_polygon;std::memcpy((*tls_state).depth_bias,saved_bias,12);}}} tweak{(*tls_state).depth,(*tls_state).channel,(*tls_state).polygon,{(*tls_state).depth_bias[0],(*tls_state).depth_bias[1],(*tls_state).depth_bias[2]},false};
    {static const char* e=std::getenv("OOT_FS_TWEAK");static const std::vector<u64> fss=[]{std::vector<u64> v;if(e)for(const char* p=e;*p&&*p!=':';){char* end;v.push_back(std::strtoull(p,&end,16));p=*end==','?end+1:end;if(p==end&&*p!=':')break;}return v;}();static const char* what=e&&std::strchr(e,':')?std::strchr(e,':')+1:"";
     if(!fss.empty()&&(*tls_state).stages[1].shader&&std::find(fss.begin(),fss.end(),(*tls_state).stages[1].shader->address)!=fss.end()){tweak.on=true;fast_draw_allowed=false;
        if(!std::strcmp(what,"nowrite"))(*tls_state).depth&=~2ull;else if(!std::strcmp(what,"notest"))(*tls_state).depth&=~3ull;else if(!std::strcmp(what,"colour"))(*tls_state).channel=~0u;
        else if(!std::strncmp(what,"bias",4)){(*tls_state).polygon|=0xfu<<15;(*tls_state).depth_bias[0]=1;(*tls_state).depth_bias[1]=what[4]?(float)std::atof(what+4):16.f;(*tls_state).depth_bias[2]=0;}}}
    // OOT_EQ_BIAS=<units> (a workaround, off by default; after Longshot's RX_VK_EQ_BIAS): a pass shading with depth EQUAL
    // after a depth pre-pass tests LESS_OR_EQUAL with that margin towards the camera instead. Our two programs do not
    // yet give the same depth for mesh-drawn objects (doors, vines, ornaments: white holes); 65536 closes them.
    {static const float eq_bias=std::getenv("OOT_EQ_BIAS")?(float)std::atof(std::getenv("OOT_EQ_BIAS")):0.f;
     const u64 d=(*tls_state).depth;
     if(eq_bias>0&&!tweak.on&&(d&1)&&!(d&2)&&((d>>4)&15)==3){tweak.on=true;fast_draw_allowed=false;
        (*tls_state).depth=(d&~0xf0ull)|(4ull<<4);(*tls_state).polygon|=0xfu<<15;
        (*tls_state).depth_bias[0]=0;(*tls_state).depth_bias[1]=-eq_bias;(*tls_state).depth_bias[2]=0;}}
    static const std::vector<u64> skipped_shaders=[]{std::vector<u64> v;if(const char* e=std::getenv("OOT_SKIP_SHADER"))for(const char* p=e;*p;){char* end;v.push_back(std::strtoull(p,&end,16));p=*end?end+1:end;}return v;}();
    if(!skipped_shaders.empty())for(const auto& st:(*tls_state).stages)if(st.shader&&std::find(skipped_shaders.begin(),skipped_shaders.end(),st.shader->address)!=skipped_shaders.end())return;
    for(const auto& st:(*tls_state).stages)if(st.shader&&SkippedByFile(st.shader->address))return;
    if(SkipDraw())return;
    // A render target the game finalized after recording this draw (its object has no host texture any more): the
    // draw is dropped. (It crashed here when pausing: textures are finalized while the executor is behind.)
    {bool gone=(*tls_state).targets.depth&&!HostTexture((*tls_state).targets.depth);
     for(u32 i=0;i<(*tls_state).color_count;++i)gone|=(*tls_state).targets.colors[i]&&!HostTexture((*tls_state).targets.colors[i]);
     if(gone){static int told=0;if(told++<8)RT_LOG("draw skipped: a render target was finalized before it executed");return;}}
    static const bool gpu_trace=std::getenv("OOT_GPU_TRACE")!=nullptr;
    if(gpu_trace){
        std::string stages;
        for(u32 i=0;i<8;++i)if((*tls_state).stages[i].shader){char word[64];std::snprintf(word,sizeof word," s%u=%llx",i,(unsigned long long)(*tls_state).stages[i].shader->address);stages+=word;}
        RT_LOG("GPU draw frame %llu op %u args %llu %llu %llu%s",(unsigned long long)frame_number,command.op,
            (unsigned long long)command.x[0],(unsigned long long)command.x[1],(unsigned long long)command.x[2],stages.c_str());
    }
    VkQueryPool stats_pool=VK_NULL_HANDLE;
    if(LogThisFrame()){LogDraw(command);static VkQueryPool pool=VK_NULL_HANDLE;
        if(!pool){VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};qi.queryType=VK_QUERY_TYPE_PIPELINE_STATISTICS;qi.queryCount=1;
            qi.pipelineStatistics=VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT|VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT|VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT|VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT|VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT|(Vk().mesh_shader_queries?VK_QUERY_PIPELINE_STATISTIC_TASK_SHADER_INVOCATIONS_BIT_EXT|VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT:0);
            VK_CHECK(vkCreateQueryPool(Vk().device,&qi,nullptr,&pool));}
        stats_pool=pool;}
    const bool mesh=(Op)command.op==Op::DrawMeshTasks;
    if(ProbeFrameSkipped())return;
    // A first GPU lookup can allocate a device copy and record a transfer,
    // which ends an open render pass. Resolve all geometry before beginning it.
    std::array<bool,16> used_streams{};
    std::array<BufferRange,16> vertex_ranges{};
    const auto lookup_cpu_start = profile_cpu ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if(!mesh)for(u32 a=0;a<32;++a)if(((*tls_state).attributes[a]&0x40)&&((*tls_state).attributes[a]&31)<16)
        used_streams[(*tls_state).attributes[a]&31]=true;
    for(u32 i=0;i<16;++i)if(used_streams[i]&&(*tls_state).vertices[i].address)
        vertex_ranges[i]=GpuLookupAddress((*tls_state).vertices[i].address);
    const Op draw_op=(Op)command.op;
    const bool indexed_draw=draw_op==Op::DrawElements||draw_op==Op::DrawElementsInstanced||draw_op==Op::DrawElementsBaseVertex;
    const BufferRange index_range=indexed_draw?GpuLookupAddress(command.x[3]):BufferRange{};
    if (profile_cpu) draw_cpu_profile.lookup_ns.fetch_add((u64)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - lookup_cpu_start).count(), std::memory_order_relaxed);
    if(resolve_bail)return;
    if(indexed_draw&&!index_range.pool){static int told=0;if(told++<16)RT_LOG("draw skipped: index buffer %#llx is not mapped",(unsigned long long)command.x[3]);return;}
    // (a pool without a GPU buffer, or indices past its end: NVIDIA's GPUs read zeros there; AMD's fetch from whatever
    // address that makes, fault page after page and are reset. Steam Deck, 2026-11-16: the geometry engine faulting on
    // pages 0x1000, 0x2000, ... as a save loaded.)
    if(indexed_draw){
        const u64 index_bytes=command.x[2]<<(command.x[1]<3?command.x[1]:0);
        if(!index_range.buffer||index_bytes>index_range.available){static int told=0;if(told++<16)RT_LOG("draw skipped: index buffer %#llx has %s (%llu indices, %llu bytes left in its pool)",(unsigned long long)command.x[3],
            index_range.buffer?"too few bytes":"no GPU buffer",(unsigned long long)command.x[2],(unsigned long long)index_range.available);return;}}
    const u64 indirect_slot=mesh?mesh_indirect_slot:0; // (DrawMeshTasksIndirect: arguments in indirect_clamp's slot - 1)
    // A mesh whose shading pipeline is still being made (MarkMeshPending, below) is left out whole: its depth pre-pass
    // and its shadow draws too, so it is simply not there for the moment instead of a hole in the picture.
    {const u64 d=(*tls_state).depth;
     if(!EqualPassWaitMs()&&(d&3)==3&&((d>>4)&15)!=3&&pending_mesh_latest.load(std::memory_order_relaxed)>=frame_number&&MeshPending(MeshKey(command,mesh))){++stats.pending;++run_note.not_run;return;}}
    // Fast path: draws that differ from the previous one only in uniform/vertex buffer bindings
    // (most of the game's per-object draws) reuse its pipeline, attachments and descriptor writes,
    // patching only the uniform buffer descriptors. OOT_NO_FAST_DRAW=1 disables it.
    static const bool fast_disabled=std::getenv("OOT_NO_FAST_DRAW")!=nullptr;
    auto& fd=fast_draw;
    if(!resolve_worker&&gpu_counters.ready&&samples.enabled)EnsureCounterRings(); // (may flush: then the fast path does not apply)
    {   // (bank 0 draw parameters, see DrawBaseVertexOffset)
        const Op op=(Op)command.op;const auto& a=command.x;
        const u32 base_vertex=op==Op::DrawElementsBaseVertex||op==Op::DrawElementsInstanced?(u32)a[4]:0;
        const u32 base_instance=op==Op::DrawElementsInstanced?(u32)a[5]:op==Op::DrawArraysInstanced?(u32)a[3]:0;
        if(base_vertex!=draw_base_vertex||base_instance!=draw_base_instance){draw_base_vertex=base_vertex;draw_base_instance=base_instance;fast_draw_allowed=false;}
    }
    if(fast_pipeline_stale){fast_pipeline_stale=false;
        if(fast_draw_allowed&&fd.valid){
            if(mesh||fd.feedback)fast_draw_allowed=false;
            else{Pipeline* p=&GetPipeline(false);if(resolve_bail)return;
                if(p==fd.pipeline||(p->push&&p->shaders==fd.pipeline->shaders)){if(p!=fd.pipeline){fd.pipeline=p;++fast_pipeline_swaps;}}
                else fast_draw_allowed=false;}}}
    {static const bool why=std::getenv("OOT_SET_STATS")!=nullptr;if(why){static std::atomic<u64> n[5]{};static std::atomic<u64> at{0};
        ++n[mesh?0:stats_pool?1:!fd.valid?2:!fast_draw_allowed?3:4];
        if(!resolve_worker&&at.load()!=frame_number&&frame_number%300==0){at=frame_number;RT_LOG("draws a frame: mesh %.0f, counted %.0f, no fast draw to follow %.0f, state changed since %.0f, fast tried %.0f",n[0].exchange(0)/300.0,n[1].exchange(0)/300.0,n[2].exchange(0)/300.0,n[3].exchange(0)/300.0,n[4].exchange(0)/300.0);}}}
    const bool fast=!fast_disabled&&!mesh&&!stats_pool&&fast_draw_allowed&&fd.valid&&(resolve_worker||draw_predicate==~0u)&&TryFastDraw(fd);
    if(fast)++fast_draw_count;
    draw_was_fast=fast;
    const auto lookup_t0=std::chrono::steady_clock::now();
    Pipeline* const pipeline_ptr=fast?fd.pipeline:&GetPipeline(mesh);
    if(!fast&&!resolve_worker){const double spent=std::chrono::duration<double>(std::chrono::steady_clock::now()-lookup_t0).count();if(spent>0.0005)hitch.lookup+=spent;}
    if(resolve_bail)return;
    auto& pipeline=*pipeline_ptr;auto& vk=Vk();
    std::array<Texture*,8> colors{};Texture* depth=nullptr;VkExtent2D extent{0,0};
    std::array<VkRenderingAttachmentInfo,8> attachments{};VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    std::vector<u64> key;
    if(fast){colors=fd.colors;depth=fd.depth;extent=fd.extent;attachments=fd.attachments;da=fd.da;key=fd.key;}
    else{
        fd.valid=false; // (this draw rewrites the descriptor lists the saved setup refers to)
    // Shaders with native global memory accesses read the guest address table: bring it up to
        // date with the pools (rewriting it waits for earlier work, so only when they need it).
        for(const auto* shader:pipeline.shaders)if(shader->compiled.global_loads||shader->compiled.global_stores){if(!resolve_worker)GlobalAddressTable();else if(!GlobalAddressTableCurrent()){RESOLVE_BAIL();return;}break;}
        // A draw that shades over a depth pre-pass (depth test EQUAL) and is left out while its pipeline compiles leaves
        // the pre-pass's hole: the object as a cut-out in the sky's colour until the pipeline is there (far trees, doors,
        // seconds on a first visit). With OOT_EQUAL_PASS_WAIT_MS such a draw waits for its pipeline, as it did by default
        // before; now the whole mesh is left out until the pipeline is there (MarkMeshPending).
        if(!pipeline.pipeline.load(std::memory_order_acquire)&&pipeline.layout&&((*tls_state).depth&1)&&(((*tls_state).depth>>4)&15)==3){
            const u32 wait_ms=EqualPassWaitMs();
            if(!wait_ms)MarkMeshPending(MeshKey(command,mesh)); // (left out whole until it is there: see MarkMeshPending)
            if(wait_ms&&!pipeline.waited){ // (once a pipeline: one that takes longer - minutes on some drivers - is left out as before)
                if(resolve_worker){RESOLVE_BAIL();return;} // (the executor waits)
                pipeline.waited=true;
                const auto from=std::chrono::steady_clock::now();const auto deadline=from+std::chrono::milliseconds(wait_ms);
                while(!pipeline.pipeline.load(std::memory_order_acquire)&&std::chrono::steady_clock::now()<deadline)std::this_thread::sleep_for(std::chrono::microseconds(200));
                const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-from).count();hitch.pipeline_wait+=ms/1000;
                static u32 told=0;if(ms>20&&told++<40)RT_LOG("waited %.0f ms for the pipeline of a draw shading over its depth pre-pass (fragment program %#llx)",ms,(unsigned long long)((*tls_state).stages[1].shader?(*tls_state).stages[1].shader->address:0));}}
        if(!pipeline.pipeline.load(std::memory_order_acquire)){if(pipeline.layout)PipelineWorkers().Urgent(&pipeline);++stats.pending;++run_note.not_run;return;} // untranslatable, or still compiling (a draw wants it now: ahead of the others)
        // Anything that may flush runs before the descriptor set is written (below): a flush retires
        // the batch and resets the descriptor pool that set comes from.
        if(!resolve_worker&&gpu_counters.ready&&samples.enabled)EnsureCounterRings();
        if(!gpu_counters.ready&&samples.enabled&&samples.next==SampleCounter::Capacity){Flush();samples.next=0;}
        depth=(*tls_state).targets.depth?HostTexture((*tls_state).targets.depth):nullptr;
        for(u32 i=0;i<(*tls_state).color_count;++i)if((*tls_state).targets.colors[i]){colors[i]=HostTexture((*tls_state).targets.colors[i]);extent={std::max(1u,colors[i]->width>>(*tls_state).targets.color_views[i].levels_base),std::max(1u,colors[i]->height>>(*tls_state).targets.color_views[i].levels_base)};}
        if(!extent.width&&depth)extent={depth->width,depth->height};if(!extent.width){
            // Attachment-less draws (storage writes, queries) rasterize over the viewport area.
            extent={(u32)std::max(1.f,std::fabs((*tls_state).viewport.x)+std::fabs((*tls_state).viewport.width)),(u32)std::max(1.f,std::fabs((*tls_state).viewport.y)+std::fabs((*tls_state).viewport.height))};}
        {// OOT_SKIP_TARGET=<w>x<h>[,...] (diagnostic): draws into targets of these sizes are dropped (what a pass costs).
         static const std::vector<std::pair<u32,u32>> skipped_targets=[]{std::vector<std::pair<u32,u32>> v;if(const char* e=std::getenv("OOT_SKIP_TARGET"))for(const char* p=e;*p;){char* end;const u32 w=std::strtoul(p,&end,10);if(*end!='x')break;const u32 h=std::strtoul(end+1,&end,10);v.push_back({w,h});p=*end?end+1:end;}return v;}();
         for(const auto& [w,h]:skipped_targets)if(extent.width==w&&extent.height==h)return;}
        for(u32 i=0;i<(*tls_state).color_count;++i){auto& a=attachments[i];a.sType=VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;a.imageLayout=VK_IMAGE_LAYOUT_GENERAL;a.loadOp=VK_ATTACHMENT_LOAD_OP_LOAD;a.storeOp=VK_ATTACHMENT_STORE_OP_STORE;if(colors[i])a.imageView=AttachmentView(*colors[i],(*tls_state).targets.color_views[i]);}
        da.imageLayout=VK_IMAGE_LAYOUT_GENERAL;da.loadOp=VK_ATTACHMENT_LOAD_OP_LOAD;da.storeOp=VK_ATTACHMENT_STORE_OP_STORE;if(depth)da.imageView=AttachmentView(*depth,(*tls_state).targets.depth_view);
        const auto descriptor_cpu_start = profile_cpu ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if(resolve_bail)return; // (an attachment view to make)
        static const bool trace_draws=std::getenv("OOT_GPU_TRACE")!=nullptr; // (the dumps of a logged frame are WriteDescriptors' own: LogThisFrame)
        const bool cache=ResolveCacheOn()&&pipeline.push&&!mesh&&!trace_draws&&!LogThisFrame();
        const bool taken=cache&&TakeResolved(pipeline,colors,depth);
        if(resolve_bail)return;
        if(cache)++(taken?resolve_cache_hits:resolve_cache_misses);
        const bool descriptors_ready = taken||WriteDescriptors(pipeline);
        if(cache&&!taken&&descriptors_ready&&descriptor_fast_eligible&&descriptor_cacheable&&!descriptor_writes)KeepResolved(pipeline);
        else if(cache&&!taken&&descriptors_ready)++resolve_cache_why[!descriptor_cacheable?4:5];
        if (profile_cpu) draw_cpu_profile.descriptors_ns.fetch_add((u64)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - descriptor_cpu_start).count(), std::memory_order_relaxed);
        if(!descriptors_ready)return;
        // Allocated only for a draw that is recorded: every slot in the counter range must be reset
        // and written, or GPU accumulation would sum stale results of earlier frames.
        // With GPU counters, occlusion samples are counted per render pass segment (see SplitSegment);
        // the per-draw queries are the CPU fallback.
        if(gpu_trace)RT_LOG("GPU submit");
        // The pass key: everything vkCmdBeginRendering depends on.
        key={extent.width|(u64)extent.height<<32,(*tls_state).color_count,(u64)da.imageView,depth?(u64)depth->info.depth|(u64)depth->info.stencil<<1:0};
        for(u32 i=0;i<(*tls_state).color_count;++i)key.push_back((u64)attachments[i].imageView);
        // Sampling an attachment of the open pass (feedback) needs the pass ended and a barrier.
        // A draw that samples the depth buffer it also tests against without writing it (soft particles, decals, water:
        // ~170 draws a frame in Castle Town) reads what the pass's earlier draws left there. Ending the pass for each
        // made ~170 more render passes a frame, each reloading the 1080p targets. With depth writes off the attachment
        // is read-only for this draw, and the GPUs this runs on rasterize in order, so the pass stays open.
        // OOT_FEEDBACK_END=1: the pass ends, as before.
        static const bool feedback_end=std::getenv("OOT_FEEDBACK_END")!=nullptr;
        static u64 feedback_stats[3]{};static const bool feedback_trace=std::getenv("OOT_FEEDBACK_TRACE")!=nullptr;
        for(auto* t:descriptor_reads){
            if(t==depth){const bool writes_depth=((*tls_state).depth&1)&&(((*tls_state).depth>>1)&1);++feedback_stats[writes_depth?1:0];if(feedback_end||writes_depth){if(resolve_worker){RESOLVE_BAIL();return;}EndPass();}break;}
            bool hit=false;for(auto* c:colors)hit|=c==t;if(hit){++feedback_stats[2];if(resolve_worker){RESOLVE_BAIL();return;}EndPass();break;}}
        if(feedback_trace){static u64 at=0;if(at!=frame_number&&frame_number%300==0){at=frame_number;RT_LOG("feedback draws a frame: %.1f read-only depth, %.1f depth written, %.1f colour",feedback_stats[0]/300.0,feedback_stats[1]/300.0,feedback_stats[2]/300.0);feedback_stats[0]=feedback_stats[1]=feedback_stats[2]=0;}}
    }
    const u32 sample_query=gpu_counters.ready?~0u:AllocateSampleQuery();const VkDescriptorSet set=pipeline.descriptor_set;
    {static const bool pass_stats=std::getenv("OOT_PASS_STATS")!=nullptr; // (diagnostic: the frame's draws by kind of target)
     if(pass_stats&&!resolve_worker){static std::map<std::tuple<u32,u32,u32,u32>,u64> by;static u64 at=0;
        ++by[{(*tls_state).color_count,extent.width,extent.height,colors[0]?colors[0]->format:depth?depth->format:0u}];
        if(at!=frame_number&&frame_number%300==0){at=frame_number;std::vector<std::pair<u64,std::tuple<u32,u32,u32,u32>>> top;for(auto& [k,n]:by)top.push_back({n,k});std::sort(top.rbegin(),top.rend());
            for(size_t i=0;i<top.size()&&i<24;++i)RT_LOG("pass stats: %.1f draws a frame into %u colour target(s) %ux%u format %u",top[i].first/300.0,std::get<0>(top[i].second),std::get<1>(top[i].second),std::get<2>(top[i].second),std::get<3>(top[i].second));
            by.clear();}}}

    const auto begin=[&](VkCommandBuffer cmd){
        // (The barrier opening the pass orders memory; only a layout change needs an image barrier.)
        for(auto* color:colors)if(color){if(color->layout!=VK_IMAGE_LAYOUT_GENERAL)Transition(cmd,*color,VK_IMAGE_LAYOUT_GENERAL);color->gpu_written=true;}
        if(depth){if(depth->layout!=VK_IMAGE_LAYOUT_GENERAL)Transition(cmd,*depth,VK_IMAGE_LAYOUT_GENERAL);depth->gpu_written=true;}
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};ri.renderArea.extent=extent;ri.layerCount=1;ri.colorAttachmentCount=(*tls_state).color_count;ri.pColorAttachments=attachments.data();if(depth&&depth->info.depth)ri.pDepthAttachment=&da;if(depth&&depth->info.stencil)ri.pStencilAttachment=&da;
        if(stats_pool)vkCmdResetQueryPool(cmd,stats_pool,0,1);
        if(sample_query!=~0u)vkCmdResetQueryPool(cmd,samples.pool,sample_query,1);
        vkCmdBeginRendering(cmd,&ri);};
    const u32 predicate=resolve_worker?~0u:draw_predicate;const bool inverted=resolve_worker?false:draw_inverted;if(!resolve_worker)draw_predicate=~0u;
    // What the draw records, by value (DrawPacket): recorded here, or by the recorder thread while this thread
    // resolves the next draw (batch.h, PassDraw's `async`).
    static DrawPacket packet_ring[DrawPacketRing];static u32 packet_next=0;
    DrawPacket& pk=resolve_worker?resolve_out->NewPacket():packet_ring[packet_next++%DrawPacketRing];
    pk.pipeline=pipeline.pipeline;pk.layout=pipeline.layout;pk.push=pipeline.push;pk.set=set;
    pk.write_data=descriptor_write_list.data();pk.write_count=pipeline.push||resolve_worker?(u32)descriptor_write_list.size():0;
    {auto viewport=(*tls_state).viewport;viewport.minDepth=std::clamp(viewport.minDepth,0.f,1.f);viewport.maxDepth=std::clamp(viewport.maxDepth,0.f,1.f);
     // Upper-left origin: NVN maps clip-space +y to the top row, which Vulkan does with a negative-height viewport.
     if(!WindowOriginLowerLeft()){viewport.y+=viewport.height;viewport.height=-viewport.height;}
     pk.viewport=viewport;
     auto scissor=(*tls_state).scissor;scissor.offset.x=std::max(scissor.offset.x,0);scissor.offset.y=std::max(scissor.offset.y,0);scissor.extent.width=std::min(scissor.extent.width,extent.width);scissor.extent.height=std::min(scissor.extent.height,extent.height);
     pk.scissor=scissor;}
    std::memcpy(pk.blend,(*tls_state).blend_color,sizeof pk.blend);std::memcpy(pk.bias,(*tls_state).depth_bias,sizeof pk.bias);
    {const u32 stencil[6]{(*tls_state).stencil_compare[0],(*tls_state).stencil_write[0],(*tls_state).stencil_ref[0],(*tls_state).stencil_compare[1],(*tls_state).stencil_write[1],(*tls_state).stencil_ref[1]};std::memcpy(pk.stencil,stencil,sizeof stencil);}
    // Only the streams this draw's attributes read. A binding left over from earlier draws may
    // name memory the game has since released (it crashed here: "unmapped vertex buffer"); an
    // attribute reading unmapped memory gets zeros (the zero buffer; reads past it are robust).
    pk.stream_mask=0;
    for(u32 i=0;i<16;++i)if(used_streams[i]){
        const BufferRange& range=vertex_ranges[i];
        VkBuffer buffer=range.buffer;VkDeviceSize offset=range.offset;
        const bool unusable=!range.pool||!range.buffer;
        if(unusable&&resolve_worker){RESOLVE_BAIL();return;} // (the zero buffer: the executor's)
        if(unusable){const auto zero=ZeroBuffer();buffer=zero.buffer;offset=0;
            static std::set<std::pair<u64,u64>> told;const u64 fs=(*tls_state).stages[1].shader?(*tls_state).stages[1].shader->address:0;
            if(told.insert({(*tls_state).vertices[i].address,fs}).second&&told.size()<64)RT_LOG("vertex stream %u at %#llx is not mapped (draw with fragment shader %#llx): reads zeros",i,(unsigned long long)(*tls_state).vertices[i].address,(unsigned long long)fs);}
        pk.stream_mask|=1u<<i;pk.vb[i]=buffer;pk.vo[i]=offset;}
    pk.index_buffer=index_range.buffer;pk.index_offset=index_range.offset;
    pk.op=command.op;std::memcpy(pk.x,command.x,sizeof pk.x);
    pk.ls_camera=(*tls_state).stages[0].uniforms[0].address;pk.ls_camera_size=(*tls_state).stages[0].uniforms[0].size; // (the vendored upscalers' scene camera: Longshot's "uniform buffer 3" is slot 0 of the vertex stage here, found by OOT_UPSCALER_SCAN)
    {static const bool scan=std::getenv("OOT_UPSCALER_SCAN")!=nullptr;static int shown=0; // (which uniform buffer holds the scene's camera: the target's size as floats at 0x120)
     if(scan&&!resolve_worker&&extent.width>=1280&&shown<30)for(u32 st=0;st<5;++st)for(u32 b=0;b<18;++b){const auto& u=(*tls_state).stages[st].uniforms[b];float c[0x230/4];
        if(u.address&&u.size>=0x230&&LsReadSubmitted(u.address,c,sizeof c))for(u32 at=0;at+8<=0x230;at+=4)if(at>=0x100&&c[at/4]==(float)extent.width&&c[at/4+1]==(float)extent.height&&shown++<30)
            RT_LOG("upscaler scan: stage %u uniform %u (+%#llx): the target's size at %#x; w row at 0xa0: %.1f %.1f %.1f %.1f",st,b,(unsigned long long)u.size,at,c[0xa0/4],c[0xa4/4],c[0xa8/4],c[0xac/4]);}}
    {static const bool scan=std::getenv("OOT_UPSCALER_SCAN")!=nullptr;static std::set<std::string> seen; // (vertex layouts of the scene's indexed draws, once each)
     if(scan&&!resolve_worker&&!mesh&&extent.width>=960&&CurrentFrameNumber()>1500&&seen.size()<40&&(command.op==(u32)Op::DrawElements||command.op==(u32)Op::DrawElementsBaseVertex)){
        const u64 bones=(*tls_state).stages[0].uniforms[3].size;if(!bones||bones%48||bones==0x100||bones==0x4240)goto ls_scan_done;
        std::string k;for(u32 i=0;i<16;++i){const u32 a=(*tls_state).attributes[i];char b[40];std::snprintf(b,sizeof b," a%u:%08x",i,a);if(a)k+=b;}
        k+=" ubo";for(u32 u=0;u<8;++u){char b[32];std::snprintf(b,sizeof b," %u:%llx",u,(unsigned long long)(*tls_state).stages[0].uniforms[u].size);k+=b;}
        char b[32];std::snprintf(b,sizeof b," stride %u",(*tls_state).streams[0][0]);k+=b;
        // (where it draws: the colour targets' formats and size, the depth state - a pass with this and the last frame's
        // bones that writes a two-channel target would be the game's own motion vectors)
        k+=" ->";for(u32 i=0;i<(*tls_state).color_count;++i)if(colors[i]){char t[48];std::snprintf(t,sizeof t," c%u:f%u %ux%u",i,colors[i]->format,colors[i]->width,colors[i]->height);k+=t;}
        {char t[64];std::snprintf(t,sizeof t," depth %s state %#x",depth?"yes":"no",(u32)(*tls_state).depth&0xff);k+=t;}
        {char t[64];std::snprintf(t,sizeof t," vs %#llx",(unsigned long long)((*tls_state).stages[0].shader?(*tls_state).stages[0].shader->address:0));k+=t;}
        if(seen.insert(k).second)RT_LOG("draw layout:%s",k.c_str());}
     ls_scan_done:;}
    pk.ls_skinned=!mesh&&LsSkinned(command,pk.ls_skin,(*tls_state).stages[0].shader?(*tls_state).stages[0].shader->address:0);
    {static const bool scan=std::getenv("OOT_SKIN_SCAN")!=nullptr; // (diagnostic: which scene draws carry a bone palette, and why the actor motion passes them by)
     if(scan&&!resolve_worker&&CurrentFrameNumber()>1500&&depth&&extent.width>=640){static std::mutex m;static std::map<std::string,u64> seen;static u64 at=0;
        const u64 b3=(*tls_state).stages[0].uniforms[3].size;
        std::string k=mesh?"mesh":"draw";u32 active=0;for(u32 i=0;i<16;++i)if((*tls_state).attributes[i]&0x40)++active;
        char t[200];std::snprintf(t,sizeof t," attrs %u a6 %s ubo3 %llx%s c0 f%u op %u",active,((*tls_state).attributes[6]&0x40)?"on":"off",(unsigned long long)b3,b3&&b3%48==0?"(bones)":"",colors[0]?colors[0]->format:0,command.op);k+=t;
        std::lock_guard l{m};++seen[k];
        if(CurrentFrameNumber()-at>600){at=CurrentFrameNumber();std::vector<std::pair<u64,std::string>> v;for(auto& [a,b]:seen)v.push_back({b,a});std::sort(v.rbegin(),v.rend());
            for(size_t i=0;i<v.size()&&i<30;++i)RT_LOG("skin scan %llu: %s",(unsigned long long)v[i].first,v[i].second.c_str());seen.clear();}}}
    pk.predicate=predicate;pk.inverted=inverted;pk.mesh=mesh?(pipeline.fallback?2:1):0;pk.indirect_slot=indirect_slot;pk.stats_pool=stats_pool;pk.sample_query=sample_query;
    pk.checkpoint=vk.checkpoint?((*tls_state).stages[1].shader?(*tls_state).stages[1].shader->address:0)|1:0;
    const u64 profile_key=(*tls_state).stages[1].shader?(*tls_state).stages[1].shader->address:0;
    if(resolve_worker){ // resolved only: the executor commits it in command order (CommitResolvedDraw)
        if(pk.mesh==2){RESOLVE_BAIL();return;} // (its task and mesh programs run as dispatches first: the executor's)
        pk.OwnWrites();pk.set_pipeline=pipeline.push?nullptr:&pipeline;
        if(fast)pk.to_general.clear();else pk.to_general=resolve_to_general;
        pk.end_pass_after=!fast&&descriptor_writes;
        pk.colour_writes=false;for(u32 i=0;i<(*tls_state).color_count;++i)pk.colour_writes|=colors[i]&&(((*tls_state).channel>>(i*4))&15);
        pk.colors=colors;pk.depth_target=depth;pk.extent=extent;pk.attachments=attachments;pk.da=da;pk.color_count=(*tls_state).color_count;pk.key=key;pk.profile_key=profile_key;
        resolve_out->items.push_back({0,0,&pk,false,0,false,0,0});++resolve_out->used;
        if(!fast){
            fd.valid=pipeline.push&&descriptor_fast_eligible&&!descriptor_writes&&!mesh;
            if(!fd.valid)++slow_draw_why[!pipeline.push?128:!descriptor_fast_eligible?129:descriptor_writes?130:132];
            if(fd.valid){fd.pipeline=&pipeline;fd.colors=colors;fd.depth=depth;fd.extent=extent;fd.attachments=attachments;fd.da=da;fd.key=key;
                fd.flushes=FlushCount();fd.pools=PoolGeneration();fd.texture_writes=TextureWriteSequence();
                fd.feedback=false;for(auto* t:descriptor_reads){if(t==depth)fd.feedback=true;for(auto* c:colors)if(c==t)fd.feedback=true;}}
            fast_draw_allowed=fd.valid;
        }
        return;
    }
    // (the mesh fallback: the task and mesh programs run as dispatches before the pass; the draw then reads their records)
    if(pk.mesh==2&&!MeshFallbackDispatch(pipeline,set,command,indirect_slot,pk.mesh_slot)){++run_note.not_run;return;}
    if(BatchTraceOn()){char note[512];int n=std::snprintf(note,sizeof note,"[draw op%u x=%llx,%llx,%llx,%llx,%llx,%llx fs%llx vs%llx %ux%u vp%.0fx%.0f mesh%u ib%llx+%llx/%llx",pk.op,(unsigned long long)pk.x[1],(unsigned long long)pk.x[2],(unsigned long long)pk.x[3],
        (unsigned long long)pk.x[4],(unsigned long long)pk.x[5],(unsigned long long)pk.x[6],(unsigned long long)profile_key,(unsigned long long)((*tls_state).stages[0].shader?(*tls_state).stages[0].shader->address:0),extent.width,extent.height,pk.viewport.width,pk.viewport.height,(unsigned)pk.mesh,
        (unsigned long long)(u64)pk.index_buffer,(unsigned long long)pk.index_offset,(unsigned long long)index_range.available);
        for(u32 i=0;i<16&&n<440;++i)if(pk.stream_mask>>i&1)n+=std::snprintf(note+n,sizeof note-n," v%u:%llx+%llx/%llx",i,(unsigned long long)(u64)pk.vb[i],(unsigned long long)pk.vo[i],(unsigned long long)vertex_ranges[i].available);
        std::snprintf(note+n,sizeof note-n,"]");BatchTraceNote(note);}
    const auto record=[&](VkCommandBuffer cmd){RecordDrawPacket(cmd,&pk,OtherWorkSerial());};
    if(!mesh)LsNoteDraw(colors[0],depth,extent,pk.ls_camera,pk.ls_camera_size);
    if(pk.ls_skinned)LsNoteSkinned(pk.ls_skin,depth,extent);
    ++run_note.draws;run_note.mesh+=pk.mesh!=0;run_note.colour+=pk.colour_writes; // (the executor only gets here: a thread resolving ahead has returned its packet)
    if(stats_pool||sample_query!=~0u){ // queries are reset outside a render pass: a pass of its own
        Immediate([&](VkCommandBuffer cmd){begin(cmd);record(cmd);vkCmdEndRendering(cmd);},profile_key);
        fd.valid=false;
    }else{
        for(auto* color:colors)if(color)color->gpu_written=true;
        if(depth)depth->gpu_written=true;
        const bool count_samples=gpu_counters.ready&&samples.enabled;
        // What this draw touches (render passes whose draws conflict with nothing since the last full
        // barrier begin without one). Sizes not known here (vertex, index) count as 1 MiB.
        static PassAccess access;
        static const bool tracking=std::getenv("OOT_PASS_HAZARDS")!=nullptr;
        if(tracking){
        access.read_textures.assign(descriptor_reads.begin(),descriptor_reads.end());
        access.written_textures.assign(descriptor_image_writes.begin(),descriptor_image_writes.end());
        for(auto* c:colors)if(c)access.written_textures.push_back(c);
        if(depth)access.written_textures.push_back(depth);
        access.reads=descriptor_buffer_reads;access.writes=descriptor_buffer_writes;
        access.global_read=access.global_write=false;
        for(const auto* shader:pipeline.shaders){access.global_read|=shader->compiled.global_loads;access.global_write|=shader->compiled.global_stores;}
        for(u32 st=0;st<8;++st)if(st!=5&&(*tls_state).stages[st].shader)for(const auto& u:(*tls_state).stages[st].uniforms)if(u.address)access.reads.push_back({u.address,u.address+std::max<u64>(u.size,256)});
        for(const auto& v:(*tls_state).vertices)if(v.address)access.reads.push_back({v.address,v.address+(v.size?v.size:(1u<<20))});
        if((command.op==(u32)Op::DrawElements||command.op==(u32)Op::DrawElementsInstanced||command.op==(u32)Op::DrawElementsBaseVertex)&&command.x[3])
            access.reads.push_back({command.x[3],command.x[3]+(1u<<20)});
        }
        // (the recorder thread reads the packet later: it gets its own copy of the pushed descriptors)
        static const bool async_enabled=AsyncDrawsEnabled();
        const bool async_draw=async_enabled&&!LogThisFrame()&&!gpu_trace;
        static const bool no_copy=std::getenv("OOT_ASYNC_NOCOPY")!=nullptr; // (diagnosis, with OOT_ASYNC_SYNC or _INLINE only)
        if(async_draw&&pk.write_count&&!no_copy)pk.OwnWrites();
        const auto pass_cpu_start = profile_cpu ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        PassDraw(key,[&](VkCommandBuffer cmd){
            begin(cmd);
            if(count_samples){ // the pass instance's first occlusion segment
                const u32 q=samples.next++;vkCmdBeginQuery(cmd,samples.pool,q,VK_QUERY_CONTROL_PRECISE_BIT);gpu_counters.segment_query=q;
                SetPassEndHook([q](VkCommandBuffer c){vkCmdEndQuery(c,samples.pool,q);gpu_counters.segment_query=~0u;});}
        },record,profile_key,tracking&&!mesh_indirect_slot?&access:nullptr,async_draw?RecordDrawPacket:nullptr,&pk);
        if (profile_cpu) draw_cpu_profile.pass_ns.fetch_add((u64)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - pass_cpu_start).count(), std::memory_order_relaxed);
        if(!fast){ // what the next draw may reuse
            fd.valid=pipeline.push&&descriptor_fast_eligible&&!descriptor_writes&&predicate==~0u&&!mesh;
            if(!fd.valid)++slow_draw_why[!pipeline.push?128:!descriptor_fast_eligible?129:descriptor_writes?130:predicate!=~0u?131:132];
            if(fd.valid){fd.pipeline=&pipeline;fd.colors=colors;fd.depth=depth;fd.extent=extent;fd.attachments=attachments;fd.da=da;fd.key=key;
                fd.flushes=FlushCount();fd.pools=PoolGeneration();fd.texture_writes=TextureWriteSequence();
                fd.feedback=false;for(auto* t:descriptor_reads){if(t==depth)fd.feedback=true;for(auto* c:colors)if(c==t)fd.feedback=true;}}
            fast_draw_allowed=fd.valid;
        }
        if(descriptor_writes)EndPass(); // later work must see its storage writes
    }
    static const bool counter_check=std::getenv("OOT_COUNTER_CHECK")!=nullptr;
    if(sample_query!=~0u&&(!gpu_counters.ready||counter_check))AfterFlush([sample_query] {
        u64 value=0;
        VK_CHECK(vkGetQueryPoolResults(Vk().device,samples.pool,sample_query,1,sizeof value,&value,sizeof value,VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT));
        samples.value+=value;
    });
    if(gpu_trace)RT_LOG("GPU completed");
    if(stats_pool){Flush();u64 r[7]{};vkGetQueryPoolResults(vk.device,stats_pool,0,1,sizeof r,r,8,VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT);
        RT_LOG("  stats: vs %llu, clip in %llu, clip out %llu, fs %llu, prims %llu, task %llu, mesh %llu",(unsigned long long)r[1],(unsigned long long)r[2],(unsigned long long)r[3],(unsigned long long)r[4],(unsigned long long)r[0],(unsigned long long)r[5],(unsigned long long)r[6]);}
    if(LogThisFrame())DumpTargets("draw");
    if(LogThisFrame()&&std::getenv("OOT_LUMA_TRACE"))for(u32 i=0;i<(*tls_state).color_count;++i)if((*tls_state).targets.colors[i]){auto* t=HostTexture((*tls_state).targets.colors[i]);
        if((t->format!=63&&t->format!=41)||t->width<16)continue;const auto px=ReadTexels(*t);double sum=0,mx=0;u64 n=0,bad=0;
        const auto f11=[](u32 b,int bits){const int e=b>>(bits-5)&31,m=b&((1<<(bits-5))-1);return e==31?INFINITY:e?std::ldexp(1.0+m/double(1<<(bits-5)),e-15):std::ldexp(m/double(1<<(bits-5)),-14);};
        const u32 stride=t->format==41?8:4;
        for(size_t k=0;k+stride<=px.size();k+=stride*97){double r,g,b;
            if(t->format==41){u16 h[4];std::memcpy(h,&px[k],8);const auto f16=[&](u16 v){const double x=f11(v&0x7fff,15+0);return (v>>15)?-x:x;};r=f16(h[0]);g=f16(h[1]);b=f16(h[2]);}
            else {u32 v;std::memcpy(&v,&px[k],4);r=f11(v&0x7ff,11);g=f11(v>>11&0x7ff,11);b=f11(v>>22&0x3ff,10);}
            const double l=0.2126*r+0.7152*g+0.0722*b;if(!std::isfinite(l)){++bad;continue;}sum+=l;mx=std::max(mx,l);++n;}
        RT_LOG("  luma c%u %ux%u f%u avg %.4g max %.4g nonfinite %llu",i,t->width,t->height,t->format,n?sum/n:0,mx,(unsigned long long)bad);}
}

}
namespace { u64 GuestDeviceAddress(u64 address); }
// Compute: one pipeline per compiled compute variant; dispatches run synchronously.
std::map<u64,std::unique_ptr<Pipeline>> compute_pipelines; // by the translation's identity (guarded by pipeline_tables_mutex)
std::vector<std::unique_ptr<Pipeline>> compute_orphans;
// Creates the pipeline of a pipeline list record without the guest (a prewarm thread) and enters it in the tables the executor's draws and
// dispatches look in. False when it could not be made (its translation is gone from the shader cache, the driver refused it, a large AMD
// compute program that is not in the driver's cache: that takes minutes and is left to the moment the game needs it).
bool PrewarmPipeline(const plist::Record& rec){
    auto& vk=Vk();
    const bool compute=rec.kind==plist::Kind::Compute;
    PipelineDesc desc{};
    if(!compute){
        if(rec.state.size()!=sizeof desc||rec.shaders.empty())return false;
        std::memcpy(&desc,rec.state.data(),sizeof desc);
        if((rec.kind==plist::Kind::Mesh)!=(desc.mesh!=0))return false;
        if(desc.mesh&&(desc.mesh==2)!=vk.mesh_fallback)return false; // (made on a GPU that draws mesh programs the other way)
    }else if(rec.shaders.size()!=1)return false;
    const bool fallback=!compute&&desc.mesh==2;
    std::vector<nvshader::Result> results(rec.shaders.size());
    std::vector<u64> identities;
    for(size_t i=0;i<rec.shaders.size();++i){
        const auto& ref=rec.shaders[i];
        if(ref.stage>7||!ShaderCacheFind(ref.cache_key,ref.identity,ref.stage,results[i]))return false;
        identities.push_back(ref.identity);
    }
    const std::string key=compute?std::string():PipelineKey(desc,identities);
    {std::lock_guard tables{pipeline_tables_mutex};if(compute?compute_pipelines.count(identities[0])!=0:pipelines.count(key)!=0)return true;} // (the game got there first)
    struct Modules{std::vector<VkShaderModule> list;~Modules(){for(auto m:list)vkDestroyShaderModule(Vk().device,m,nullptr);}} modules;
    for(auto& r:results){
        if(r.global_loads||r.global_stores)PatchAddressTable(r.spirv);
        VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};info.codeSize=r.spirv.size()*4;info.pCode=r.spirv.data();
        VkShaderModule module=VK_NULL_HANDLE;
        if(vkCreateShaderModule(vk.device,&info,nullptr,&module)!=VK_SUCCESS)return false;
        modules.list.push_back(module);
    }
    auto pipeline=std::make_unique<Pipeline>();pipeline->record=rec.id;pipeline->touched=false;
    {std::vector<StageResources> stages;
     for(size_t i=0;i<results.size();++i)stages.emplace_back(compute?VK_SHADER_STAGE_COMPUTE_BIT:BindingStage(rec.shaders[i].stage,fallback),&results[i].resources);
     BuildPipelineLayout(*pipeline,stages);}
    VkPipeline created=VK_NULL_HANDLE;
    if(fallback){ // (the task and mesh programs' compute pipelines, and the generated vertex stage's with the fragment program)
        MeshFallbackStages m;
        for(size_t i=0;i<results.size();++i){const u32 st=rec.shaders[i].stage;
            if(st==6){m.task=modules.list[i];m.task_threads=results[i].mesh_threads;}
            if(st==1)m.fragment=modules.list[i];
            if(st==7){m.mesh=modules.list[i];m.mesh_threads=results[i].mesh_threads;pipeline->fallback_payload_words=results[i].mesh_payload_words;pipeline->fallback_record_words=results[i].mesh_record_words;
                if(results[i].fallback_vertex_spirv.empty())break;
                VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};info.codeSize=results[i].fallback_vertex_spirv.size()*4;info.pCode=results[i].fallback_vertex_spirv.data();
                if(vkCreateShaderModule(vk.device,&info,nullptr,&m.vertex)!=VK_SUCCESS)m.vertex=VK_NULL_HANDLE;else modules.list.push_back(m.vertex);}}
        VkPipeline made[3];
        if(CreateMeshFallbackPipelines(desc,pipeline->layout,m,0,false,made)){created=made[0];pipeline->task_compute=made[1];pipeline->mesh_compute=made[2];
            pipeline->fallback=true;pipeline->fallback_task_threads=m.task?std::max(m.task_threads,1u):0;}
    }else if(compute){
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};ci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=modules.list[0];ci.stage.pName="main";ci.layout=pipeline->layout;
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo pin;PinSubgroup(ci.stage,pin);
        static const u64 noopt_words=std::getenv("OOT_NOOPT_WORDS")?std::strtoull(std::getenv("OOT_NOOPT_WORDS"),nullptr,10):40000;
        if(vk.properties.vendorID==0x1002&&noopt_words&&results[0].spirv.size()>noopt_words)ci.flags|=VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT;
        if(vkCreateComputePipelines(vk.device,PipelineCache(),1,&ci,nullptr,&created)!=VK_SUCCESS)created=VK_NULL_HANDLE;
    }else{
        std::vector<VkPipelineShaderStageCreateInfo> stages;
        for(size_t i=0;i<results.size();++i){VkPipelineShaderStageCreateInfo si{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};si.stage=ShaderVkStage[rec.shaders[i].stage];si.module=modules.list[i];si.pName="main";stages.push_back(si);}
        created=CreateGraphicsPipeline(desc,pipeline->layout,stages,0,false);
    }
    const auto discard=[&](VkPipeline p){if(p)vkDestroyPipeline(vk.device,p,nullptr);vkDestroyPipelineLayout(vk.device,pipeline->layout,nullptr);vkDestroyDescriptorSetLayout(vk.device,pipeline->descriptors,nullptr);};
    if(!created){discard(VK_NULL_HANDLE);return false;}
    pipeline->pipeline.store(created,std::memory_order_release);
    std::lock_guard tables{pipeline_tables_mutex};
    const bool placed=compute?compute_pipelines.try_emplace(identities[0],std::move(pipeline)).second:pipelines.try_emplace(key,std::move(pipeline)).second;
    if(!placed)discard(created); // (the game made it while this thread did: pipeline is intact because try_emplace leaves it alone)
    return true;
}
// indirect: guest address of {x, y, z} workgroup counts written by GPU work in the open batch;
// they are read on the GPU (vkCmdDispatchIndirect, and copied into the driver bank's +0x4..+0xc).
// ---- the vendored temporal upscalers (third_party/longshot-upscale; longshot_upscale.cpp) -----------------------------
// Longshot's chain replaces the gameplay upscale network with NVIDIA DLSS or AMD FSR 3.1. It asks three things of the
// renderer: each draw into a scene-sized target with its vertex stage's uniform buffer 3 (the camera) and its depth
// target (LsNoteDraw); the network's dispatch, with its input and output (LsUpscale, from NetworkStandIn); and a call
// each frame (LsFrame). The scene's colour is RGB9E5 on the console, kept here as RGBA16F; the chain is told
// B10G11R11, the format Longshot's renderer has there (it samples it through the view it is handed either way).
bool LsReadSubmitted(u64 address, void* dst, u64 size) {
    const auto range = LookupAddress(address);
    if (!range.pool || size > range.available || !size) return false;
    const u64 cpu = range.cpu_address;
    std::memcpy(dst, reinterpret_cast<const void*>(cpu), size);
    // (uniform data the game replaced after this command was recorded: as the draw's own reads see it)
    for (u64 page = cpu >> 12; page <= (cpu + size - 1) >> 12; ++page)
        if (const auto found = uniform_update_pages.find(page); found != uniform_update_pages.end())
            for (const u32 index : found->second) { const auto& u = pending_uniform_updates[index];
                const u64 from = std::max(cpu, u.address), to = std::min(cpu + size, u.address + u.bytes.size());
                if (from < to) std::memcpy(static_cast<u8*>(dst) + (from - cpu), u.bytes.data() + (from - u.address), to - from); }
    return true;
}
Texture* ls_scene_depth = nullptr; u32 ls_scene_w = 0, ls_scene_h = 0; u64 ls_scene_frame = ~0ull; // (the depth target of the frame's scene draws, by size)
mkwvk::Texture* LsColour(Texture& t) { mkwvk::Texture* l = ls::Texture(t); if (t.decoded_rgb9e5 || l->format == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 || l->format == VK_FORMAT_R16G16B16A16_SFLOAT) l->format = VK_FORMAT_B10G11R11_UFLOAT_PACK32; return l; }
void LsNoteDraw(Texture* colour, Texture* depth, VkExtent2D extent, u64 camera, u64 camera_size) {
    if (!colour || !depth || !camera || !depth->image) return;
    if (!mkwvk::temporal_wanted() && !mkwvk::temporal_fg_inputs() && !mkwvk::fsr3_inputs::wanted()) return;
    static const bool log = std::getenv("OOT_UPSCALER_LOG") != nullptr; static int shown = 0;
    if (log && extent.width >= 960 && shown < 40) { static u64 seen_frame = ~0ull; static u32 in_frame = 0; if (seen_frame != CurrentFrameNumber()) { seen_frame = CurrentFrameNumber(); in_frame = 0; }
        if (in_frame++ < 3) { ++shown; float c[0x230 / 4]{}; const bool read = camera_size >= 0x230 && LsReadSubmitted(camera, c, sizeof c);
            RT_LOG("upscaler: draw into %ux%u vk format %d (ours %u), depth %ux%u; camera %#llx+%#llx read %d: size %.0fx%.0f, w row %.1f %.1f %.1f %.1f; scene size ok %d", extent.width, extent.height, (int)colour->info.vk, colour->format, depth->width, depth->height,
                   (unsigned long long)camera, (unsigned long long)camera_size, (int)read, c[0x120 / 4], c[0x124 / 4], c[0xa0 / 4], c[0xa4 / 4], c[0xa8 / 4], c[0xac / 4], (int)mkwvk::temporal_scene_size(extent.width, extent.height)); } }
    if (!mkwvk::temporal_scene_size(extent.width, extent.height)) return;
    if (ls_scene_frame != CurrentFrameNumber() || extent.width * extent.height >= ls_scene_w * ls_scene_h) { ls_scene_frame = CurrentFrameNumber(); ls_scene_depth = depth; ls_scene_w = extent.width; ls_scene_h = extent.height; }
    ls::SceneDepth(depth);
    mkwvk::temporal_note_scene_draw(LsColour(*colour), extent.width, extent.height, ls::Texture(*depth), camera, camera_size);
    mkwvk::fsr3_inputs::note_scene_draw(LsColour(*colour), extent.width, extent.height, ls::Texture(*depth), camera, camera_size); // (frame generation's inputs)
}
// A skinned draw into the scene, for the chain's actor motion (Longshot's vk_draw.cpp skin_check, for this renderer's
// state): the game's characters and what moves by itself are indexed draws of one vertex stream - the position at
// attribute 0 (RGB32F or RGBA16F), the weights at 4 (and 5), the joints at 6 (and 7), the bone palette (3 x 4 matrices)
// in the vertex stage's uniform slot 3 and the object's origin in slot 1 (at 0x40). What it reads is the state of the
// draw being resolved, so a thread resolving ahead fills it too; the executor notes it when the draw is committed.
void SkinRejectReport(){static const bool on=std::getenv("OOT_SKIN_SCAN")!=nullptr;static u64 at=0;if(!on||CurrentFrameNumber()-at<600)return;at=CurrentFrameNumber();
    RT_LOG("skin check: not wanted %llu, not indexed %llu, position %llu, bones %llu, joints %llu, stride/offsets %llu, index type %llu, accepted %llu; noted: no depth %llu, not scene size %llu, off %llu, noted %llu",
        (unsigned long long)skin_reject[0].exchange(0),(unsigned long long)skin_reject[1].exchange(0),(unsigned long long)skin_reject[2].exchange(0),(unsigned long long)skin_reject[3].exchange(0),(unsigned long long)skin_reject[4].exchange(0),(unsigned long long)skin_reject[5].exchange(0),(unsigned long long)skin_reject[6].exchange(0),(unsigned long long)skin_reject[7].exchange(0),
        (unsigned long long)skin_reject[8].exchange(0),(unsigned long long)skin_reject[9].exchange(0),(unsigned long long)skin_reject[10].exchange(0),(unsigned long long)skin_reject[11].exchange(0));}
bool LsSkinned(const Command& command, mkwvk::SkinnedDraw& k, u64 program) {
    SkinRejectReport();
    if (!mkwvk::temporal_wanted() && !mkwvk::temporal_fg_inputs()) { ++skin_reject[0]; return false; }
    const Op op = (Op)command.op;
    // (this game version draws its characters as instanced draws of one instance: x[6] the count, x[5] the first)
    const bool one_instance = op == Op::DrawElementsInstanced && command.x[6] == 1 && command.x[5] == 0;
    if (op != Op::DrawElements && op != Op::DrawElementsBaseVertex && !one_instance) { ++skin_reject[1]; return false; }
    const State& st = *tls_state;
    struct Attr { u32 fmt, offset, stream; bool on; };
    const auto attr = [&](int i) { const u32 a = st.attributes[i]; return Attr{a >> 21 & 0x3ff, a >> 7 & 0x3fff, a & 31, (a & 0x40) != 0}; };
    enum : u32 { R8 = 0x1, R8UI = 0x3, R16UI = 0x8, RG8 = 0xd, RG8UI = 0xf, RGB32F = 0x22, RGBA8 = 0x25, RGBA8UI = 0x27, RGBA16F = 0x29, RGBA16UI = 0x2c };
    const Attr pos = attr(0);
    if (!pos.on || (pos.fmt != RGB32F && pos.fmt != RGBA16F) || pos.stream != 0 || !attr(6).on) { ++skin_reject[2]; return false; }
    const Buffer& bones = st.stages[0].uniforms[3];
    if (!bones.address || bones.size < 48 || bones.size % 48) { ++skin_reject[3]; return false; }
    k = mkwvk::SkinnedDraw{};
    k.formats = pos.fmt == RGBA16F ? dlss_actor::kHalfPosition : 0;
    for (int set = 0; set < 2; ++set) {
        const Attr w = attr(4 + set), j = attr(6 + set);
        if (!j.on) break;
        const u32 wf = w.on ? w.fmt : 0;
        if (set == 1 && wf == R8 && (j.fmt == R8UI || j.fmt == R16UI) && !w.stream && !j.stream) { // (a fifth bone: one influence)
            k.formats |= dlss_actor::kSingle << set; if (j.fmt == R8UI) k.formats |= dlss_actor::kByteJoints << set;
            k.weights[set] = w.offset; k.joints[set] = j.offset; k.sets = set + 1; break; }
        if (set == 0 && !wf && (j.fmt == R8UI || j.fmt == R16UI) && !j.stream) { // (a rigid model on one bone: a joint, no weight)
            k.formats |= (dlss_actor::kSingle | dlss_actor::kNoWeight) << set; if (j.fmt == R8UI) k.formats |= dlss_actor::kByteJoints << set;
            k.weights[set] = 0; k.joints[set] = j.offset; k.sets = set + 1; break; }
        if (wf == RG8 && j.fmt == RG8UI && !w.stream && !j.stream) { // (two bones)
            k.formats |= (dlss_actor::kPair | dlss_actor::kByteJoints) << set; k.weights[set] = w.offset; k.joints[set] = j.offset; k.sets = set + 1; continue; }
        if (wf != RGBA8 || (j.fmt != RGBA16UI && j.fmt != RGBA8UI) || w.stream || j.stream || w.offset % 4 || j.offset % 4) break;
        if (j.fmt == RGBA8UI) k.formats |= dlss_actor::kByteJoints << set;
        k.weights[set] = w.offset; k.joints[set] = j.offset; k.sets = set + 1;
    }
    const u32 stride = st.streams[0][0];
    if (!k.sets) { ++skin_reject[4]; { static int shown = 0; static const bool scan = std::getenv("OOT_SKIN_SCAN") != nullptr; if (scan && shown++ < 12) { std::string d; for (int i = 0; i < 10; ++i) { const Attr a = attr(i); char b[48]; std::snprintf(b, sizeof b, " a%d:%s fmt %#x off %u s%u", i, a.on ? "on" : "off", a.fmt, a.offset, a.stream); d += b; } RT_LOG("skin check: joints not understood:%s", d.c_str()); } } return false; }
    if (!stride || stride % 4 || pos.offset % 4 || !st.vertices[0].address) { ++skin_reject[5]; return false; }
    if (command.x[1] > 2) { ++skin_reject[6]; return false; }
    ++skin_reject[7];
    k.vbuf = st.vertices[0].address; k.vsize = st.vertices[0].size; k.stride = stride; k.pos = pos.offset;
    k.ibuf = command.x[3]; k.itype = (int)command.x[1]; k.count = (int)command.x[2]; k.base_vertex = op == Op::DrawElementsBaseVertex || one_instance ? (int)(s32)command.x[4] : 0;
    k.bones = bones.address; k.bones_size = bones.size; k.object = st.stages[0].uniforms[1].address; k.object_size = st.stages[0].uniforms[1].size;
    k.program = program;
    return true;
}
void LsNoteSkinned(mkwvk::SkinnedDraw& k, Texture* depth, VkExtent2D extent) {
    static const bool off = std::getenv("OOT_ACTOR_MOTION") && !std::atoi(std::getenv("OOT_ACTOR_MOTION")); // (=0: the camera's motion alone)
    if (off) { ++skin_reject[10]; return; }
    if (!depth || !depth->image) { ++skin_reject[8]; return; }
    if (!mkwvk::temporal_scene_size(extent.width, extent.height)) { ++skin_reject[9]; return; }
    ++skin_reject[11];
    k.depth = ls::Texture(*depth);
    mkwvk::temporal_note_skinned_draw(k);
}
bool LsWanted() { return mkwvk::temporal_wanted() || mkwvk::temporal_fg_inputs() || mkwvk::fsr3_inputs::wanted(); }
// The gameplay network's dispatch: true when an engine of the chain wrote the picture.
bool LsUpscale(Texture& in, Texture& out, u64 key) {
    if (!LsWanted()) return false;
    Texture* depth = ls_scene_frame == CurrentFrameNumber() && ls_scene_w == in.width && ls_scene_h == in.height ? ls_scene_depth : nullptr;
    bool drawn = false;
    Immediate([&](VkCommandBuffer cmd) {
        Transition(cmd, in, VK_IMAGE_LAYOUT_GENERAL); Transition(cmd, out, VK_IMAGE_LAYOUT_GENERAL);
        if (depth && depth->image) Transition(cmd, *depth, VK_IMAGE_LAYOUT_GENERAL);
        mkwvk::Texture* lin = LsColour(in); mkwvk::Texture* lout = LsColour(out);
        TextureView storage; storage.swizzle[0] = 2; storage.swizzle[1] = 3; storage.swizzle[2] = 4; storage.swizzle[3] = 5; // (a storage view: identity)
        const VkImageView in_view = ImageView(in), out_view = ImageView(out, storage);
        // (as Longshot's renderer: FSR 3.1's own path and frame generation's inputs first, then the chain - with no engine
        // it makes frame generation's motion and leaves the picture to the network)
        // (the pass in the upscale's place at 100%: the same size in and out, the scene's sqrt-encoded view (64) into the
        // picture (63). The chain takes an equal-sized pass only when told it is the gameplay one)
        const bool known = in.format == 64 && out.format == 63 && in.width == out.width && in.height == out.height;
        drawn = mkwvk::fsr3_upscale(cmd, key, known, lin, in_view, lout, out_view, 0) ||
                ((mkwvk::temporal_wanted() || mkwvk::temporal_fg_inputs()) && mkwvk::temporal_upscale(cmd, key, known, lin, in_view, lout, out_view, 0));
    });
    { static const bool log = std::getenv("OOT_UPSCALER_LOG") != nullptr; static int shown = 0;
      if (log && shown++ < 12) RT_LOG("upscaler: network %#llx: in %ux%u vk %d, out %ux%u vk %d, scene depth %s: %s", (unsigned long long)key, in.width, in.height, (int)in.info.vk, out.width, out.height, (int)out.info.vk, depth ? "known" : "not known", drawn ? "drawn by the engine" : "not taken"); }
    if (drawn) out.gpu_written = true;
    return drawn;
}
void LsFrame() {
    ls::Start(); ls::SyncOptions(); mkwvk::temporal_idle();
    static u32 n = 0; // (the chain's own status every 600 frames while an engine draws: jitter, actor motion, its images)
    if (++n % 600 == 0) { char b[512]; b[0] = 0; if (mkwvk::temporal_status(b, sizeof b)) RT_LOG("upscaler status%s", b); }
}
// ---- the game's neural upscaler, where its matrix multiplies are slow --------------------------------------
// OoT upscales movies and gameplay with networks of compute passes built on the console's matrix unit (HMMA). The
// translator expands each multiply in software: fine on NVIDIA, but on AMD (the Steam Deck) a pass took up to
// 113 ms and its pipeline 3.5 minutes to build (a black screen at 3 fps). There, as Hookshot does (b306c4d), the
// network's passes are left out and the pass writing the output image becomes a linear upscale of its input: the
// largest colour texture the pass samples, into the largest image it writes (not also sampled).
// OOT_NETWORKS=1: always run them; =0: always the stand-in (also on NVIDIA, to compare).
bool IsNetwork(const ShaderCode& code){
    static std::unordered_map<const ShaderCode*,bool> known;
    auto it=known.find(&code);if(it!=known.end())return it->second;
    u32 hits=0;for(size_t at=0;at+16<=code.code.size();at+=16){u32 w;std::memcpy(&w,code.code.data()+at,4);hits+=(w&0xfff)==0x23c;}
    const bool network=hits>=4;
    if(network){static int shown=0;if(shown++<32)RT_LOG("compute program %#llx is an upscaling network (%u matrix multiplies)",(unsigned long long)code.address,hits);}
    return known[&code]=network;
}
bool SlowNetworks(){
    if(gfx::NetworksOff())return true; // (the Anti-aliasing row at Off or FXAA: a plain upscale in the networks' place)
    if(mkwvk::temporal_wanted())return true; // (an engine of the vendored chain draws the picture: LsUpscale)
    const int choice=gfx::Current().ai_upscaler; // (the settings window: 1 off, 2 on; 0 as below; read every call, so it is live)
    if(choice)return choice==1;
    static const bool slow=[]{const char* e=std::getenv("OOT_NETWORKS");if(e)return std::atoi(e)==0;
        const u32 vendor=Vk().properties.vendorID;const bool s=vendor==0x1002||vendor==0x8086;
        if(s)RT_LOG("upscaling networks: a linear upscale instead (this GPU's matrix multiplies are emulated slowly)");return s;}();
    return slow;
}
void NetworkStandIn(ShaderVariant& v,bool inputs_only=false){ // (inputs_only: the network itself runs; frame generation's depth and motion are made here)
    std::vector<Texture*> images,textures;
    for(const auto& r:v.compiled.resources){
        if(r.kind!=nvshader::ResourceKind::Image&&r.kind!=nvshader::ResourceKind::Texture)continue;
        for(u32 e=0;e<r.count;++e){const u32 handle=ResourceHandle(r,v,e);if(!handle)continue;
            Texture* t=Binding(handle).host.get();if(!t)continue;(r.kind==nvshader::ResourceKind::Image?images:textures).push_back(t);}
    }
    const auto area=[](const Texture* t){return (u64)t->width*t->height;};
    const auto has=[](const std::vector<Texture*>& list,const Texture* t){for(auto* x:list)if(x==t||(x->storage&&x->storage==t->storage))return true;return false;};
    Texture* out=nullptr;for(auto* t:images)if(t->depth<=1&&!has(textures,t)&&(!out||area(t)>area(out)))out=t;
    {static const bool trace=std::getenv("OOT_STANDIN_TRACE")!=nullptr;static std::map<u64,u32> seen; // (every network program once: what it binds)
     if(trace&&seen[v.source->address]++==0){std::string c;
        for(auto* t:textures){char b[96];std::snprintf(b,sizeof b," [tex %ux%ux%u f%u w%d]",t->width,t->height,t->depth,t->format,(int)t->gpu_written);c+=b;}
        for(auto* t:images){char b[96];std::snprintf(b,sizeof b," [img %ux%ux%u f%u]",t->width,t->height,t->depth,t->format);c+=b;}
        RT_LOG("network %#llx frame %llu binds:%s",(unsigned long long)v.source->address,(unsigned long long)CurrentFrameNumber(),c.c_str());}}
    if(!out)return; // (an earlier pass: feature maps in buffers)
    Texture* in=nullptr;for(auto* t:textures)if(t->depth<=1&&t->target==out->target&&!has(images,t)&&!t->info.depth&&(!in||area(t)>area(in)))in=t;
    // Some networks first convert the decoded movie into an equally sized
    // intermediate, then upscale it in a later pass. Keep that conversion in
    // the stand-in chain too, or the final upscale reads an unwritten image.
    if(!in||area(in)>area(out))return;
    EnsureUploaded(*in);EnsureImage(*out);
    if(area(out)>=area(in)&&in->width>=960&&LsUpscale(*in,*out,v.identity))return; // (DLSS or FSR 3.1 in the network's place)
    if(inputs_only)return;
    auto& vk=Vk();VkFormatProperties fi,fo;vkGetPhysicalDeviceFormatProperties(vk.physical,in->info.vk,&fi);vkGetPhysicalDeviceFormatProperties(vk.physical,out->info.vk,&fo);
    if(!(fi.optimalTilingFeatures&VK_FORMAT_FEATURE_BLIT_SRC_BIT)||!(fi.optimalTilingFeatures&VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)||!(fo.optimalTilingFeatures&VK_FORMAT_FEATURE_BLIT_DST_BIT)){
        static int told=0;if(told++<8)RT_LOG("upscaling network stand-in: formats %d -> %d cannot be blitted",(int)in->info.vk,(int)out->info.vk);return;}
    {static int shown=0;if(shown++<8)RT_LOG("upscaling network stand-in: %ux%u fmt %u -> %ux%u fmt %u",in->width,in->height,in->format,out->width,out->height,out->format);}
    {   // (OOT_STANDIN_TRACE: every candidate, when the choice changes)
        static const bool trace=std::getenv("OOT_STANDIN_TRACE")!=nullptr;
        static std::map<u64,std::pair<const Texture*,const Texture*>> last;static int told=0;
        auto& l=last[v.source->address];
        if(trace&&(l.first!=in||l.second!=out)&&told++<200){l={in,out};
            std::string c;for(auto* t:textures){char b[160];std::snprintf(b,sizeof b," [tex %p %ux%u f%u st %#llx w%d u%d img %p]",(void*)t,t->width,t->height,t->format,(unsigned long long)t->storage,(int)t->gpu_written,(int)t->uploaded,(void*)t->image);c+=b;}
            for(auto* t:images){char b[160];std::snprintf(b,sizeof b," [img %p %ux%u f%u st %#llx w%d]",(void*)t,t->width,t->height,t->format,(unsigned long long)t->storage,(int)t->gpu_written);c+=b;}
            RT_LOG("stand-in %#llx frame %llu: in %p out %p;%s",(unsigned long long)v.source->address,(unsigned long long)CurrentFrameNumber(),(void*)in,(void*)out,c.c_str());}
    }
    Texture* src=in;Texture* dst=out;
    // The picture's network (gameplay): its input is the scene at 1080p as RGB9E5 holding the square roots of the linear
    // values, its output the 1440p picture as R11G11B10F, linear. A blit left them as they were: the whole picture pale
    // and washed out with the network off (means measured: the output is the input to the power 1.94). The stand-in there
    // is the present path's Catmull-Rom filter with the values squared. OOT_STANDIN_CURVE=0: as they are.
    // The title's chain has two steps, 1080p RGBA8 (fmt 37) -> 1080p RGB10A2 (fmt 61) -> 1440p RGBA8 sRGB (fmt 56). The
    // network writes the last image's bytes as they are (a storage view has no sRGB encoding); a blit into an sRGB image
    // encodes them once more, and the title was pale (frame means over the same 40 s: 101 against the network's 50).
    // The last step goes through the same filter with the values as they are. OOT_STANDIN_TITLE=0: a blit as before.
    // (NVIDIA only by default: on a Steam Deck (RADV) the first draw that sampled the image written this way never
    // finished, 2026-11-16 runs 10 and 11. There the title stays a blit, pale.)
    static const bool title_curve=std::getenv("OOT_STANDIN_TITLE")?std::atoi(std::getenv("OOT_STANDIN_TITLE"))!=0:Vk().properties.vendorID==0x10DE;
    if((in->format==64&&out->format==63)||(title_curve&&in->format==61&&out->format==56)){
        const VkDescriptorSetLayout layout=UpscaleSetLayout();
        if(layout){
            TextureView from;from.present=true;from.levels_count=1;from.layers_count=1;from.target=(u32)Target::T2D+1;from.swizzle[0]=2;from.swizzle[1]=3;from.swizzle[2]=4;from.swizzle[3]=5;
            TextureView to=from;if(dst->format==56)to.format=37; // (an sRGB image is written through its UNORM view, as the network writes it)
            const VkImageView src_view=ImageView(*src,from),dst_view=ImageView(*dst,to);
            const VkDescriptorSet set=AllocateBatchSet(layout);
            const VkExtent2D extent{dst->width,dst->height};const int mode=area(out)>area(in)?1:0;
            static const int scene_curve=std::getenv("OOT_STANDIN_CURVE")?std::atoi(std::getenv("OOT_STANDIN_CURVE")):2;
            const int curve=in->format==64?scene_curve:0;
            Immediate([=](VkCommandBuffer cmd){
                Transition(cmd,*src,VK_IMAGE_LAYOUT_GENERAL);Transition(cmd,*dst,VK_IMAGE_LAYOUT_GENERAL);
                UpscaleCopy(cmd,set,src_view,dst_view,extent,mode,curve);});
            out->gpu_written=true;return;}
    }
    Immediate([src,dst](VkCommandBuffer cmd){
        Transition(cmd,*src,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);Transition(cmd,*dst,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageBlit b{};b.srcSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};b.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};
        b.srcOffsets[1]={(s32)src->width,(s32)src->height,1};b.dstOffsets[1]={(s32)dst->width,(s32)dst->height,1};
        vkCmdBlitImage(cmd,src->image,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,dst->image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&b,VK_FILTER_LINEAR);
    });
    out->gpu_written=true;
}
void Dispatch(u32 gx,u32 gy,u32 gz,u64 indirect=0){
    // (Compute is never skipped: exposure/luminance measurement and simulations run every frame on
    // the last rendered images; skipping them left stale exposure: strongly exaggerated bloom.)
    dispatch_grid[0]=gx;dispatch_grid[1]=gy;dispatch_grid[2]=gz;
    if((*tls_state).root[1][0][2]<8192&&(*tls_state).root[1][0][3]<8192&&(*tls_state).root[1][0][2]>16)frame_render_width=std::max(frame_render_width,(*tls_state).root[1][0][2]+1); // (compute root: render size - 1)
    auto& stage=(*tls_state).stages[5];if(!stage.shader)Fatal("compute dispatch without a compute shader");
    if(SkippedByFile(stage.shader->address))return;
    struct Phases{std::chrono::steady_clock::time_point t0=std::chrono::steady_clock::now(),last=t0;double ms[6]{};u64 shader=0;
        void mark(int i){const auto now=std::chrono::steady_clock::now();ms[i]+=std::chrono::duration<double,std::milli>(now-last).count();last=now;}
        ~Phases(){const double total=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
            if(total>100){static std::atomic<u32> shown{0};if(shown++<30)RT_LOG("slow dispatch of shader %#llx: %.0f ms (shader %.0f, address table %.0f, pipeline %.0f, descriptors %.0f, recording %.0f)",
                (unsigned long long)shader,total,ms[0],ms[1],ms[2],ms[3],ms[4]);}}} phases;phases.shader=stage.shader->address;
    auto& v=CompileShader(*stage.shader,0,nullptr);auto& vk=Vk();phases.mark(0);
    if(!v.module)return; // not translatable
    ++frame_dispatches;frame_dispatch_hash=(frame_dispatch_hash^v.identity^((u64)gx<<40)^((u64)gy<<20)^gz)*0x100000001b3ull;
    if(IsNetwork(*stage.shader)){++frame_net;frame_net_hash=(frame_net_hash^v.identity)*0x100000001b3ull;}
    {static const bool scan=std::getenv("OOT_UPSCALER_SCAN")!=nullptr;static std::map<u64,u32> seen; // (every compute program once: what it samples and writes)
     if(scan&&CurrentFrameNumber()>1500&&seen[v.identity]++==0){std::string c;
        for(const auto& r:v.compiled.resources){if(r.kind!=nvshader::ResourceKind::Image&&r.kind!=nvshader::ResourceKind::Texture)continue;
            for(u32 e=0;e<r.count;++e){const u32 handle=ResourceHandle(r,v,e);if(!handle)continue;Texture* t=Binding(handle).host.get();if(!t)continue;
                char b[96];std::snprintf(b,sizeof b," [%s %ux%u f%u]",r.kind==nvshader::ResourceKind::Image?"img":"tex",t->width,t->height,t->format);c+=b;}}
        RT_LOG("compute %#llx at %#llx (network %d, frame %llu):%s",(unsigned long long)v.identity,(unsigned long long)stage.shader->address,(int)IsNetwork(*stage.shader),(unsigned long long)CurrentFrameNumber(),c.c_str());}}
    {static const u64 dump=std::getenv("OOT_DUMP_COMPUTE")?std::strtoull(std::getenv("OOT_DUMP_COMPUTE"),nullptr,16):0;static int dumps=0; // (diagnostic: the textures a compute program reads, as raw files in the data folder)
     if(dump&&v.identity==dump&&CurrentFrameNumber()>1500&&CurrentFrameNumber()%90==0&&dumps<4){++dumps;int index=0;
        for(const auto& r:v.compiled.resources){if(r.kind!=nvshader::ResourceKind::Image&&r.kind!=nvshader::ResourceKind::Texture)continue;
            for(u32 e=0;e<r.count;++e){const u32 handle=ResourceHandle(r,v,e);if(!handle)continue;Texture* t=Binding(handle).host.get();if(!t||!t->image||t->decoded_astc)continue;
                const std::vector<u8> texels=ReadTexels(*t);char path[512];std::snprintf(path,sizeof path,"%s/dump%d-%d-%ux%u-f%u-%s.raw",rt::DataDir().c_str(),dumps,index++,t->width,t->height,t->format,r.kind==nvshader::ResourceKind::Image?"img":"tex");
                if(FILE* f=std::fopen(path,"wb")){std::fwrite(texels.data(),1,texels.size(),f);std::fclose(f);RT_LOG("dumped %s (%zu bytes)",path,texels.size());}}}}}
    if(IsNetwork(*stage.shader)&&SlowNetworks()){NetworkStandIn(v);return;}
    if(IsNetwork(*stage.shader)&&LsWanted())NetworkStandIn(v,true); // (the game's own network draws the picture: frame generation's inputs only)
    if(v.compiled.global_loads||v.compiled.global_stores)GlobalAddressTable(); // (pools may have changed)
    phases.mark(1);
    Pipeline* slot_ptr=nullptr;
    static const bool by_module=std::getenv("OOT_PIPELINE_KEY_MODULE")!=nullptr; // (diagnostic: one pipeline per module, as before)
    const u64 compute_key=by_module?(u64)(uintptr_t)v.module:v.identity;
    {std::lock_guard tables{pipeline_tables_mutex};const auto it=compute_pipelines.find(compute_key);if(it!=compute_pipelines.end())slot_ptr=it->second.get();}
    if(slot_ptr&&!slot_ptr->touched){slot_ptr->touched=true;++prewarm_hits;plist::Use(slot_ptr->record);} // (made before the game)
    if(slot_ptr&&(slot_ptr->shaders.size()!=1||slot_ptr->shaders[0]!=&v))slot_ptr->shaders={&v};
    std::unique_ptr<Pipeline> slot;
    if(!slot_ptr){
        slot=std::make_unique<Pipeline>();slot->shaders={&v};
        {std::vector<StageResources> stages;stages.emplace_back(VK_SHADER_STAGE_COMPUTE_BIT,&v.compiled.resources);BuildPipelineLayout(*slot,stages);}
        if(plist::Enabled()&&v.cache_key){
            plist::Record record;record.kind=plist::Kind::Compute;record.shaders.push_back({v.source->stage,v.cache_key,v.identity});
            record.id=plist::MakeId(record.kind,record.shaders,record.state);slot->record=record.id;
            if(!plist::Add(record))plist::Use(record.id);
        }
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};ci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=v.module;ci.stage.pName="main";ci.layout=slot->layout;
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo pin;PinSubgroup(ci.stage,pin);
        // AMD (RADV's optimizer): OoT's largest compute programs (300k+ SPIR-V words) took up to 3.5 minutes to build on the
        // Steam Deck, the game frozen meanwhile. Programs past OOT_NOOPT_WORDS (40000; 0: never) are built without the
        // optimizer first (from Hookshot f414a76), and optimized on a worker, which takes over when done.
        static const u64 noopt_words=std::getenv("OOT_NOOPT_WORDS")?std::strtoull(std::getenv("OOT_NOOPT_WORDS"),nullptr,10):40000;
        // (NVIDIA's driver took as long without the optimizer: 27.8 s against 28.7 s for a cold start's 58 programs)
        bool quick=vk.properties.vendorID==0x1002&&noopt_words&&v.compiled.spirv.size()>noopt_words;
        const auto t0=std::chrono::steady_clock::now();
        VkPipeline created=VK_NULL_HANDLE;
        if(quick){ // (the optimized one from the driver's cache when it is there: unoptimized ones are only for first builds)
            VkComputePipelineCreateInfo cached=ci;cached.flags|=VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT;
            if(vkCreateComputePipelines(vk.device,PipelineCache(),1,&cached,nullptr,&created)==VK_SUCCESS&&created)quick=false;
            else created=VK_NULL_HANDLE;
        }
        if(quick)ci.flags|=VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
        if(!created)VK_CHECK(vkCreateComputePipelines(vk.device,PipelineCache(),1,&ci,nullptr,&created));
        slot->pipeline.store(created);
        if(quick){
            Pipeline* target=slot.get();const VkShaderModule module=v.module;const VkPipelineLayout layout=slot->layout;const u64 address=stage.shader->address;
            PipelineWorkers().Run([target,module,layout,address]{
                auto& vk=Vk();const auto t0=std::chrono::steady_clock::now();
                VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};ci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
                ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=module;ci.stage.pName="main";ci.layout=layout;
                VkPipelineShaderStageRequiredSubgroupSizeCreateInfo pin;PinSubgroup(ci.stage,pin);
                VkPipeline optimized=VK_NULL_HANDLE;
                if(vkCreateComputePipelines(vk.device,PipelineCache(),1,&ci,nullptr,&optimized)!=VK_SUCCESS)return;
                const VkPipeline old=target->pipeline.exchange(optimized);
                RT_LOG("compute pipeline for shader %#llx optimized on a worker in %.0f ms",(unsigned long long)address,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());
                AfterFlush([old]{vkDestroyPipeline(Vk().device,old,nullptr);}); // (after work recorded with it)
            });
        }
        {const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
         if(ms>20){u64 h=0xcbf29ce484222325ull;for(u32 w:v.compiled.spirv)h=(h^w)*0x100000001b3ull;
             if(std::getenv("OOT_COMPUTE_DEPS")){std::string d;for(const auto& x:v.compiled.dependencies){char b[64];std::snprintf(b,sizeof b," k%u b%u+%#x=%#x",(u32)x.kind,x.bank,x.offset,x.value);d+=b;}RT_LOG("  compute variant: binding base %u, %zu dependencies:%s",v.request.binding_base,v.compiled.dependencies.size(),d.c_str());}
             RT_LOG("compute pipeline for shader %#llx built in %.0f ms (SPIR-V %zu words, hash %016llx%s%s)",(unsigned long long)stage.shader->address,ms,v.compiled.spirv.size(),(unsigned long long)h,
                    v.compiled.global_loads||v.compiled.global_stores?", global memory":"",quick?", without the optimizer":"");}}
        slot_ptr=slot.get();
        {std::lock_guard tables{pipeline_tables_mutex};
         const auto placed=compute_pipelines.try_emplace(compute_key,std::move(slot));
         if(!placed.second){slot_ptr=placed.first->second.get();compute_orphans.push_back(std::move(slot));}} // (a prewarm thread made it meanwhile: ours stays alive for a worker that may hold it)
    }
    auto& pipeline=*slot_ptr;phases.mark(2);
    if(!WriteDescriptors(pipeline))return;
    phases.mark(3);
    // Storage images the shader writes now hold GPU-produced contents.
    std::vector<Texture*> written;
    for(const auto& r:v.compiled.resources)if(r.kind==nvshader::ResourceKind::Image)
        for(u32 e=0;e<r.count;++e){const u32 handle=ResourceHandle(r,v,e);if(handle)written.push_back(Binding(handle).host.get());}
    // Consecutive dispatches with nothing else recorded between them skip the full barrier when
    // this one neither reads what the earlier ones wrote nor writes what they accessed.
    struct ComputeChain{u64 serial=~0ull;std::vector<std::pair<u64,u64>> reads,writes;std::vector<Texture*> textures,images;bool global_read=false,global_write=false;};
    static ComputeChain chain;
    static const bool chain_dispatches=std::getenv("OOT_DISPATCH_BARRIERS")==nullptr;
    const auto overlaps=[](const std::vector<std::pair<u64,u64>>& a,const std::vector<std::pair<u64,u64>>& b){
        for(const auto& x:a)for(const auto& y:b)if(x.first<y.second&&y.first<x.second)return true;return false;};
    const auto contains=[](const std::vector<Texture*>& a,const std::vector<Texture*>& b){for(auto* x:a)for(auto* y:b)if(x==y)return true;return false;};
    const bool global_read=v.compiled.global_loads,global_write=v.compiled.global_stores;
    const auto& tex_reads=descriptor_reads;const auto& buf_reads=descriptor_buffer_reads;const auto& buf_writes=descriptor_buffer_writes;
    static const bool chain_unsafe=std::getenv("OOT_CHAIN_UNSAFE")!=nullptr; // measurement only: ignores hazards
    bool independent=chain_dispatches&&!indirect&&chain.serial!=~0ull;
    for(auto* t:written)independent&=t->layout==VK_IMAGE_LAYOUT_GENERAL; // no transition needed
    if(independent){
        const bool chain_writes=!chain.writes.empty()||!chain.images.empty()||chain.global_write;
        const bool writes=!buf_writes.empty()||!written.empty()||global_write;
        const bool accesses=writes||!buf_reads.empty()||!tex_reads.empty()||global_read;
        const bool chain_accesses=chain_writes||!chain.reads.empty()||!chain.textures.empty()||chain.global_read;
        independent=!(chain.global_write&&accesses)&&!(global_write&&chain_accesses)&&!(global_read&&chain_writes)&&!(chain.global_read&&writes)
            &&!overlaps(buf_reads,chain.writes)&&!overlaps(buf_writes,chain.writes)&&!overlaps(buf_writes,chain.reads)
            &&!contains(tex_reads,chain.images)&&!contains(written,chain.images)&&!contains(written,chain.textures);
        independent|=chain_unsafe;
    }
    u64 indirect_source=0;if(indirect){InitializeIndirectClamp();indirect_source=GuestDeviceAddress(indirect);}
    const auto record_dispatch=[&](VkCommandBuffer cmd){
        // (Ordered after all earlier work, or after independent work only: see above.)
        for(auto* t:written){Transition(cmd,*t,VK_IMAGE_LAYOUT_GENERAL);t->gpu_written=true;}
        u64 slot=0;
        if(indirect){
            auto& c=indirect_clamp;slot=u64(c.next++%IndirectClamp::Slots)*16;
            const u64 push[3]{indirect_source,c.address+slot,256};
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,c.pipeline);
            vkCmdPushConstants(cmd,c.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof push,push);
            vkCmdDispatch(cmd,1,1,1);
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT|VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,0,1,&barrier,0,nullptr,0,nullptr);
            const VkBufferCopy grid{slot,compute_bank0_offset+4,12};
            vkCmdCopyBuffer(cmd,c.buffer,ring.buffer,1,&grid);
            barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_UNIFORM_READ_BIT|VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
        }
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline.pipeline);
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline.layout,0,1,&pipeline.descriptor_set,0,nullptr);
        float push[32]{};push[8]=1;{const u64 pages=GuestPageTableAddress();std::memcpy((char*)push+GuestPagesPushOffset,&pages,8);}vkCmdPushConstants(cmd,pipeline.layout,VK_SHADER_STAGE_ALL,0,sizeof push,push);
        if(vk.checkpoint)vk.checkpoint(cmd,reinterpret_cast<void*>(stage.shader->address|2));
        if(indirect)vkCmdDispatchIndirect(cmd,indirect_clamp.buffer,slot);
        else vkCmdDispatch(cmd,gx,gy,gz);
    };
    static const bool chain_trace=std::getenv("OOT_CHAIN_TRACE")!=nullptr;
    if(chain_trace){static u64 n[6];
        const bool same=chain.serial==RecordSerial();
        ++n[!chain_dispatches||indirect?0:!same?1:independent?2:(global_write||chain.global_write||global_read||chain.global_read)?3:overlaps(buf_reads,chain.writes)||overlaps(buf_writes,chain.writes)||overlaps(buf_writes,chain.reads)?4:5];
        if((n[0]+n[1]+n[2]+n[3]+n[4]+n[5])%20000==0)RT_LOG("dispatch chaining: indirect %llu, other work between %llu, chained %llu, global %llu, buffer overlap %llu, image %llu",
            (unsigned long long)n[0],(unsigned long long)n[1],(unsigned long long)n[2],(unsigned long long)n[3],(unsigned long long)n[4],(unsigned long long)n[5]);}
    if(!independent||!ImmediateUnordered(record_dispatch,chain.serial)){
        work_kind=1;Immediate(record_dispatch,stage.shader->address);
        chain.reads.clear();chain.writes.clear();chain.textures.clear();chain.images.clear();chain.global_read=chain.global_write=false;
    }
    chain.reads.insert(chain.reads.end(),buf_reads.begin(),buf_reads.end());chain.writes.insert(chain.writes.end(),buf_writes.begin(),buf_writes.end());
    chain.textures.insert(chain.textures.end(),tex_reads.begin(),tex_reads.end());chain.images.insert(chain.images.end(),written.begin(),written.end());
    chain.global_read|=global_read;chain.global_write|=global_write;
    chain.serial=RecordSerial();
    const auto small_buffers=[&](const char* when){
        for(u32 i=0;i<16;++i){const auto& b=stage.storage[i];if(!b.address||b.size>0x100)continue;const auto range=LookupAddress(b.address);if(!range.pool)continue;
            const float* f=reinterpret_cast<const float*>(range.cpu_address);std::string w;for(u32 k=0;k<std::min<u64>(b.size,32)/4;++k){char t[24];std::snprintf(t,sizeof t," %g",f[k]);w+=t;}
            RT_LOG("  %s storage %u @%#llx:%s",when,i,(unsigned long long)b.address,w.c_str());}};
    if(LogThisFrame()){Flush();small_buffers("before");}
    if(LogThisFrame())for(u32 i=0;i<4;++i){const auto& u=stage.uniforms[i];if(!u.address){RT_LOG("  cs uniform %u unbound",i);continue;}
        const auto range=LookupAddress(u.address);std::string w;if(range.pool){const u32* p=static_cast<const u32*>(ResolveAddress(u.address,std::min<u64>(u.size,32)));for(u32 k=0;k<std::min<u64>(u.size,32)/4;++k){char b[16];std::snprintf(b,sizeof b," %08x",p[k]);w+=b;}}
        RT_LOG("  cs uniform %u @%#llx size %#llx:%s",i,(unsigned long long)u.address,(unsigned long long)u.size,w.c_str());}
    if(LogThisFrame()){Flush();small_buffers("after");}
    static const bool nan_trace=std::getenv("OOT_NAN_TRACE")!=nullptr;
    if(nan_trace){Flush();static std::map<u64,bool> had_nan;
        for(u32 i=0;i<16;++i){const auto& b=stage.storage[i];if(!b.address||b.size>0x400)continue;const auto range=LookupAddress(b.address);if(!range.pool)continue;
            const u32* w=reinterpret_cast<const u32*>(range.cpu_address);bool nan=false;for(u32 k=0;k<std::min<u64>(b.size,range.available)/4;++k)nan|=(w[k]&0x7f800000u)==0x7f800000u&&(w[k]&0x7fffff);
            if(nan&&!had_nan[b.address]){std::string t;for(u32 k=0;k<8;++k){char x[16];std::snprintf(x,sizeof x," %08x",w[k]);t+=x;}
                RT_LOG("NaN appeared in storage %u @%#llx after dispatch %u %u %u of shader %#llx (frame %llu):%s",i,(unsigned long long)b.address,gx,gy,gz,(unsigned long long)stage.shader->address,(unsigned long long)frame_number,t.c_str());}
            had_nan[b.address]=nan;}}
    if(LogThisFrame()){RT_LOG("dispatch %u %u %u shader %#llx push(+0xa9)=%u words; root: %08x %08x %08x %08x %08x %08x %08x %08x",gx,gy,gz,(unsigned long long)stage.shader->address,
        stage.shader->control.size()>0xa9?stage.shader->control[0xa9]:0,(*tls_state).root[1][0][0],(*tls_state).root[1][0][1],(*tls_state).root[1][0][2],(*tls_state).root[1][0][3],(*tls_state).root[1][0][4],(*tls_state).root[1][0][5],(*tls_state).root[1][0][6],(*tls_state).root[1][0][7]);
        for(const auto& r:v.compiled.resources)if(r.kind==nvshader::ResourceKind::StorageBuffer){
            const u64 a=Cbuf(&stage,r.bank,r.offset)|(u64)Cbuf(&stage,r.bank,r.offset+4)<<32;
            RT_LOG("  cs storage bank %u+%#x -> %#llx%s",r.bank,r.offset,(unsigned long long)a,r.written?" (written)":"");}
        for(auto* t:written)if(t->width*t->height<=256&&(t->format==17||t->format==46||t->format==10||t->format==41||t->format==5)){
            const auto bytes=ReadTexels(*t);std::string v;
            for(size_t i=0;i+2<=bytes.size()&&i<64;i+=(t->format==46||t->format==10)?4:2){char b[24];
                if(t->format==46||t->format==10){float f;std::memcpy(&f,&bytes[i],4);std::snprintf(b,sizeof b," %g",f);}
                else{u16 h;std::memcpy(&h,&bytes[i],2);const int e=h>>10&31,m=h&1023;float f=e?std::ldexp(1.f+m/1024.f,e-15):std::ldexp(m/1024.f,-14);if(h>>15)f=-f;std::snprintf(b,sizeof b," %g",f);}
                v+=b;}
            RT_LOG("  output %ux%u fmt %u:%s",t->width,t->height,t->format,v.c_str());
            if(t->format==17&&bytes.size()>(size_t)t->width*t->height*4){const size_t slice=(size_t)t->width*t->height*4;std::string sums;
                const auto half=[](u16 h){const int e=h>>10&31,m=h&1023;double f=e?std::ldexp(1.0+m/1024.0,e-15):std::ldexp(m/1024.0,-14);return h>>15?-f:f;};
                for(size_t z=0;z*slice<bytes.size();++z){double r=0,g=0;for(size_t i=z*slice;i+4<=(z+1)*slice;i+=4){u16 a,b2;std::memcpy(&a,&bytes[i],2);std::memcpy(&b2,&bytes[i+2],2);r+=half(a);g+=half(b2);}
                    if(r==0&&g==0)continue;char b[48];std::snprintf(b,sizeof b," z%zu:%.3g/%.3g",z,r,g);sums+=b;}
                RT_LOG("  slice sums:%s",sums.c_str());}}
        if(const char* dir=std::getenv("OOT_DRAW_DUMP")){static u32 n=0;for(auto* t:written){char name[96];std::snprintf(name,sizeof name,"/cs%03u-%ux%u-f%u.ppm",n++,t->width,t->height,t->format);DumpTexture(*t,dir+std::string(name));}}}
}
void PushRoot(u32 which,const Command& command,const u8* data,size_t size){
    const u32 table=command.x[0]&7;const u64 offset=command.x[1];
    if(LogThisFrame()){std::string w;for(size_t i=0;i+4<=std::min<size_t>(size,32);i+=4){u32 v;std::memcpy(&v,data+i,4);char b[16];std::snprintf(b,sizeof b," %08x",v);w+=b;}
        RT_LOG("push %s table %llu offset %#llx size %zu:%s",which?"compute":"graphics",(unsigned long long)command.x[0],(unsigned long long)offset,size,w.c_str());}
    if(offset>=0x100)return;
    std::memcpy(reinterpret_cast<u8*>((*tls_state).root[which][table])+offset,data,std::min<size_t>(size,0x100-offset));
}
bool ExecuteRenderCommand(const Command& command) {
    const auto* x=command.x;const auto* data=command.data.data();
    switch((Op)command.op) {
    case Op::BindTexturesDeferred: case Op::BindImagesDeferred:
    case Op::BindUniformBuffersDeferred: case Op::BindStorageBuffersDeferred: {
        Command binding=command;
        const bool buffer=(Op)command.op==Op::BindUniformBuffersDeferred||(Op)command.op==Op::BindStorageBuffersDeferred;
        binding.op=(u32)((Op)command.op==Op::BindTexturesDeferred?Op::BindTextures:
            (Op)command.op==Op::BindImagesDeferred?Op::BindImages:
            (Op)command.op==Op::BindUniformBuffersDeferred?Op::BindUniformBuffers:Op::BindStorageBuffers);
        binding.data.resize(x[2]*(buffer?16:8));
        if(!binding.data.empty())std::memcpy(binding.data.data(),reinterpret_cast<const void*>(x[3]),binding.data.size());
        return ExecuteRenderCommand(binding);
    }
    case Op::SetGeometryProcessingMode:
        // The Vulkan frontend is selected by DrawMeshTasks versus the ordinary
        // draw calls; no additional hardware mode register is needed.
        if(x[0]>1){static bool reported=false;if(!reported){reported=true;RT_LOG("NVN geometry processing mode %llu (draw type selects the pipeline)",(unsigned long long)x[0]);}}
        break;
    case Op::BindProgram: {
        const auto before=std::array<const ShaderCode*,9>{(*tls_state).stages[0].shader,(*tls_state).stages[1].shader,(*tls_state).stages[2].shader,(*tls_state).stages[3].shader,(*tls_state).stages[4].shader,(*tls_state).stages[5].shader,(*tls_state).stages[6].shader,(*tls_state).stages[7].shader,(*tls_state).stages[8].shader};
        struct Check{const std::array<const ShaderCode*,9>& b;~Check(){for(u32 i=0;i<9;++i)if((*tls_state).stages[i].shader!=b[i]){fast_draw_allowed=false;break;}}} check{before}; // (the same program bound again keeps the fast path)
        auto* program=reinterpret_cast<const Program*>(x[0]); // (the host's program, taken when the command was recorded)
        for(u32 i=0;i<9;++i) if(x[1]&(1u<<i)) (*tls_state).stages[i].shader=nullptr;
        if(program) for(const auto& code:program->shaders) {
            if(code.stage>7)Fatal("binding NVN shader stage %u",code.stage);
            const u32 stage=ShaderPublicStage[code.stage];if(x[1]&(1u<<stage))(*tls_state).stages[stage].shader=&code;}
        break;
    }
    case Op::BindUniformBuffer: if(x[0]>=9||x[1]>=18) Fatal("invalid uniform binding"); (*tls_state).stages[x[0]].uniforms[x[1]]={x[2],x[3]};break;
    case Op::BindStorageBuffer: if(x[0]>=9||x[1]>=16) Fatal("invalid storage binding"); if((*tls_state).stages[x[0]].storage[x[1]].address!=x[2]||(*tls_state).stages[x[0]].storage[x[1]].size!=x[3]){(*tls_state).stages[x[0]].storage[x[1]]={x[2],x[3]};fast_draw_allowed=false;}break;
    case Op::BindVertexBuffer: (*tls_state).vertices.at(x[0])={x[1],x[2]};break;
    case Op::BindVertexBuffers: for(u32 i=0;i<x[1];++i)std::memcpy(&(*tls_state).vertices.at(x[0]+i),data+i*16,16);break;
    case Op::BindVertexAttribState: {decltype((*tls_state).attributes) next{};std::memcpy(next.data(),data,std::min(command.data.size(),sizeof next));
        if(next!=(*tls_state).attributes){(*tls_state).attributes=next;PipelineStateChanged();}break;} // (a different vertex format changes the pipeline)
    case Op::BindVertexStreamState: {decltype((*tls_state).streams) next{};std::memcpy(next.data(),data,std::min(command.data.size(),sizeof next));
        if(next!=(*tls_state).streams){(*tls_state).streams=next;PipelineStateChanged();}break;}
    case Op::BindTexture: if((*tls_state).stages.at(x[0]).textures.at(x[1])!=x[2]){(*tls_state).stages[x[0]].textures[x[1]]=x[2];if(!FastTextures())fast_draw_allowed=false;}break;
    case Op::BindTextures: for(u32 i=0;i<x[2];++i)std::memcpy(&(*tls_state).stages.at(x[0]).textures.at(x[1]+i),data+i*8,8);break;
    case Op::BindImage: (*tls_state).stages.at(x[0]).images.at(x[1])=x[2];break;
    case Op::BindImages: for(u32 i=0;i<x[2];++i)std::memcpy(&(*tls_state).stages.at(x[0]).images.at(x[1]+i),data+i*8,8);break;
    case Op::SetTexturePool: (*tls_state).textures=HostTexturePool(x[0]);break;
    case Op::SetSamplerPool: (*tls_state).samplers=HostSamplerPool(x[0]);break;
    case Op::SetRenderTargets: if(std::memcmp(&(*tls_state).targets,data,sizeof (*tls_state).targets)||(*tls_state).color_count!=x[0]){std::memcpy(&(*tls_state).targets,data,sizeof (*tls_state).targets);(*tls_state).color_count=x[0];fast_draw_allowed=false;}break;
    case Op::SetViewport: (*tls_state).viewport.x=(s32)x[0];(*tls_state).viewport.y=(s32)x[1];(*tls_state).viewport.width=(s32)x[2];(*tls_state).viewport.height=(s32)x[3];break;
    case Op::SetViewports: if(x[0]==0 && x[1])std::memcpy(&(*tls_state).viewport,data,16);break;
    case Op::SetDepthRange: (*tls_state).viewport.minDepth=command.f[0];(*tls_state).viewport.maxDepth=command.f[1];break;
    case Op::SetDepthRanges: if(x[0]==0&&x[1])std::memcpy(&(*tls_state).viewport.minDepth,data,8);break;
    case Op::SetScissor: (*tls_state).scissor={{(s32)x[0],(s32)x[1]},{(u32)x[2],(u32)x[3]}};break;
    case Op::SetScissors: if(x[0]==0&&x[1])std::memcpy(&(*tls_state).scissor,data,16);break;
    case Op::BindBlendState: {u64 blend;std::memcpy(&blend,data,8);if((*tls_state).blends[blend&7]!=blend){(*tls_state).blends[blend&7]=blend;PipelineStateChanged();}break;}
    case Op::BindColorState: {u32 v;std::memcpy(&v,data,4);if(v!=(*tls_state).color){(*tls_state).color=v;PipelineStateChanged();}break;}
    case Op::BindChannelMaskState: {u32 v;std::memcpy(&v,data,4);if(v!=(*tls_state).channel){(*tls_state).channel=v;fast_draw_allowed=false;}break;}
    case Op::BindDepthStencilState: {u64 value;std::memcpy(&value,data,8);if(value!=(*tls_state).depth){(*tls_state).depth=value;PipelineStateChanged();}break;}
    case Op::BindPolygonState: {u32 v;std::memcpy(&v,data,4);if(v!=(*tls_state).polygon){(*tls_state).polygon=v;PipelineStateChanged();}break;}
    case Op::BindMultisampleState: break;
    case Op::SetBlendColor: std::memcpy((*tls_state).blend_color,data,16);break;
    case Op::SetDepthBias: std::memcpy((*tls_state).depth_bias,command.f,12);break;
    case Op::SetLineWidth: if((*tls_state).line_width!=command.f[0]){(*tls_state).line_width=command.f[0];fast_draw_allowed=false;}break;
    case Op::SetStencilMask: for(u32 i=0;i<2;++i)if(x[0]&(1u<<i))(*tls_state).stencil_write[i]=x[1];break;
    case Op::SetStencilValueMask: for(u32 i=0;i<2;++i)if(x[0]&(1u<<i))(*tls_state).stencil_compare[i]=x[1];break;
    case Op::SetStencilRef: for(u32 i=0;i<2;++i)if(x[0]&(1u<<i))(*tls_state).stencil_ref[i]=x[1];break;
    case Op::SetPrimitiveRestart: if((*tls_state).restart!=(bool)x[0]){(*tls_state).restart=x[0];fast_draw_allowed=false;}break;
    case Op::SetRenderEnable: (*tls_state).render_enable=x[0];(*tls_state).render_condition=0;(*tls_state).render_decided=0;break;
    case Op::SetRenderEnableConditional:
        if(x[0]>1)Fatal("invalid conditional render mode %llu",(unsigned long long)x[0]);
        (*tls_state).render_enable=true;(*tls_state).render_condition=x[1];(*tls_state).render_if_equal=x[0]==0;(*tls_state).render_decided=(u8)x[7];break;
    case Op::SetConservativeRasterEnable: if(x[0])Fatal("conservative rasterization pending");break;
    case Op::SetTiledCacheAction: case Op::SetTiledCacheTileSize: break; // console tile-cache optimization
    case Op::ClearColor:case Op::ClearColori:case Op::ClearColorui: {
        if(!DrawEnabled()){++frame_clears_skipped;break;}
        if(ProbeFrameSkipped())break;
        const u32 predicate=draw_predicate;const bool inverted=draw_inverted;draw_predicate=~0u;
        ++stats.clears;
        const u64 object=(*tls_state).targets.colors[x[0]];if(!object||!HostTexture(object))break;auto& t=*HostTexture(object);
        if(LogThisFrame()){float f[4];std::memcpy(f,data,16);RT_LOG("clear color %llu target %#llx %ux%u fmt %u to (%g %g %g %g)",(unsigned long long)x[0],(unsigned long long)object,t.width,t.height,t.format,f[0],f[1],f[2],f[3]);}
        if(x[2]!=15)Fatal("partial color clear requires attachment clear");
        if(predicate!=~0u){PredicatedClear(predicate,inverted,&t,(*tls_state).targets.color_views[x[0]],VK_IMAGE_ASPECT_COLOR_BIT,data,0,0);t.gpu_written=true;break;}
        Immediate([&](VkCommandBuffer cmd){Transition(cmd,t,VK_IMAGE_LAYOUT_GENERAL);VkClearColorValue color;std::memcpy(&color,data,16);
            const auto& view=(*tls_state).targets.color_views[x[0]];VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT,view.levels_base,1,view.layers_base,view.layers_count?view.layers_count:1};
            vkCmdClearColorImage(cmd,t.image,t.layout,&color,1,&range);});t.gpu_written=true;break;
    }
    case Op::ClearDepthStencil: {
        if(!DrawEnabled()){++frame_clears_skipped;break;}
        if(ProbeFrameSkipped())break;
        const u32 predicate=draw_predicate;const bool inverted=draw_inverted;draw_predicate=~0u;
        if(!(*tls_state).targets.depth||!HostTexture((*tls_state).targets.depth))break;auto& t=*HostTexture((*tls_state).targets.depth);
        const auto& view=(*tls_state).targets.depth_view;
        VkImageAspectFlags aspect=(x[0]?VK_IMAGE_ASPECT_DEPTH_BIT:0)|(x[2]?VK_IMAGE_ASPECT_STENCIL_BIT:0);aspect &= AspectOf(t);if(!aspect)break;
        if(predicate!=~0u){++frame_clears_predicated;PredicatedClear(predicate,inverted,&t,view,aspect,nullptr,command.f[0],(u32)x[1]);t.gpu_written=true;break;}
        Immediate([&](VkCommandBuffer cmd){Transition(cmd,t,VK_IMAGE_LAYOUT_GENERAL);VkClearDepthStencilValue value{command.f[0],(u32)x[1]};
            VkImageSubresourceRange range{aspect,view.levels_base,1,view.layers_base,view.layers_count?view.layers_count:1};vkCmdClearDepthStencilImage(cmd,t.image,t.layout,&value,1,&range);});t.gpu_written=true;break;
    }
    case Op::SetDepthClamp: if((*tls_state).depth_clamp!=(bool)x[0]){(*tls_state).depth_clamp=x[0];fast_draw_allowed=false;}break;
    case Op::SetPointSize: if((*tls_state).point_size!=command.f[0]){(*tls_state).point_size=command.f[0];fast_draw_allowed=false;}break;
    case Op::SetRasterizerDiscard: {const bool value=x[0]!=0;if((*tls_state).discard!=value){(*tls_state).discard=value;fast_draw_allowed=false;}break;}
    case Op::SetSampleMask: {const u32 value=(u32)x[0];if((*tls_state).sample_mask!=value){(*tls_state).sample_mask=value;fast_draw_allowed=false;}break;}
    case Op::SetDepthBounds: {
        const bool enabled=x[0]!=0;
        const bool changed=(*tls_state).depth_bounds!=enabled||(enabled&&std::memcmp((*tls_state).depth_bounds_range,command.f,8)!=0);
        (*tls_state).depth_bounds=enabled;(*tls_state).depth_bounds_range[0]=command.f[0];(*tls_state).depth_bounds_range[1]=command.f[1];
        if(changed)fast_draw_allowed=false;
        break;
    }
    case Op::SetAlphaRef: break; // alpha test is done by the game's shaders on this platform
    case Op::BindUniformBuffers: for(u32 i=0;i<x[2];++i){if(x[0]>=9||x[1]+i>=18)Fatal("invalid uniform binding");std::memcpy(&(*tls_state).stages[x[0]].uniforms[x[1]+i],data+i*16,16);}break;
    case Op::BindStorageBuffers: for(u32 i=0;i<x[2];++i){if(x[0]>=9||x[1]+i>=16)Fatal("invalid storage binding");std::memcpy(&(*tls_state).stages[x[0]].storage[x[1]+i],data+i*16,16);}break;
    case Op::UpdateUniformBuffer: {
        // (buffer, size, offset, data size): written in command order, before later draws read it.
        // Ordered with the batched GPU work around it.
        const auto range=GpuLookupAddress(x[0]+x[2]);const u64 size=command.data.size();
        if(!range.pool||size>range.available)Fatal("uniform update outside mapped memory");
        if(size%4||range.offset%4||size>65536){Flush();std::memcpy(reinterpret_cast<void*>(range.cpu_address),data,size);break;}
        std::vector<u8> bytes(data,data+size);
        if(UniformOverlayEnabled()&&!HasPendingWrite(x[0]+x[2],size)){
            unwritten_uniform_updates.push_back({range.buffer,range.offset,range.cpu_address,bytes});
            for(u64 page=range.cpu_address>>12;page<=(range.cpu_address+size-1)>>12;++page)++unwritten_uniform_pages[page];
            unwritten_lo=std::min(unwritten_lo,range.cpu_address);unwritten_hi=std::max(unwritten_hi,range.cpu_address+size);
            ++uniform_updates_deferred;uniform_overlay_cache.clear();
            static bool hooked=false;if(!hooked){hooked=true;SetPreSubmitHook(PreSubmit);}
        }else{
            WriteUnwrittenUniformUpdates(); // (order)
            Immediate([&](VkCommandBuffer cmd){vkCmdUpdateBuffer(cmd,range.buffer,range.offset,size,bytes.data());});
        }
        NoteWriteBack(x[0]+x[2],size); // (CPU readers see pending_uniform_updates)
        // Shader specialization and handle evaluation run on the CPU before the
        // batch executes. They must see the same ordered uniform updates as the GPU.
        const u64 batch=SubmittedBatches(); // the open batch's number
        pending_uniform_updates.push_back({range.cpu_address,std::move(bytes),batch});
        IndexUniformUpdate(pending_uniform_updates.size()-1);
        if(uniform_retirement_batch!=batch){uniform_retirement_batch=batch;AfterFlush([batch]{RetireUniformUpdates(batch);});}
        break;
    }
    case Op::ClearBuffer: {
        // ClearBuffer(address, size, 32-bit value)
        const auto range=GpuLookupAddress(x[0]);if(!range.pool||x[1]>range.available)Fatal("buffer clear outside mapped memory");
        NotePendingWrite(x[0],x[1]);
        const u64 size=x[1]&~3ULL;const u32 value=(u32)x[2];
        if(size&&range.offset%4==0)Immediate([&](VkCommandBuffer cmd){vkCmdFillBuffer(cmd,range.buffer,range.offset,size,value);});
        else{Flush();u32* dst=reinterpret_cast<u32*>(range.cpu_address);for(u64 i=0;i<size/4;++i)dst[i]=value;}
        break;
    }
    case Op::ClearTexture:case Op::ClearTexturei:case Op::ClearTextureui: {
        auto& t=*HostTexture(x[0]);if(t.skip){WantFullImage(t);EnsureImage(t);}TextureCopy region;std::memcpy(&region,data,sizeof region);
        VkClearColorValue color;std::memcpy(&color,data+sizeof region,16);
        Immediate([&](VkCommandBuffer cmd){Transition(cmd,t,VK_IMAGE_LAYOUT_GENERAL);
            if(t.info.depth||t.info.stencil){VkClearDepthStencilValue v{color.float32[0],color.uint32[1]};VkImageSubresourceRange r{AspectOf(t),region.view.levels_base,1,region.view.layers_base,region.view.layers_count?region.view.layers_count:1};vkCmdClearDepthStencilImage(cmd,t.image,t.layout,&v,1,&r);}
            else{VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT,region.view.levels_base,1,region.view.layers_base,region.view.layers_count?region.view.layers_count:1};vkCmdClearColorImage(cmd,t.image,t.layout,&color,1,&r);}});
        t.gpu_written=true;break;
    }
    case Op::DispatchCompute: if(x[0]&&x[1]&&x[2])Dispatch((u32)x[0],(u32)x[1],(u32)x[2]);break;
    case Op::DispatchComputeIndirect: {
        // Arguments produced by GPU work still in the open batch are consumed on the GPU.
        static const bool sync_indirect=std::getenv("OOT_SYNC_INDIRECT")!=nullptr; // (GPU-read counts are validated: indirect_clamp.comp)
        if(HasPendingWrite(x[0],12)){if(!sync_indirect){const auto args=LookupAddress(x[0]);if(args.pool&&args.offset%4==0&&args.available>=12){Dispatch(1,1,1,x[0]);{static const char* f=std::getenv("OOT_INDIRECT_FLUSH");if(f){if(*f=='s')Submit();else Flush();}}
            static const bool trace=std::getenv("OOT_INDIRECT_TRACE")!=nullptr;
            if(trace){const u64 a=x[0];const u64 shader=(*tls_state).stages[5].shader?(*tls_state).stages[5].shader->address:0;AfterFlush([a,shader]{const u32* g=static_cast<const u32*>(ResolveAddress(a,12));
                static std::map<std::tuple<u64,u32,u32,u32>,u32> seen;if(seen[{shader,g[0],g[1],g[2]}]++==0)RT_LOG("indirect dispatch shader %#llx: %u %u %u",(unsigned long long)shader,g[0],g[1],g[2]);});}
            break;}}Flush();}
        const u32* g=static_cast<const u32*>(ResolveAddress(x[0],12));if(g[0]&&g[1]&&g[2])Dispatch(g[0],g[1],g[2]);break;}
    case Op::PushGraphicsConstants: PushRoot(0,command,command.data.data(),command.data.size());break;
    case Op::PushComputeConstants: PushRoot(1,command,command.data.data(),command.data.size());break;
    case Op::PushGraphicsConstantsDeferred: case Op::PushComputeConstantsDeferred: {
        const size_t size=std::min<u64>(x[2],0x100);
        PushRoot(command.op==(u32)Op::PushComputeConstantsDeferred,command,static_cast<const u8*>(ResolveAddress(x[3],size)),size);break;}
    case Op::DrawMeshTasks:
        (*tls_state).primitive=4;if(DrawEnabled()){Draw(command);++stats.draws;}else ++stats.skipped;break;
    case Op::DrawMeshTasksIndirect: {
        if(!DrawEnabled()){++stats.skipped;break;}
        // (the Foliage Shadows row: GPU-culled foliage - grass - is left out of the shadow maps, the depth-only targets of
        // format D16. Each cascade drew it again.)
        if(!gfx::Current().foliage_shadows&&!(*tls_state).color_count&&(*tls_state).targets.depth){const Texture* shadow=HostTexture((*tls_state).targets.depth);
            if(shadow&&shadow->format==50){++stats.skipped;break;}}
        // NVN's two-word command {task count, first task} is usually written by GPU culling work
        // earlier in the same batch (foliage): it is converted on the GPU into Vulkan's
        // {x, y, z} group counts (indirect_clamp.comp, mesh mode) and drawn from there. Reading it
        // on the CPU at recording time saw zero tasks and dropped every grass draw.
        static const bool cpu_args=std::getenv("OOT_MESH_INDIRECT_CPU")!=nullptr;
        if(cpu_args){
            if(HasPendingWrite(x[0],8))Flush();
            u32 args[2];std::memcpy(args,ResolveAddress(x[0],sizeof args),sizeof args);
            Command direct=command;direct.op=(u32)Op::DrawMeshTasks;direct.x[0]=args[1];direct.x[1]=args[0];
            (*tls_state).primitive=4;if(args[0]){Draw(direct);++stats.draws;}break;
        }
        InitializeIndirectClamp();
        const u64 source=GuestDeviceAddress(x[0]);
        if(LogThisFrame()){const u64 address=x[0];const bool pending=HasPendingWrite(address,8);
            AfterFlush([address,pending]{u32 a[2];std::memcpy(a,ResolveAddress(address,8),8);
                RT_LOG("mesh indirect args @%#llx (GPU-pending %d) after its batch: count %u first %u",(unsigned long long)address,(int)pending,a[0],a[1]);});}
        auto& c=indirect_clamp;
        const auto converted=mesh_preclamped.find(x[0]); // (done for the whole group of draws, ExecuteSegmentsParallel)
        const u64 slot=converted!=mesh_preclamped.end()?converted->second:u64(c.next++%IndirectClamp::Slots)*16;
        if(converted==mesh_preclamped.end())MeshHoistOrLight([&](VkCommandBuffer cmd){
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
            const u64 push[3]{source,(c.address+slot)|1,FoliageKeep()};
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,c.pipeline);
            vkCmdPushConstants(cmd,c.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof push,push);
            vkCmdDispatch(cmd,1,1,1);
            barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,0,1,&barrier,0,nullptr,0,nullptr);
        });
        Command direct=command;direct.op=(u32)Op::DrawMeshTasks;direct.x[0]=0;direct.x[1]=0;
        mesh_indirect_slot=slot+1;mesh_indirect_source=source;
        (*tls_state).primitive=4;Draw(direct);++stats.draws;
        mesh_indirect_slot=0;
        break;
    }
    case Op::DrawElementsBaseVertex:
    case Op::DrawArrays:case Op::DrawElements:case Op::DrawArraysInstanced:case Op::DrawElementsInstanced:
        (*tls_state).primitive=x[0];if(DrawEnabled()){Draw(command);++stats.draws;}else ++stats.skipped;break;
    default:return false;
    }
    return true;
}
}

namespace nvn {
// OOT_DRAW_STATS: rendering counters since the last report.
namespace {
void InitializeGpuCounters(){
    auto& vk=Vk();
    if(!samples.pool){VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};info.queryType=VK_QUERY_TYPE_OCCLUSION;info.queryCount=SampleCounter::Capacity;
        VK_CHECK(vkCreateQueryPool(vk.device,&info,nullptr,&samples.pool));}
    static const bool disabled=std::getenv("OOT_CPU_PREDICATES")!=nullptr;
    if(gpu_counters.ready||!vk.conditional_rendering||disabled)return;
    const auto make=[&](VkDeviceSize size,VkBufferUsageFlags usage,bool host=false){
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=size;bi.usage=usage|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VkBuffer buffer;VK_CHECK(TracedCreateBuffer(&bi,&buffer,__FILE__,__LINE__));VkMemoryRequirements req;vkGetBufferMemoryRequirements(vk.device,buffer,&req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=req.size;ai.memoryTypeIndex=FindMemoryType(req.memoryTypeBits,host?VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT:VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory memory;VK_CHECK(vkAllocateMemory(vk.device,&ai,nullptr,&memory));VK_CHECK(vkBindBufferMemory(vk.device,buffer,memory,0));
        if(host)VK_CHECK(vkMapMemory(vk.device,memory,0,size,0,reinterpret_cast<void**>(&gpu_counters.predicate_values)));return buffer;};
    gpu_counters.results=make(SampleCounter::Capacity*8ull,0);
    gpu_counters.total=make(16,0);
    gpu_counters.predicates=make(GpuCounters::PredicateCapacity*4ull,VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT,std::getenv("OOT_COUNTER_CHECK")!=nullptr);
    {   // report entries: written by the CPU, read by the processing dispatch
        auto* saved=gpu_counters.predicate_values;
        gpu_counters.entries=make(GpuCounters::EntryCapacity*sizeof(GpuCounters::Entry),0,true);
        gpu_counters.entry_map=reinterpret_cast<GpuCounters::Entry*>(gpu_counters.predicate_values);
        gpu_counters.predicate_values=saved;
    }
    VkDescriptorSetLayoutBinding bindings[5];
    for(u32 i=0;i<5;++i)bindings[i]={i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    VkDescriptorSetLayoutCreateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};di.bindingCount=5;di.pBindings=bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(vk.device,&di,nullptr,&gpu_counters.set_layout));
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT,0,16};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};li.setLayoutCount=1;li.pSetLayouts=&gpu_counters.set_layout;li.pushConstantRangeCount=1;li.pPushConstantRanges=&push;
    VK_CHECK(vkCreatePipelineLayout(vk.device,&li,nullptr,&gpu_counters.layout));
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};mi.codeSize=sizeof occlusion_spv;mi.pCode=occlusion_spv;
    VkShaderModule module;VK_CHECK(vkCreateShaderModule(vk.device,&mi,nullptr,&module));
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};ci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=module;ci.stage.pName="main";ci.layout=gpu_counters.layout;
    VK_CHECK(vkCreateComputePipelines(vk.device,PipelineCache(),1,&ci,nullptr,&gpu_counters.pipeline));
    vkDestroyShaderModule(vk.device,module,nullptr);
    gpu_counters.begin=reinterpret_cast<PFN_vkCmdBeginConditionalRenderingEXT>(vkGetDeviceProcAddr(vk.device,"vkCmdBeginConditionalRenderingEXT"));
    gpu_counters.end=reinterpret_cast<PFN_vkCmdEndConditionalRenderingEXT>(vkGetDeviceProcAddr(vk.device,"vkCmdEndConditionalRenderingEXT"));
    Immediate([&](VkCommandBuffer cmd){vkCmdFillBuffer(cmd,gpu_counters.total,0,16,0);});
    // Queries are used once per batch: reports are processed before submission, and the ring is
    // reset on the host after the batch completes.
    Flush();
    vkResetQueryPool(vk.device,samples.pool,0,SampleCounter::Capacity);samples.next=0;
    SetPreSubmitHook(PreSubmit);
    gpu_counters.ready=gpu_counters.begin&&gpu_counters.end;
}
// mode 0: add query results [first, first+count) to the GPU total (copying them first) and write it
// to the guest report at `guest` (0: none); 1: reset the total; 2: predicate slot `first` from the
// operands at `guest`, `count` = render-if-equal.
void CounterDispatch(u32 mode,u32 first,u32 count,u64 guest,u32 copy_queries){
    auto& vk=Vk();
    const VkDescriptorSet set=AllocateBatchSet(gpu_counters.set_layout);
    VkDescriptorBufferInfo infos[5]={{gpu_counters.results,0,VK_WHOLE_SIZE},{gpu_counters.total,0,VK_WHOLE_SIZE},{gpu_counters.results,0,VK_WHOLE_SIZE},{gpu_counters.predicates,0,VK_WHOLE_SIZE},
        {gpu_counters.entries,gpu_counters.entries_used*sizeof(GpuCounters::Entry),VK_WHOLE_SIZE}};
    u32 word=~0u;
    if(guest){const auto range=GpuLookupAddress(guest);if(!range.pool||range.available<24)Fatal("counter memory %#llx unmapped",(unsigned long long)guest);
        const u64 alignment=vk.properties.limits.minStorageBufferOffsetAlignment,aligned=range.offset&~(alignment-1);
        word=(u32)((range.offset-aligned)/4);infos[2]={range.buffer,aligned,(range.offset-aligned)+24};}
    VkWriteDescriptorSet writes[5];
    for(u32 i=0;i<5;++i){writes[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};writes[i].dstSet=set;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&infos[i];}
    vkUpdateDescriptorSets(vk.device,5,writes,0,nullptr);
    const u32 push[4]={mode,first,count,word};
    // A light item: the query copy waits for its queries itself (VK_QUERY_RESULT_WAIT_BIT), and the
    // trailing barrier makes the results visible to later work while only waiting for this dispatch.
    ImmediateLight([&](VkCommandBuffer cmd){
        if(copy_queries){
            vkCmdCopyQueryPoolResults(cmd,samples.pool,first,copy_queries,gpu_counters.results,first*8ull,8,VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT);
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
        }
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,gpu_counters.pipeline);
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,gpu_counters.layout,0,1,&set,0,nullptr);
        vkCmdPushConstants(cmd,gpu_counters.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof push,push);
        vkCmdDispatch(cmd,1,1,1);
        VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};after.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;after.dstAccessMask=VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,1,&after,0,nullptr,0,nullptr);
    });
}
// Device address of guest memory (for the processing dispatch's report writes).
u64 GuestDeviceAddress(u64 address){
    // (not cached by VkBuffer handle: imports come and go and a new buffer often gets a destroyed one's handle, so
    // a cached address pointed into whatever the driver placed there next - the report dispatch then wrote into it)
    const auto range=GpuLookupAddress(address);if(!range.pool)Fatal("report memory %#llx unmapped",(unsigned long long)address);
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};info.buffer=range.buffer;
    return vkGetBufferDeviceAddress(Vk().device,&info)+range.offset;
}
// Processes the pending reports: one copy of the new query results and one dispatch.
void ProcessReports(){
    auto& g=gpu_counters;
    if(g.pending.empty())return;
    if(g.entries_used+g.pending.size()+1>GpuCounters::EntryCapacity)Fatal("counter report ring overflow");
    static_assert(sizeof(GpuCounters::Entry)==24,"matches occlusion.comp");
    u32 end=g.processed;for(const auto& e:g.pending)if(e.kind!=1)end=std::max(end,e.end);
    std::memcpy(g.entry_map+g.entries_used,g.pending.data(),g.pending.size()*sizeof(GpuCounters::Entry));
    const u32 first=g.processed,count=(u32)g.pending.size();
    ++predicate_stats.processing;predicate_stats.processed_reports+=count;
    CounterDispatch(3,first,count,0,end-first);
    g.entries_used=(g.entries_used+count+1)&~1u; // 2 x 24-byte entries: keeps descriptor offsets 16-byte aligned
    g.processed=end;g.pending.clear();
}
// A clear under a GPU-evaluated predicate: an attachment clear inside a render pass, which
// conditional rendering covers (image clears are not predicated).
void PredicatedClear(u32 predicate,bool inverted,Texture* t,const TextureView& view,VkImageAspectFlags aspect,const u8* color,float depth_value,u32 stencil_value){
    const u32 level=view.levels_base,layers=view.layers_count?view.layers_count:1;
    const VkExtent2D extent{std::max(1u,t->width>>level),std::max(1u,t->height>>level)};
    const VkImageView image_view=AttachmentView(*t,view);
    Immediate([&](VkCommandBuffer cmd){
        Transition(cmd,*t,VK_IMAGE_LAYOUT_GENERAL);
        VkRenderingAttachmentInfo a{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};a.imageView=image_view;a.imageLayout=VK_IMAGE_LAYOUT_GENERAL;a.loadOp=VK_ATTACHMENT_LOAD_OP_LOAD;a.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};ri.renderArea.extent=extent;ri.layerCount=layers;
        VkClearAttachment clear{};clear.aspectMask=aspect;
        if(aspect&VK_IMAGE_ASPECT_COLOR_BIT){ri.colorAttachmentCount=1;ri.pColorAttachments=&a;std::memcpy(&clear.clearValue.color,color,16);}
        else{if(t->info.depth)ri.pDepthAttachment=&a;if(t->info.stencil)ri.pStencilAttachment=&a;clear.clearValue.depthStencil={depth_value,stencil_value};}
        vkCmdBeginRendering(cmd,&ri);
        VkConditionalRenderingBeginInfoEXT ci{VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT};ci.buffer=gpu_counters.predicates;ci.offset=predicate*4ull;
        if(inverted)ci.flags=VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT;gpu_counters.begin(cmd,&ci);
        const VkClearRect rect{{{0,0},extent},0,layers};
        vkCmdClearAttachments(cmd,1,&clear,1,&rect);
        gpu_counters.end(cmd);
        vkCmdEndRendering(cmd);
    });
}
} // namespace
// Query, predicate and report-entry slots are used once each and shared by the in-flight batches;
// when a ring runs low everything is flushed (the GPU idles) and all of them restart at zero.
// OoT uses ~3,000 to 8,000 query slots a frame, so the 65,536 ran out every 9 to 20 frames, and starting over waited
// for the GPU each time (a slow frame several times a second in Kokiri Forest). There are two sets of rings now: when
// one runs low the batch is submitted (not waited for) and the other takes over, which the GPU finished with a whole
// cycle ago. OOT_COUNTER_RING_FLUSH=1: one set and the wait, as before.
u64 counter_ring_flushes=0; // (statistics: switches, or waits with OOT_COUNTER_RING_FLUSH)
namespace {
struct CounterSet{VkQueryPool pool=VK_NULL_HANDLE;VkBuffer results=VK_NULL_HANDLE,predicates=VK_NULL_HANDLE,entries=VK_NULL_HANDLE;
    GpuCounters::Entry* entry_map=nullptr;u32* predicate_values=nullptr;u64 last_batch=0;bool made=false;} counter_sets[2];
u32 counter_set=0;
void MakeCounterSet(CounterSet& set){
    auto& vk=Vk();
    VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};info.queryType=VK_QUERY_TYPE_OCCLUSION;info.queryCount=SampleCounter::Capacity;
    VK_CHECK(vkCreateQueryPool(vk.device,&info,nullptr,&set.pool));
    const auto make=[&](VkDeviceSize size,VkBufferUsageFlags usage,void** mapped){
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=size;bi.usage=usage|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VkBuffer buffer;VK_CHECK(TracedCreateBuffer(&bi,&buffer,__FILE__,__LINE__));VkMemoryRequirements req;vkGetBufferMemoryRequirements(vk.device,buffer,&req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=req.size;ai.memoryTypeIndex=FindMemoryType(req.memoryTypeBits,mapped?VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT:VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory memory;VK_CHECK(vkAllocateMemory(vk.device,&ai,nullptr,&memory));VK_CHECK(vkBindBufferMemory(vk.device,buffer,memory,0));
        if(mapped)VK_CHECK(vkMapMemory(vk.device,memory,0,size,0,mapped));return buffer;};
    set.results=make(SampleCounter::Capacity*8ull,0,nullptr);
    set.predicates=make(GpuCounters::PredicateCapacity*4ull,VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT,std::getenv("OOT_COUNTER_CHECK")?reinterpret_cast<void**>(&set.predicate_values):nullptr);
    set.entries=make(GpuCounters::EntryCapacity*sizeof(GpuCounters::Entry),0,reinterpret_cast<void**>(&set.entry_map));
    vkResetQueryPool(vk.device,set.pool,0,SampleCounter::Capacity);
    set.made=true;
}
}
void EnsureCounterRings(){
    auto& g=gpu_counters;
    if(samples.next+256<SampleCounter::Capacity&&g.next_predicate+4096<GpuCounters::PredicateCapacity&&g.entries_used+4096<GpuCounters::EntryCapacity)return;
    ++counter_ring_flushes;
    static const bool wait=std::getenv("OOT_COUNTER_RING_FLUSH")!=nullptr||std::getenv("OOT_COUNTER_CHECK")!=nullptr;
    if(wait){
        Flush();
        vkResetQueryPool(Vk().device,samples.pool,0,SampleCounter::Capacity);
    }else{
        Submit(); // (the batch that used these rings last: ends its pass and processes its reports)
        auto& old=counter_sets[counter_set];
        if(!old.made){old.pool=samples.pool;old.results=g.results;old.predicates=g.predicates;old.entries=g.entries;old.entry_map=g.entry_map;old.predicate_values=g.predicate_values;old.made=true;}
        old.last_batch=SubmittedBatches();
        counter_set^=1;auto& next=counter_sets[counter_set];
        if(!next.made)MakeCounterSet(next);
        else{WaitForBatches(next.last_batch); // (long done: a whole cycle of the other set lies between)
            vkResetQueryPool(Vk().device,next.pool,0,SampleCounter::Capacity);}
        samples.pool=next.pool;g.results=next.results;g.predicates=next.predicates;g.entries=next.entries;g.entry_map=next.entry_map;g.predicate_values=next.predicate_values;
    }
    samples.next=0;g.processed=0;g.entries_used=0;g.next_predicate=0;g.segment_query=~0u;g.reports.clear();
}
// Ends the open pass's occlusion segment and starts the next inside the same pass, so every query
// before the returned boundary belongs to work recorded so far.
u32 SplitSegment(){
    auto& g=gpu_counters;
    EnsureCounterRings();
    if(g.segment_query==~0u)return samples.next;
    const u32 ended=g.segment_query,next=samples.next++;
    // (queued behind the draws the recorder thread has not recorded yet: an occlusion report follows almost every object)
    const bool split=RecordInPassAsync([](VkCommandBuffer cmd,void* packed,u64){
        const u64 v=reinterpret_cast<u64>(packed);
        vkCmdEndQuery(cmd,samples.pool,(u32)(v>>32));
        vkCmdBeginQuery(cmd,samples.pool,(u32)v,VK_QUERY_CONTROL_PRECISE_BIT);},reinterpret_cast<void*>((u64)ended<<32|next));
    if(!split){g.segment_query=~0u;return samples.next;} // (the pass already ended)
    g.segment_query=next;
    SetPassEndHook([next](VkCommandBuffer cmd){vkCmdEndQuery(cmd,samples.pool,next);gpu_counters.segment_query=~0u;});
    return next;
}
void ResetRenderCounter(u32 counter) {
    if(counter!=1)return;
    if(!OcclusionQueries())return; // (no queries: every report reads visible)
    run_note={};samples_run_open=true;samples_run_batch=SubmittedBatches();
    samples.enabled=true;
    InitializeGpuCounters();
    static const bool counter_check=std::getenv("OOT_COUNTER_CHECK")!=nullptr;
    if(!gpu_counters.ready||counter_check)AfterFlush([]{samples.value=0;}); // (the CPU total)
    if(!gpu_counters.ready)return;
    gpu_counters.pending.push_back({0,SplitSegment(),~0u,2,0});
    gpu_counters.last_report=0;
}
// Report timestamps come from the GPU: a timestamp query where the report sits in the command
// stream, copied into the report (+8) at the end of the batch and written back with the other GPU
// results, so the guest sees it once the GPU got there (not when the host noticed the batch
// completed). The game times its GPU work with them (dynamic resolution): late or host-side times
// made it alternate between its lowest and highest resolution. The guest's timestamp clock
// (nvnDeviceGetCurrentTimestampInNanoseconds) is the GPU clock, calibrated against the steady clock.
struct ReportTimes{VkQueryPool pool=VK_NULL_HANDLE;u32 next=0;static constexpr u32 Capacity=16384;
    PFN_vkGetCalibratedTimestampsKHR calibrate=nullptr;s64 offset=0;u64 calibrated_at=0;bool enabled=false,initialized=false;
    struct Write{u32 query;u64 address;bool has_value;u32 value;};std::vector<Write> writes;} report_times;
u64 last_present_ns=0; // guest timestamp of the last present (fixed resolution timing)
// Headroom for the game's dynamic resolution: the game sizes its 3D picture so that the GPU time it measures with
// report timestamps fills its frame budget, which leaves a PC GPU at ~15-16.5 ms of a 16.7 ms frame and no room for
// anything else. The guest's GPU clock (report timestamps and the device's current timestamp alike) runs faster by
// this factor, so the game measures that much more and settles at a resolution with room to spare.
// OOT_GPU_TIME_SCALE=<factor> (1 = the real clock).
double GpuTimeScale(){
    static const double scale=[]{const char* e=std::getenv("OOT_GPU_TIME_SCALE");const double v=e?std::atof(e):1.0;return v>=0.25&&v<=4.0?v:1.0;}();
    return scale;
}
u64 SteadyNs(){return (u64)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
void PreSubmit();
bool GpuReportTimes(){
    auto& vk=Vk();auto& r=report_times;
    if(r.initialized)return r.enabled;
    r.initialized=true;
    if(std::getenv("OOT_HOST_REPORT_TIMES")||!vk.calibrated_timestamps)return false;
    // (a tick of another length, as the Steam Deck's 10 ns: converted on the GPU, timestamp_scale.comp. Host-written times
    // left the game's dynamic resolution at its highest there)
    r.calibrate=reinterpret_cast<PFN_vkGetCalibratedTimestampsKHR>(vkGetDeviceProcAddr(vk.device,vk.calibrated_timestamps));
    if(!r.calibrate)return false;
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};qi.queryType=VK_QUERY_TYPE_TIMESTAMP;qi.queryCount=ReportTimes::Capacity;
    VK_CHECK(vkCreateQueryPool(vk.device,&qi,nullptr,&r.pool));
    r.enabled=true;
    SetPreSubmitHook(PreSubmit);
    return true;
}
// Records the GPU timestamp for the report at `address` (and its 32-bit value, if given).
// False: the host writes them when the batch completes.
bool RecordReportTimestamp(u64 address,const u32* value){
    static const bool fake=std::getenv("OOT_FAKE_FAST_GPU")!=nullptr; // diagnostic: reports stamped at record time
    if(fake){auto* out=static_cast<u64*>(ResolveAddress(address,16));const u64 now=GuestTimestampNs();const u32 v=value?*value:0;const bool has=value!=nullptr;
        AfterFlush([out,now,v,has]{if(has)out[0]=v;out[1]=now;});return true;}
    if(!GpuReportTimes())return false;
    auto& vk=Vk();auto& r=report_times;
    const u32 q=r.next++%ReportTimes::Capacity; // reused long after its batch retired
    vkResetQueryPool(vk.device,r.pool,q,1);
    const auto record=[q](VkCommandBuffer cmd){vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,report_times.pool,q);};
    if(!RecordInPassAsync([](VkCommandBuffer cmd,void* packed,u64){vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,report_times.pool,(u32)reinterpret_cast<u64>(packed));},reinterpret_cast<void*>((u64)q)))ImmediateLight(record);
    r.writes.push_back({q,address,value!=nullptr,value?*value:0});
    NoteWriteBack(address,16);
    return true;
}
// (Before submission, outside render passes.)
struct TimestampScale{VkPipeline pipeline=VK_NULL_HANDLE;VkPipelineLayout layout=VK_NULL_HANDLE;VkBuffer buffer=VK_NULL_HANDLE;VkDeviceMemory memory=VK_NULL_HANDLE;
    VkDeviceAddress address=0;u64 period=0;static constexpr u32 Slots=ReportTimes::Capacity;} timestamp_scale;
bool TimestampScaling(){ // (ticks of another length than 1 ns: the conversion's pipeline and tick buffer, made once)
    auto& vk=Vk();auto& t=timestamp_scale;
    static const bool needed=vk.properties.limits.timestampPeriod*GpuTimeScale()!=1.0;
    if(!needed||t.pipeline)return needed;
    t.period=(u64)std::llround(vk.properties.limits.timestampPeriod*GpuTimeScale()*65536.0);
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=TimestampScale::Slots*8ull;
    bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VK_CHECK(TracedCreateBuffer(&bi,&t.buffer,__FILE__,__LINE__));
    VkMemoryRequirements req;vkGetBufferMemoryRequirements(vk.device,t.buffer,&req);
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};flags.flags=VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.pNext=&flags;ai.allocationSize=req.size;ai.memoryTypeIndex=FindMemoryType(req.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(vk.device,&ai,nullptr,&t.memory));VK_CHECK(vkBindBufferMemory(vk.device,t.buffer,t.memory,0));
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};info.buffer=t.buffer;t.address=vkGetBufferDeviceAddress(vk.device,&info);
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT,0,24};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};li.pushConstantRangeCount=1;li.pPushConstantRanges=&push;
    VK_CHECK(vkCreatePipelineLayout(vk.device,&li,nullptr,&t.layout));
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};mi.codeSize=sizeof timestamp_scale_spv;mi.pCode=timestamp_scale_spv;
    VkShaderModule module;VK_CHECK(vkCreateShaderModule(vk.device,&mi,nullptr,&module));
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};ci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=module;ci.stage.pName="main";ci.layout=t.layout;
    VK_CHECK(vkCreateComputePipelines(vk.device,VK_NULL_HANDLE,1,&ci,nullptr,&t.pipeline));
    RT_LOG("GPU timestamps: %.2f ns ticks, converted on the GPU",vk.properties.limits.timestampPeriod);
    return true;
}
void WriteReportTimestamps(){
    auto& r=report_times;
    if(r.writes.empty())return;
    if(TimestampScaling()){
        auto& t=timestamp_scale;auto& vk=Vk();
        ImmediateLight([&](VkCommandBuffer cmd){
            for(const auto& w:r.writes)vkCmdCopyQueryPoolResults(cmd,r.pool,w.query,1,t.buffer,w.query*8ull,8,VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT);
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,t.pipeline);
            for(const auto& w:r.writes){
                const auto range=GpuLookupAddress(w.address);if(!range.pool||range.available<16)continue;
                VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};info.buffer=range.buffer;
                const u64 push[3]{t.address+w.query*8ull,vkGetBufferDeviceAddress(vk.device,&info)+range.offset+8,t.period};
                vkCmdPushConstants(cmd,t.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof push,push);
                vkCmdDispatch(cmd,1,1,1);
            }
            barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_MEMORY_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT|VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,1,&barrier,0,nullptr,0,nullptr);
            for(const auto& w:r.writes)if(w.has_value){const auto range=GpuLookupAddress(w.address);if(!range.pool||range.available<16)continue;
                vkCmdFillBuffer(cmd,range.buffer,range.offset,4,w.value);vkCmdFillBuffer(cmd,range.buffer,range.offset+4,4,0);}
        });
        r.writes.clear();
        return;
    }
    ImmediateLight([&](VkCommandBuffer cmd){
        for(const auto& w:r.writes){
            const auto range=GpuLookupAddress(w.address);if(!range.pool||range.available<16)continue;
            vkCmdCopyQueryPoolResults(cmd,r.pool,w.query,1,range.buffer,range.offset+8,8,VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT);
            if(w.has_value){vkCmdFillBuffer(cmd,range.buffer,range.offset,4,w.value);vkCmdFillBuffer(cmd,range.buffer,range.offset+4,4,0);}
        }
    });
    r.writes.clear();
}
void PreSubmit(){WriteUnwrittenUniformUpdates();if(gpu_counters.ready)ProcessReports();WriteReportTimestamps();}
// The guest's timestamp clock: GPU time in nanoseconds (steady-clock time without GPU timestamps).
u64 GuestTimestampNs(){
    if(!GpuReportTimes())return SteadyNs();
    auto& vk=Vk();auto& r=report_times;
    const u64 now=SteadyNs();
    if(!r.calibrated_at||now-r.calibrated_at>500'000'000){ // clocks drift slowly
        VkCalibratedTimestampInfoKHR infos[2]{{VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR,nullptr,VK_TIME_DOMAIN_DEVICE_KHR},
                                               {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR,nullptr,VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR}};
        u64 stamps[2]{},deviation=0;
        if(r.calibrate(vk.device,2,infos,stamps,&deviation)==VK_SUCCESS){r.offset=(s64)stamps[1]-(s64)(stamps[0]*(double)vk.properties.limits.timestampPeriod);r.calibrated_at=now;}
    }
    const u64 gpu_now=(u64)((s64)now-r.offset);
    return GpuTimeScale()==1.0?gpu_now:(u64)(gpu_now*GpuTimeScale());
}
void ReportRenderCounter(u32 counter,u64 address,bool flag_applied) {
    auto* out=static_cast<u64*>(ResolveAddress(address,16));
    static const bool force_visible=std::getenv("OOT_SAMPLES_VISIBLE")!=nullptr; // (diagnostic: every sample count large)
    if((frame_skipping||force_visible)&&counter==1){ // (no draws ran: report the object visible so the next rendered frame draws it)
        out[0]=1u<<20;out[1]=GuestTimestampNs();SetKnownReport(address,1u<<20);gpu_counters.reports.erase(address);return;}
    // Reports of other counters (OoT: TIMESTAMP_TOP, 16) are also the game's one-shot flags: its CPU writes 1 into a
    // report's counter word to ask for something drawn under `SetRenderEnableConditional(not equal)` (doors, vines,
    // ornaments on a shield, prompts), and the report after that draw puts the word back to 0. The 0 was written when the
    // batch completed, a frame later, over the game's next request, and was remembered as the word's value for good:
    // those draws never ran again, and their depth-only pre-pass left holes that bloom blew up to white. It is written
    // here, in command order, as Longshot's replay does (vk_replay.cpp, NOP_REPORT_COUNTER). OOT_FLAG_REPORTS_LATE=1:
    // as before.
    {static const bool late=std::getenv("OOT_FLAG_REPORTS_LATE")!=nullptr,dynamic=std::getenv("OOT_DYNAMIC_RESOLUTION")!=nullptr;
     if(counter!=1&&!late&&!dynamic){
        static u64 ticks=0;const u64 stamp=last_present_ns+(++ticks%100000)*100; // (the fixed-resolution time, see below)
        EraseKnownReport(address);
        // (its word was cleared when the game recorded it, commands.cpp; the time still is the executor's to write: the
        // game measures its GPU load from these, and without them fell to its lowest render size, 854 wide)
        if(flag_applied){PrepareHostWrite(address+8,8);out[1]=stamp;return;}
        PrepareHostWrite(address,16);
        out[0]=0;out[1]=stamp;
        return;}}
    // A samples count is only given to the game when it is sure (Longshot's rule, f3fe1763/cd906265): the game leaves out
    // what reads 0, and a wrong 0 takes away something the player sees - here an object's shading for a frame, its
    // depth-only pre-pass left as a hole that bloom blew up (objects glowing for single frames, one in ten frames in the
    // woods). Sure is a run of the game's bounding boxes: depth-only draws that all ran. A run with a mesh draw, a draw
    // that writes colour, a draw that did not run (its pipeline not ready) or no draw at all reads "visible" (4096),
    // written here in command order. OOT_OCC_UNSURE=<bits>: which of those stay counted (1 mesh, 2 colour, 4 not run, 8 empty).
    if(counter==1&&!OcclusionQueries()){
        SetKnownReport(address,4096);gpu_counters.reports.erase(address);
        PrepareHostWrite(address,16);out[0]=4096;out[1]=GuestTimestampNs();
        return;}
    if(counter==1){
        static const u32 keep=std::getenv("OOT_OCC_UNSURE")?(u32)std::atoi(std::getenv("OOT_OCC_UNSURE")):0;
        // (16: a submission went out between the run's start and here. Its queries are numbered per submission, and
        // what was counted before it is not in this report's range: the count read 0 whenever one of the pieces a long
        // list is submitted in ended inside a run - for Link's bounds every fourth frame in the woods, his shading gone
        // for that frame. Pieces now wait for the run's end, commands.cpp; a submission forced by something else
        // leaves the run uncounted.)
        const bool split=samples_run_open&&SubmittedBatches()!=samples_run_batch;
        const u32 cls=(run_note.mesh?1u:0)|(run_note.colour?2u:0)|(run_note.not_run?4u:0)|(!run_note.draws&&!run_note.not_run?8u:0)|(split?16u:0);
        ++run_classes[!cls?0:(cls&4)?3:(cls&1)?1:(cls&2)?2:4];if(split)++run_split;
        run_note={};samples_run_open=false;
        if(cls&~keep){
            SetKnownReport(address,4096);gpu_counters.reports.erase(address);
            PrepareHostWrite(address,16);out[0]=4096;out[1]=GuestTimestampNs();
            if(gpu_counters.ready){gpu_counters.pending.push_back({0,SplitSegment(),~0u,2,0});gpu_counters.last_report=0;} // (its queries belong to no report)
            return;}
    }
    if(counter==1){NotePendingReport(address);EraseKnownReport(address);}
    else SetKnownReport(address,0);
    const bool on_gpu=gpu_counters.ready;
    if(LogThisFrame()&&counter==1){const u32 open=gpu_counters.segment_query;AfterFlush([out,address,open]{RT_LOG("report @%#llx after its batch: %llu samples (segment query open when recorded: %d)",(unsigned long long)address,(unsigned long long)out[0],(int)(open!=~0u));});}
    if(on_gpu){
        auto& g=gpu_counters;
        if(counter==1){
            EnsureCounterRings();
            const u32 slot=g.next_predicate++;
            g.pending.push_back({GuestDeviceAddress(address),SplitSegment(),slot,0,0});
            g.reports[address]={slot,g.last_report!=0&&g.last_report+16==address,g.last_report==0};
            g.last_report=address;
        }else g.pending.push_back({GuestDeviceAddress(address),0,~0u,1,0}); // other counters report 0
        {static u32 shown=0;if(std::getenv("OOT_PREDICATE_TRACE")&&shown++<40)RT_LOG("report counter %u @%#llx",counter,(unsigned long long)address);}
    }
    // Sample counts written on the GPU need nothing more: their timestamps are not read (each
    // timestamp costs a query and a copy; there are thousands of occlusion reports per frame).
    static const bool counter_check=std::getenv("OOT_COUNTER_CHECK")!=nullptr;
    if(on_gpu&&counter==1&&!counter_check)return;
    // Fixed resolution (default): the game's dynamic resolution lowers its render size whenever
    // the GPU time it measures between timestamp reports exceeds its budget. Timestamp reports
    // get the last present time plus a tick per report, so that time stays tiny and the game
    // renders at its full size (1920x1080 here). OOT_DYNAMIC_RESOLUTION=1: real GPU timestamps.
    static const bool dynamic_resolution=std::getenv("OOT_DYNAMIC_RESOLUTION")!=nullptr;
    if(!dynamic_resolution&&counter!=1){
        static u64 ticks=0;
        const u64 stamp=last_present_ns+(++ticks%100000)*100;
        out[1]=stamp;
        if(on_gpu)NoteWriteBack(address,16);
        AfterFlush([out,stamp]{out[0]=0;out[1]=stamp;});
        return;
    }
    const bool gpu_time=RecordReportTimestamp(address,nullptr);
    if(on_gpu&&counter!=1)NoteWriteBack(address,16); // (the GPU writes its 0)
    AfterFlush([out,counter,address,on_gpu,gpu_time] {
        static const bool counter_check=std::getenv("OOT_COUNTER_CHECK")!=nullptr;
        static u32 mismatches=0;
        if(on_gpu&&counter==1&&counter_check&&out[0]!=samples.value&&mismatches++<40)RT_LOG("counter mismatch @%#llx: gpu %llu cpu %llu",(unsigned long long)address,(unsigned long long)out[0],(unsigned long long)samples.value);
        if(!on_gpu||counter!=1)out[0]=counter==1?samples.value:0; // (sample counts come back from the GPU)
        if(!gpu_time)out[1]=GuestTimestampNs();
    });
}
// ---- Draws of several command lists resolved at once ----------------------------------------------------------------
// OoT records its heavy passes (the scene, the shadow maps) as groups of command lists, ~100 draws each, and the
// executor translated them one draw after the other: ~9 us a draw, 17 ms a frame in Castle Town, on one thread while
// the others idled. Resolving a draw (its pipeline, descriptors, buffers) only reads what the executor owns, so the
// lists of a group are resolved on several threads, each from the state its list starts with, and the executor then
// commits the results in command order: render passes, occlusion reports and the command buffer see the same sequence
// as before. A draw that needs more than reading (see batch.h, resolve_bail) ends the parallel part: from there on
// the group runs the usual way. OOT_PARALLEL_DRAWS=<helper threads> (0: off).
namespace {
// A draw or dispatch of the open batch's conditional rendering state, as DrawEnabled reads it.
void ResolveDrawCommand(const Command& command,bool mesh=false){
    State& st=*tls_state;st.primitive=mesh?4:command.x[0];
    if(!st.render_enable)return;
    bool deferred=false;
    if(st.render_condition&&st.render_decided){if(st.render_decided==2)return;} // (one of the game's flags, decided when recorded)
    else if(st.render_condition){
        // Decided here when its operands are final and no report of this group writes them; else when committed.
        const auto& reports=*resolve_out->group_reports;const u64 c=st.render_condition;
        const auto it=std::lower_bound(reports.begin(),reports.end(),c>=15?c-15:0);
        if(it!=reports.end()&&*it<c+24)deferred=true;
        else if(PredicateOperandsFinal()){if(!resolve_bail&&!EvaluatePredicateOnCpu())return;}
        else deferred=true;
        if(resolve_bail)return;
    }
    const size_t before=resolve_out->items.size();
    if(mesh){ // (as ExecuteRenderCommand's DrawMeshTasksIndirect, with the arguments ExecuteSegmentsParallel had converted)
        if(!gfx::Current().foliage_shadows&&!st.color_count&&st.targets.depth){const Texture* shadow=HostTexture(st.targets.depth);if(shadow&&shadow->format==50)return;}
        if((Op)command.op==Op::DrawMeshTasks)Draw(command); // (its task counts are in the command)
        else{
            const auto converted=mesh_preclamped.find(command.x[0]);
            Command direct=command;direct.op=(u32)Op::DrawMeshTasks;direct.x[0]=0;direct.x[1]=0;
            mesh_indirect_slot=converted->second+1;mesh_indirect_source=0;
            Draw(direct);
            mesh_indirect_slot=0;
        }
    }else Draw(command);
    if(deferred&&resolve_out->items.size()>before){auto& item=resolve_out->items.back();
        item.deferred=true;item.condition=st.render_condition;item.if_equal=st.render_if_equal;item.depth=st.depth;item.depth_target=st.targets.depth;}
}
// Commands in a resolved segment that the executor runs itself, in their place and from the state they are met in:
// they record work of their own (a clear, a mesh draw with its argument copy) but leave what later draws resolve alone.
bool IsSerialItem(Op op){return op==Op::ClearColor||op==Op::ClearColori||op==Op::ClearColorui||op==Op::ClearDepthStencil||op==Op::DrawMeshTasks||op==Op::DrawMeshTasksIndirect;}
bool IsDrawCommand(Op op){return op==Op::DrawArrays||op==Op::DrawElements||op==Op::DrawArraysInstanced||op==Op::DrawElementsInstanced||op==Op::DrawElementsBaseVertex;}
void ResolveSegment(const CommandList& list,size_t begin,size_t end,SegmentOutput& out){
    State* const saved=tls_state;
    tls_state=&out.state;resolve_out=&out;resolve_checked=&out.checked;resolve_worker=true;resolve_bail=false;fast_draw.valid=false;fast_draw_allowed=false;
    const bool mesh_ahead=MeshAhead()&&!LogThisFrame()&&!std::getenv("OOT_MESH_INDIRECT_CPU");
    for(size_t i=begin;i<end;++i){
        const Command& command=list.commands[i];
        NoteFastDrawBreaker(command);
        const Op op=(Op)command.op;
        if(IsDrawCommand(op))ResolveDrawCommand(command);
        else if(op==Op::ReportCounter||op==Op::ResetCounter)out.items.push_back({1,(u32)i,nullptr,false,0,false,0,0});
        else if(mesh_ahead&&(op==Op::DrawMeshTasks||(op==Op::DrawMeshTasksIndirect&&mesh_preclamped.count(command.x[0])))){ResolveDrawCommand(command,true);++mesh_ahead_stats[resolve_bail?1:0];}
        else if(IsSerialItem(op)){
            if(op==Op::DrawMeshTasks)++mesh_ahead_stats[2];else if(op==Op::DrawMeshTasksIndirect)++mesh_ahead_stats[3];
            if(op==Op::DrawMeshTasks||op==Op::DrawMeshTasksIndirect)out.state.primitive=4;
            if(out.snapshots_used==out.snapshots.size())out.snapshots.emplace_back();
            out.snapshots[out.snapshots_used]=out.state;out.items.push_back({2,(u32)i,nullptr,false,out.snapshots_used++,false,0,0});
        }
        else if(op!=Op::Barrier)ExecuteRenderCommand(command);
        if(resolve_bail&&(IsDrawCommand(op)||op==Op::DrawMeshTasksIndirect||op==Op::DrawMeshTasks)){
            // A draw that needs the executor (a texture with no image yet, a pipeline to build): the executor runs that
            // draw in its place, from this state, and the draws after it are resolved here as before.
            if(out.snapshots_used==out.snapshots.size())out.snapshots.emplace_back();
            out.snapshots[out.snapshots_used]=out.state;out.items.push_back({2,(u32)i,nullptr,false,out.snapshots_used++,false,0,0});
            out.reasons.push_back({resolve_bail_file,resolve_bail_line});
            resolve_bail=false;fast_draw.valid=false;fast_draw_allowed=false;
            continue;
        }
        if(resolve_bail){out.bail=i;out.bail_file=resolve_bail_file;out.bail_line=resolve_bail_line;break;}
    }
    resolve_worker=false;resolve_bail=false;resolve_out=nullptr;resolve_checked=nullptr;tls_state=saved;fast_draw.valid=false;fast_draw_allowed=false;
}
// The segment's state changes alone: what the next segment starts from.
void ApplyStateOnly(const CommandList& list,size_t begin,size_t end){
    for(size_t i=begin;i<end;++i){
        const Command& command=list.commands[i];const Op op=(Op)command.op;
        if(IsDrawCommand(op))(*tls_state).primitive=command.x[0];
        else if(IsSerialItem(op)){if(op==Op::DrawMeshTasks||op==Op::DrawMeshTasksIndirect)(*tls_state).primitive=4;}
        else if(op!=Op::ReportCounter&&op!=Op::ResetCounter&&op!=Op::Barrier)ExecuteRenderCommand(command);
        if(op==Op::SetRenderTargets){ // (render targets get their images now: the threads resolving the draws only read)
            const auto& targets=(*tls_state).targets;
            for(u32 c=0;c<(*tls_state).color_count;++c)if(targets.colors[c])if(Texture* t=HostTexture(targets.colors[c]))EnsureImage(*t);
            if(targets.depth)if(Texture* t=HostTexture(targets.depth))EnsureImage(*t);
        }
    }
}
void CommitResolvedDraw(DrawPacket& pk){
    if(BatchTraceOn()){char note[256];std::snprintf(note,sizeof note,"[rdraw op%u x=%llx,%llx,%llx,%llx,%llx,%llx fs%llx %ux%u vp%.0fx%.0f]",pk.op,(unsigned long long)pk.x[1],(unsigned long long)pk.x[2],(unsigned long long)pk.x[3],
        (unsigned long long)pk.x[4],(unsigned long long)pk.x[5],(unsigned long long)pk.x[6],(unsigned long long)pk.profile_key,pk.extent.width,pk.extent.height,pk.viewport.width,pk.viewport.height);BatchTraceNote(note);}
    ++run_note.draws;run_note.mesh+=pk.mesh;run_note.colour+=pk.colour_writes;
    for(auto* color:pk.colors)if(color)color->gpu_written=true;
    if(pk.depth_target)pk.depth_target->gpu_written=true;
    const bool count_samples=gpu_counters.ready&&samples.enabled;
    static const bool async_enabled=AsyncDrawsEnabled();
    PassDraw(pk.key,[&](VkCommandBuffer cmd){
        for(auto* color:pk.colors)if(color&&color->layout!=VK_IMAGE_LAYOUT_GENERAL)Transition(cmd,*color,VK_IMAGE_LAYOUT_GENERAL);
        Texture* const depth=pk.depth_target;
        if(depth&&depth->layout!=VK_IMAGE_LAYOUT_GENERAL)Transition(cmd,*depth,VK_IMAGE_LAYOUT_GENERAL);
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};ri.renderArea.extent=pk.extent;ri.layerCount=1;ri.colorAttachmentCount=pk.color_count;ri.pColorAttachments=pk.attachments.data();
        if(depth&&depth->info.depth)ri.pDepthAttachment=&pk.da;if(depth&&depth->info.stencil)ri.pStencilAttachment=&pk.da;
        vkCmdBeginRendering(cmd,&ri);
        if(count_samples){ // the pass instance's first occlusion segment (as Draw)
            const u32 q=samples.next++;vkCmdBeginQuery(cmd,samples.pool,q,VK_QUERY_CONTROL_PRECISE_BIT);gpu_counters.segment_query=q;
            SetPassEndHook([q](VkCommandBuffer c){vkCmdEndQuery(c,samples.pool,q);gpu_counters.segment_query=~0u;});}
    },[&](VkCommandBuffer cmd){RecordDrawPacket(cmd,&pk,OtherWorkSerial());},pk.profile_key,nullptr,async_enabled?RecordDrawPacket:nullptr,&pk);
}
// False: from some command on the segment ran the usual way (to its end), and the segments after it must too.
bool CommitSegment(SegmentOutput& out,const CommandList& list,size_t end,const std::function<bool()>& resolutions_hold){
    for(auto& item:out.items){
        if(item.kind==1){const Command& c=list.commands[item.index];
            if((Op)c.op==Op::ReportCounter)ReportRenderCounter((u32)c.x[0],c.x[1]);else ResetRenderCounter((u32)c.x[0]);
            continue;}
        if(item.kind==2){
            // (what the later draws resolved may no longer hold after it: see ExecuteSegmentsParallel)
            main_state=out.snapshots[item.condition];fast_draw.valid=false;fast_draw_allowed=false;
            {static const bool times=std::getenv("OOT_PARALLEL_TIMES")!=nullptr; // (diagnostic: what the commands the executor runs itself cost, by kind)
             if(times){static u64 ns[3]{},n[3]{},at=0;const Op o=(Op)list.commands[item.index].op;const u32 kind=o==Op::DrawMeshTasks||o==Op::DrawMeshTasksIndirect?0:IsDrawCommand(o)?1:2;
                const auto t0=std::chrono::steady_clock::now();ExecuteRenderCommand(list.commands[item.index]);
                ns[kind]+=(u64)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-t0).count();++n[kind];
                if(at!=frame_number&&frame_number%300==0){at=frame_number;RT_LOG("executor's own commands a frame: %.1f mesh draws %.2f ms, %.1f other draws %.2f ms, %.1f clears %.2f ms",n[0]/300.0,ns[0]/300e6,n[1]/300.0,ns[1]/300e6,n[2]/300.0,ns[2]/300e6);RT_LOG("mesh draws a frame: %.1f resolved ahead, %.1f tried and left to the executor, %.1f direct, %.1f indirect not tried",mesh_ahead_stats[0].exchange(0)/300.0,mesh_ahead_stats[1].exchange(0)/300.0,mesh_ahead_stats[2].exchange(0)/300.0,mesh_ahead_stats[3].exchange(0)/300.0);std::memset(ns,0,sizeof ns);std::memset(n,0,sizeof n);}
             }else ExecuteRenderCommand(list.commands[item.index]);}
            if(resolutions_hold())continue;
            ExecuteSegmentSerial(list,item.index+1,end);
            return false;
        }
        DrawPacket& pk=*item.packet;
        if(item.deferred){ // (the condition as the draw had it: DrawEnabled reads the executor's state)
            State& m=main_state;m.render_enable=true;m.render_condition=item.condition;m.render_if_equal=item.if_equal;m.depth=item.depth;m.targets.depth=item.depth_target;
            if(!DrawEnabled()){++stats.skipped;continue;}
            pk.predicate=draw_predicate;pk.inverted=draw_inverted;draw_predicate=~0u;
        }
        if(!pk.mesh)LsNoteDraw(pk.colors[0],pk.depth_target,pk.extent,pk.ls_camera,pk.ls_camera_size);
        if(pk.ls_skinned)LsNoteSkinned(pk.ls_skin,pk.depth_target,pk.extent);
        if(gpu_counters.ready&&samples.enabled)EnsureCounterRings();
        for(Texture* t:pk.to_general){EnsureUploaded(*t);if(t->layout!=VK_IMAGE_LAYOUT_GENERAL)Immediate([&](VkCommandBuffer cmd){Transition(cmd,*t,VK_IMAGE_LAYOUT_GENERAL);});}
        if(pk.set_pipeline)pk.set=WriteBatchSet(*pk.set_pipeline,pk.writes);
        CommitResolvedDraw(pk);++stats.draws;
        if(pk.end_pass_after)EndPass(); // (it writes a storage image: later work must see that)
    }
    if(out.bail==~size_t{0})return true;
    // The rest of the segment runs the usual way, from the state the command that needed the executor starts with.
    main_state=out.state;fast_draw.valid=false;fast_draw_allowed=false;
    ExecuteSegmentSerial(list,out.bail,end);
    return resolutions_hold();
}
struct ResolvePool {
    std::vector<std::thread> threads;
    std::mutex mutex;std::condition_variable wake;
    const std::vector<ParallelSegment>* segments=nullptr;std::vector<std::unique_ptr<SegmentOutput>> outputs;u32 count=0;
    std::atomic<u32> next{0},done{0},active{0};std::atomic<u64> generation{0};std::atomic<bool> open{false};
    void Work(){for(;;){const u32 k=next.fetch_add(1);if(k>=count)break;const auto& seg=(*segments)[k];ResolveSegment(*seg.list,seg.begin,seg.end,*outputs[k]);done.fetch_add(1);}}
    void Loop(){
        rt::SetThreadName("nvn-resolve");
        u64 seen=0;
        for(;;){
            // (groups follow each other within a frame: spin through the gaps, sleep between frames)
            int spins=0;u64 now;
            while((now=generation.load())==seen){
                static const int spin_limit=std::getenv("OOT_RESOLVE_SPIN")?std::atoi(std::getenv("OOT_RESOLVE_SPIN")):20000;
                if(++spins<spin_limit){
#if defined(__x86_64__)||defined(__i386__)
                    __builtin_ia32_pause();
#endif
                    continue;}
                std::unique_lock lock{mutex};wake.wait(lock,[&]{return generation.load()!=seen;});
            }
            seen=now;
            active.fetch_add(1);
            if(open.load()&&generation.load()==now)Work();
            active.fetch_sub(1);
        }
    }
};
ResolvePool& Resolvers(){
    static ResolvePool pool;static std::once_flag once;
    std::call_once(once,[]{for(unsigned i=0;i<ParallelResolveThreads();++i)pool.threads.emplace_back([]{pool.Loop();});for(auto& t:pool.threads)t.detach();});
    return pool;
}
struct ParallelStats{u64 groups=0,segments=0,draws=0,bails=0,serial_segments=0,depth_groups=0;std::map<std::pair<std::string,int>,u64> why;} parallel_stats;
}
unsigned ParallelResolveThreads(){
    static const unsigned n=[]{if(const char* e=std::getenv("OOT_PARALLEL_DRAWS"))return (unsigned)std::clamp(std::atoi(e),0,15);
        const unsigned hw=std::thread::hardware_concurrency(); // (the game's three cores, the executor and the recorder keep theirs)
        return hw>=16?7u:hw>=12?5u:hw>=8?3u:hw>=6?2u:0u;}();
    return n;
}
bool ParallelDrawsUsable(){
    // (where texture memory is not watched - Windows, AMD without placed maps - a draw resolved ahead cannot know its
    // textures are current: each is handed to the executor, which checks it by content when the draw is committed
    // (EnsureUploaded's resolve_texture_unready). Draws were not resolved ahead at all there: 38 against 53 fps in a
    // heavy view. OOT_PARALLEL_UNWATCHED=0: not resolved ahead, as before.)
    static const bool unwatched=!std::getenv("OOT_PARALLEL_UNWATCHED")||std::atoi(std::getenv("OOT_PARALLEL_UNWATCHED"))!=0;
    static const bool diagnostics=std::getenv("OOT_PASS_HAZARDS")||std::getenv("OOT_GPU_TRACE")||std::getenv("OOT_DRAW_CPU_PROFILE")||std::getenv("OOT_COUNTER_CHECK")||std::getenv("OOT_SKIP_SHADER")||std::getenv("OOT_OLD_PIPELINE")||std::getenv("OOT_SYNC_PIPELINES");
    return ParallelResolveThreads()&&!diagnostics&&!FrameSkipOn()&&!LogThisFrame()&&!GpuProfilingNow()&&(gpu_counters.ready||!samples.enabled)&&ring.buffer&&(TextureWatchUsable()||unwatched);
}
void ExecuteSegmentsParallel(const std::vector<ParallelSegment>& segments,const std::vector<u64>& report_addresses){
    // OOT_PARALLEL_TIMES=1 (diagnostic): where a group's time goes on the executor.
    static const bool times=std::getenv("OOT_PARALLEL_TIMES")!=nullptr;
    static u64 tn[4]{},tg=0,td=0,tmax=0,tat=0;
    const auto clock=[]{return (u64)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();};
    u64 t0=times?clock():0,t1=0,t2=0,t3=0;
    struct TimesEnd{const bool on;u64& t0;u64& t1;u64& t2;u64& t3;const std::function<u64()> clock;~TimesEnd(){if(!on||!t3)return;const u64 t4=clock();tn[0]+=t1-t0;tn[1]+=t2-t1;tn[2]+=t3-t2;tn[3]+=t4-t3;++tg;
        if(tat!=frame_number&&frame_number%300==0){tat=frame_number;RT_LOG("parallel groups a frame: %.1f, %.0f draws, the largest list %.0f%% of its group's; per frame: state pass %.2f ms, executor's own share %.2f ms, waiting for the others %.2f ms, committing %.2f ms",
            tg/300.0,td/300.0,tg?tmax*1.0/tg:0.0,tn[0]/300e6,tn[1]/300e6,tn[2]/300e6,tn[3]/300e6);tn[0]=tn[1]=tn[2]=tn[3]=tg=td=tmax=0;}}} times_end{times,t0,t1,t2,t3,clock};
    std::lock_guard lock{BatchMutex()}; // (held throughout: nothing is recorded, submitted or retired while draws are resolved)
    WaitAsyncDraws();                   // (the packets of the group before are reused)
    auto& pool=Resolvers();
    while(pool.outputs.size()<segments.size())pool.outputs.push_back(std::make_unique<SegmentOutput>());
    static State entry_state;entry_state=main_state;
    u32 depth_writing=0,other=0;
    for(size_t k=0;k<segments.size();++k){ // each segment's entry state
        auto& out=*pool.outputs[k];out.items.clear();out.used=0;out.snapshots_used=0;out.reasons.clear();out.checked.clear();out.bail=~size_t{0};out.group_reports=&report_addresses;
        out.state=main_state;
        {u32 draws=0;for(size_t i=segments[k].begin;i<segments[k].end;++i){const Op op=(Op)segments[k].list->commands[i].op;draws+=IsDrawCommand(op)||op==Op::DrawMeshTasks||op==Op::DrawMeshTasksIndirect;}
         ((out.state.depth&3)==3&&out.state.targets.depth?depth_writing:other)+=draws;}
        ApplyStateOnly(*segments[k].list,segments[k].begin,segments[k].end);
    }
    // Groups that mostly write depth (the pass of occlusion tests, the shadow maps) run the usual way, in order.
    // Resolved ahead, about one frame in ninety came out with a result of theirs wrong: no haze for that frame and the
    // white light in a house's doorway (a flash). Which step goes wrong is not found yet (not the mesh draws or clears
    // among them, not the recorder thread, uploads, fences, image sharing or the counter rings); in order there is none
    // in 4,500 frames. It costs ~8% of the frame rate in Kokiri Forest. OOT_PARALLEL_DEPTH_PASSES=1: resolved ahead too.
    // (2026-11-16: with the game's occlusion reports no longer counted - OcclusionQueries above - these groups show no such
    // frame resolved ahead either: 0 in 5,300 frames at the door, 0 spikes in the woods. So the fault lay in the counts
    // of runs resolved ahead, and they are resolved ahead again unless the counts are on. OOT_SERIAL_DEPTH_PASSES=1: in order.)
    {static const bool all=std::getenv("OOT_PARALLEL_DEPTH_PASSES")!=nullptr||(!OcclusionQueries()&&std::getenv("OOT_SERIAL_DEPTH_PASSES")==nullptr);
     if(!all&&depth_writing>other){
        main_state=entry_state;fast_draw.valid=false;fast_draw_allowed=false;
        for(const auto& segment:segments)ExecuteSegmentSerial(*segment.list,segment.begin,segment.end);
        ++parallel_stats.depth_groups;
        return;}}
    {static const bool trace=std::getenv("OOT_PARALLEL_GROUP_TRACE")!=nullptr;static u32 shown=0; // (diagnostic: what each segment draws into)
     static const u64 from=std::getenv("OOT_PARALLEL_GROUP_TRACE")?std::strtoull(std::getenv("OOT_PARALLEL_GROUP_TRACE"),nullptr,10):0;
     if(trace&&frame_number>=from&&shown<400)for(size_t k=0;k<segments.size();++k,++shown){const State& st=pool.outputs[k]->state;
        Texture* c=st.color_count&&st.targets.colors[0]?HostTexture(st.targets.colors[0]):nullptr;Texture* d=st.targets.depth?HostTexture(st.targets.depth):nullptr;
        u32 draws=0,mesh=0,reports=0;for(size_t i=segments[k].begin;i<segments[k].end;++i){const Op op=(Op)segments[k].list->commands[i].op;draws+=IsDrawCommand(op);mesh+=op==Op::DrawMeshTasks||op==Op::DrawMeshTasksIndirect;reports+=op==Op::ReportCounter;}
        RT_LOG("  frame %llu segment %zu/%zu: colour %ux%u f%u (%u targets) depth %ux%u, state depth %#x cond %#llx; %u draws %u mesh %u reports",(unsigned long long)frame_number,k,segments.size(),c?c->width:0,c?c->height:0,c?c->format:0,st.color_count,
            d?d->width:0,d?d->height:0,(u32)st.depth,(unsigned long long)st.render_condition,draws,mesh,reports);}}
    static State exit_state;exit_state=main_state;
    // (the group's indirect mesh draws, found before the helper threads start: they resolve those draws from the slots
    // given here. The conversion itself is recorded below, before anything of the group is committed.)
    static std::vector<u64> mesh_sources;bool mesh_scanned=false;
    {static const bool preclamp=!std::getenv("OOT_MESH_PRECLAMP")||std::atoi(std::getenv("OOT_MESH_PRECLAMP"))!=0;
     static const bool cpu_args=std::getenv("OOT_MESH_INDIRECT_CPU")!=nullptr;
     if(MeshAhead()&&preclamp&&!cpu_args){mesh_scanned=true;mesh_sources.clear();mesh_preclamped.clear();
        for(const auto& seg:segments)for(size_t i=seg.begin;i<seg.end;++i)if((Op)seg.list->commands[i].op==Op::DrawMeshTasksIndirect){
            const u64 address=seg.list->commands[i].x[0];
            if(mesh_preclamped.count(address)||mesh_preclamped.size()>=IndirectClamp::Slots/4)continue;
            mesh_preclamped[address]=u64(indirect_clamp.next++%IndirectClamp::Slots)*16;mesh_sources.push_back(address);}}}
    pool.segments=&segments;pool.count=(u32)segments.size();pool.next.store(0);pool.done.store(0);pool.open.store(true);
    pool.generation.fetch_add(1);
    {std::lock_guard wake_lock{pool.mutex};}pool.wake.notify_all();
    if(times){t1=clock();u32 all=0,most=0;for(const auto& seg:segments){u32 d=0;for(size_t i=seg.begin;i<seg.end;++i)d+=IsDrawCommand((Op)seg.list->commands[i].op);all+=d;most=std::max(most,d);}td+=all;tmax+=all?most*100/all:0;}
    pool.Work();
    if(times)t2=clock();
    while(pool.done.load()<pool.count){
#if defined(__x86_64__)||defined(__i386__)
        __builtin_ia32_pause();
#endif
    }
    pool.open.store(false);
    while(pool.active.load()){
#if defined(__x86_64__)||defined(__i386__)
        __builtin_ia32_pause();
#endif
    }
    if(times)t3=clock();
    ++frame_groups;
    ++parallel_stats.groups;parallel_stats.segments+=segments.size();
    // The indirect mesh draws of the group: their arguments are converted for all of them here, in one piece of work,
    // instead of one between every two of them (each ended the render pass and put a compute dispatch between its
    // draws: ~30 more passes a frame in Castle Town). Nothing in a group writes those arguments: the culling passes
    // that do are compute dispatches, which no group holds. OOT_MESH_PRECLAMP=0: each at its draw, as before.
    {static const bool preclamp=!std::getenv("OOT_MESH_PRECLAMP")||std::atoi(std::getenv("OOT_MESH_PRECLAMP"))!=0;
     static const bool cpu_args=std::getenv("OOT_MESH_INDIRECT_CPU")!=nullptr;
     auto& sources=mesh_sources;
     if(!mesh_scanned){sources.clear();mesh_preclamped.clear();}
     if(preclamp&&!cpu_args&&!mesh_scanned)for(size_t j=0;j<segments.size();++j)for(const auto& item:pool.outputs[j]->items)
        if(item.kind==2&&(Op)segments[j].list->commands[item.index].op==Op::DrawMeshTasksIndirect){
            const u64 address=segments[j].list->commands[item.index].x[0];
            if(mesh_preclamped.count(address)||mesh_preclamped.size()>=IndirectClamp::Slots/4)continue;
            mesh_preclamped[address]=u64(indirect_clamp.next++%IndirectClamp::Slots)*16;sources.push_back(address);}
     if(!sources.empty()){
        InitializeIndirectClamp();auto& c=indirect_clamp;
        ImmediateLight([&](VkCommandBuffer cmd){
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,c.pipeline);
            for(const u64 address:sources){
                const u64 push[3]{GuestDeviceAddress(address),(c.address+mesh_preclamped[address])|1,FoliageKeep()};
                vkCmdPushConstants(cmd,c.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof push,push);
                vkCmdDispatch(cmd,1,1,1);}
            barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,0,1,&barrier,0,nullptr,0,nullptr);
        });}}
    // After the executor ran a command itself: what the draws from segment `k` on resolved still holds unless the ring
    // started over (their constants may be overwritten) or GPU writes were noted to guest memory their resolution
    // asked about (handles and pointers read on the CPU: read too early then).
    static std::vector<std::pair<u64,u64>> noted;noted.clear();pending_write_watch=&noted;
    const u64 rings=ring_generation;size_t k=0;
    static const bool strict=std::getenv("OOT_PARALLEL_STRICT")!=nullptr; // (after any such command the rest runs the usual way)
    const std::function<bool()> resolutions_hold=[&]{
        if(rings!=ring_generation)return false;
        if(noted.empty())return true;
        if(strict)return false;
        for(size_t j=k;j<segments.size();++j)for(const auto& [address,size]:pool.outputs[j]->checked)
            for(const auto& [written,bytes]:noted)if(address<written+bytes&&written<address+size)return false;
        noted.clear();
        return true;
    };
    bool bailed=false;
    for(;k<segments.size();++k){
        auto& out=*pool.outputs[k];
        const bool kept=CommitSegment(out,*segments[k].list,segments[k].end,resolutions_hold);
        parallel_stats.draws+=out.used;
        if(out.bail!=~size_t{0})++frame_bails;frame_bails+=out.reasons.size();if(!kept)++frame_fallbacks;
        if(out.bail!=~size_t{0}){++parallel_stats.bails;++parallel_stats.why[{out.bail_file,out.bail_line}];}
        for(const auto& [file,line]:out.reasons){++parallel_stats.bails;++parallel_stats.why[{file,line}];}
        if(kept){
            // The GPU starts on a long group's first lists while the rest are committed (as SubmitWhileRecording does
            // between the queue's segments; a group holds the batch lock from end to end). Castle Town's main pass was 7 ms
            // of committing with nothing submitted, then 11 ms of GPU work in one batch that the game's next frame
            // waited for (Steam Deck, 2026-11-17). OOT_GROUP_SUBMIT_US=<us> (0: only between groups).
            static const u64 every=std::getenv("OOT_GROUP_SUBMIT_US")?std::strtoull(std::getenv("OOT_GROUP_SUBMIT_US"),nullptr,10)*1000:PoolsInGpuMemory()?500'000:0;
            if(every&&k+1<segments.size()){
                static auto last=std::chrono::steady_clock::now();static u64 flushes_seen=0;
                const auto now=std::chrono::steady_clock::now();
                if(FlushCount()!=flushes_seen){flushes_seen=FlushCount();last=now;}
                else if((u64)std::chrono::duration_cast<std::chrono::nanoseconds>(now-last).count()>=every){Submit();flushes_seen=FlushCount();last=now;}
            }
            continue;}
        parallel_stats.serial_segments+=segments.size()-k-1;
        for(size_t j=k+1;j<segments.size();++j)ExecuteSegmentSerial(*segments[j].list,segments[j].begin,segments[j].end);
        bailed=true;break;
    }
    pending_write_watch=nullptr;mesh_preclamped.clear();
    if(!bailed)main_state=exit_state;
    fast_draw.valid=false;fast_draw_allowed=false;
}
void ReportParallelStats(){
    auto& p=parallel_stats;
    if(!p.groups)return;
    std::string why;std::vector<std::pair<u64,std::pair<std::string,int>>> top;for(auto& [k,n]:p.why)top.push_back({n,k});std::sort(top.rbegin(),top.rend());
    for(size_t i=0;i<top.size()&&i<6;++i){const auto slash=top[i].second.first.find_last_of('/');char b[96];std::snprintf(b,sizeof b," %s:%d x%.1f",top[i].second.first.c_str()+(slash==std::string::npos?0:slash+1),top[i].second.second,top[i].first/300.0);why+=b;}
    RT_LOG("parallel draws per frame: %.1f groups of %.1f lists, %.0f draws resolved on %u threads; %.2f draws needed the executor (%.1f lists run serially after one), %.1f depth-writing groups run in order%s%s",
        p.groups/300.0,(double)p.segments/std::max<u64>(p.groups,1),p.draws/300.0,ParallelResolveThreads()+1,p.bails/300.0,p.serial_segments/300.0,p.depth_groups/300.0,why.empty()?"":" at",why.c_str());
    p=ParallelStats{};
    {static const bool rs=std::getenv("OOT_RUN_STATS")!=nullptr;if(rs)RT_LOG("occlusion runs in 300 frames: %llu counted, %llu with a mesh draw, %llu with a colour draw, %llu with a draw not run, %llu empty; %llu split by a submission",(unsigned long long)run_classes[0],(unsigned long long)run_classes[1],(unsigned long long)run_classes[2],(unsigned long long)run_classes[3],(unsigned long long)run_classes[4],(unsigned long long)run_split);
     for(auto& c:run_classes)c=0;run_split=0;}
    if(counter_ring_flushes){RT_LOG("occlusion counter rings started over %llu times in 300 frames",(unsigned long long)counter_ring_flushes);counter_ring_flushes=0;}
}
std::shared_ptr<void> SaveRenderState() { return std::make_shared<State>((*tls_state)); }
void RestoreRenderState(const std::shared_ptr<void>& saved) { (*tls_state) = *static_cast<const State*>(saved.get()); }
bool FrameLogged() { return LogThisFrame(); }
u64 CurrentFrameNumber() { return frame_number; }
bool SamplesRunOpen() { return samples_run_open; }
bool CurrentFrameSkipped() { return frame_skipping; }
bool FrameSkipEnabled() { return FrameSkipOn(); }
void NextFrame() {
    LsFrame();
    {static const bool events=std::getenv("OOT_FRAME_EVENTS")!=nullptr; // (diagnostic: one line a frame of what was unusual in it)
     if(events){static u64 flushes=0,batches=0,pools=0;const u64 f=FlushCount(),b=SubmittedBatches(),g=PoolGeneration();
        RT_LOG("events frame %llu: flushes %llu submits %llu pools %llu images %llu spares %llu groups %llu bails %llu fallbacks %llu skipped %llu cond %llu/%llu/%llu clears skipped %llu predicated %llu net %llu:%04llx disp %llu:%04llx",(unsigned long long)frame_number,(unsigned long long)(f-flushes),(unsigned long long)(b-batches),
            (unsigned long long)(g-pools),(unsigned long long)frame_images_made.exchange(0),(unsigned long long)frame_spares_taken.exchange(0),(unsigned long long)frame_groups,(unsigned long long)frame_bails,(unsigned long long)frame_fallbacks,(unsigned long long)stats.skipped,(unsigned long long)frame_cond_drawn,(unsigned long long)frame_cond_skipped,(unsigned long long)frame_cond_gpu,(unsigned long long)frame_clears_skipped,(unsigned long long)frame_clears_predicated,(unsigned long long)frame_net,(unsigned long long)(frame_net_hash&0xffff),(unsigned long long)frame_dispatches,(unsigned long long)(frame_dispatch_hash&0xffff));
        flushes=f;batches=b;pools=g;frame_groups=frame_bails=frame_fallbacks=0;frame_cond_drawn=frame_cond_skipped=frame_cond_gpu=0;frame_clears_skipped=frame_clears_predicated=0;frame_net=frame_net_hash=frame_dispatches=frame_dispatch_hash=0;}}
    if(FrameSkipOn()){
        // Wall time per frame, rendered and skipped kept apart; every 30 frames the smallest skip
        // level whose average frame stays under ~17 ms is chosen (render 1 of skip_level + 1).
        static auto previous=std::chrono::steady_clock::now();static double rendered_ms=16,skipped_ms=8;
        const auto now=std::chrono::steady_clock::now();const double ms=std::chrono::duration<double,std::milli>(now-previous).count();previous=now;
        if(ms<200){double& e=frame_skipping?skipped_ms:rendered_ms;e=e*0.9+ms*0.1;}
        if(frame_number%30==0){
            // OOT_FRAME_SKIP_MAX: most frames skipped per rendered one (default 1: at least every
            // other frame is shown; the game may then run below full speed in the heaviest scenes).
            static const u32 max_level=std::getenv("OOT_FRAME_SKIP_MAX")?(u32)std::atoi(std::getenv("OOT_FRAME_SKIP_MAX")):1;
            u32 k=0;while(k<max_level&&(rendered_ms+k*skipped_ms)/(k+1)>17.2)++k;skip_level=k;}
        if(frame_number%300==0){static auto last=now;const double secs=std::chrono::duration<double>(now-last).count();last=now;RT_LOG("frame skip: %.1f game frames/s, level %u (rendered frame %.1f ms, skipped frame %.1f ms), %llu draws/dispatches dropped",300/secs,skip_level,rendered_ms,skipped_ms,(unsigned long long)frame_skip_dropped);frame_skip_dropped=0;}
    }
    ++frame_number;
    if(frame_number%30==0)PollGpuProfileSwitch();
    frame_skipping=FrameSkipOn()&&skip_level&&frame_number%(skip_level+1)!=0;
    if(frame_number%300==0){ReportParallelStats();ReportDrawCpuProfile();ReportGpuProfile();{static u64 lf=0,ld=0;static const bool st=std::getenv("OOT_SET_STATS")!=nullptr;if(st&&stats.draws>ld)RT_LOG("fast draws: %.0f%% of %.0f draws per frame",100.0*(fast_draw_count-lf)/(stats.draws-ld),(stats.draws-ld)/300.0);if(st){std::string b;for(u32 i=0;i<128;++i)if(fast_draw_breakers[i]>300){char t[32];std::snprintf(t,sizeof t," op%u:%.1f",i,fast_draw_breakers[i]/300.0);b+=t;}
            RT_LOG("  fast path ended by (per frame):%s; declined: flush %.1f pools %.1f pass %.1f shader %.1f constants %.1f texture handle %.1f texture not ready %.1f storage %.1f; textures patched %.1f, pipelines changed %.1f; resolutions taken back %.1f, made anew %.1f (new batch %.1f, first of its pipeline %.1f, stale %.1f, patch failed %.1f; not kept: other resources %.1f, ineligible %.1f)",b.c_str(),fast_draw_fails[0]/300.0,fast_draw_fails[1]/300.0,fast_draw_fails[2]/300.0,fast_draw_fails[3]/300.0,fast_draw_fails[4]/300.0,fast_draw_fails[5]/300.0,fast_draw_fails[6]/300.0,fast_draw_fails[7]/300.0,fast_texture_patches.exchange(0)/300.0,fast_pipeline_swaps.exchange(0)/300.0,resolve_cache_hits.exchange(0)/300.0,resolve_cache_misses.exchange(0)/300.0,resolve_cache_why[0].exchange(0)/300.0,resolve_cache_why[1].exchange(0)/300.0,resolve_cache_why[2].exchange(0)/300.0,resolve_cache_why[3].exchange(0)/300.0,resolve_cache_why[4].exchange(0)/300.0,resolve_cache_why[5].exchange(0)/300.0);
            {std::string w;for(u32 i=0;i<136;++i)if(slow_draw_why[i]>150){char t[40];std::snprintf(t,sizeof t,i<128?" op%u:%.1f":i==128?" nopush:%.1f":i==129?" ineligible:%.1f":i==130?" writes:%.1f":i==131?" predicate:%.1f":" mesh:%.1f",i<128?i:0,slow_draw_why[i]/300.0);if(i>=128)std::snprintf(t,sizeof t,i==128?" nopush:%.1f":i==129?" ineligible:%.1f":i==130?" writes:%.1f":i==131?" predicate:%.1f":" mesh:%.1f",slow_draw_why[i]/300.0);w+=t;}RT_LOG("  full-path draws caused by (per frame):%s",w.c_str());std::memset(slow_draw_why,0,sizeof slow_draw_why);}std::memset(fast_draw_breakers,0,sizeof fast_draw_breakers);std::memset(fast_draw_fails,0,sizeof fast_draw_fails);}lf=fast_draw_count;ld=stats.draws;}static const bool rate=std::getenv("OOT_BINDING_RATE")!=nullptr;if(rate)ReportBindingRate();}
    { static std::map<u32,u32> widths; if(frame_render_width>1)++widths[frame_render_width]; frame_render_width=0;
      if(frame_number%300==0&&!widths.empty()){std::string t;for(auto& [w,n]:widths){char b[32];std::snprintf(b,sizeof b," %u:%u",w,n);t+=b;}RT_LOG("render widths (frames):%s",t.c_str());widths.clear();} }
    static const char* trigger = std::getenv("OOT_DRAW_LOG_TRIGGER");
    if (trigger && std::remove(trigger) == 0) triggered_frame = (s64)frame_number;
    if (LogThisFrame()) RT_LOG("==== logged frame %llu", (unsigned long long)frame_number);
}
void ReportPredicateStats() {
    RT_LOG("conditional draws: %llu drawn as depth tested, %llu decided on the CPU, %llu after a flush, %llu bracketed GPU, %llu with a predicate dispatch; %llu report passes for %llu reports; %llu waited for submitted reports (%.2f ms a frame)",
        (unsigned long long)predicate_stats.depth_tested,(unsigned long long)predicate_stats.cpu,(unsigned long long)predicate_stats.flushed,(unsigned long long)predicate_stats.bracketed,(unsigned long long)predicate_stats.dispatched,
        (unsigned long long)predicate_stats.processing,(unsigned long long)predicate_stats.processed_reports,(unsigned long long)predicate_stats.waited,predicate_stats.wait_ns/300e6);
    predicate_stats={};
}
}
namespace nvn { extern std::atomic<u64> upload_ns_frame,uploads_frame,upload_staging_ns,upload_swizzle_ns,upload_decode_ns; }
namespace nvn {
// Called per presented frame with its duration: a slow frame logs what it spent time on.
void ReportSlowFrame(double ms){
    static u64 last_pending=0;
    const double up=upload_ns_frame.exchange(0)/1e6;const u64 n=uploads_frame.exchange(0);
    const u64 pending=stats.pending-std::min(stats.pending,last_pending);last_pending=stats.pending;
    static const double threshold=std::getenv("OOT_SLOW_FRAME_MS")?std::atof(std::getenv("OOT_SLOW_FRAME_MS")):40.0;
    const double st=upload_staging_ns.exchange(0)/1e6,sw=upload_swizzle_ns.exchange(0)/1e6,dc=upload_decode_ns.exchange(0)/1e6;
    static u64 last_wait=FenceWaitNanoseconds();const u64 wait_now=FenceWaitNanoseconds();const double fence=(wait_now-last_wait)/1e6;last_wait=wait_now;
    extern std::atomic<u64> executor_idle_ns,executor_parked_ns;const double idle=executor_idle_ns.exchange(0)/1e6,parked=executor_parked_ns.exchange(0)/1e6;
    extern std::atomic<u64> image_make_ns_frame;const double made=image_make_ns_frame.exchange(0)/1e6;
    if(ms>threshold)RT_LOG("slow frame %.0f ms: shader translation %.1f ms, pipeline waits %.1f ms (%llu draws skipped), texture uploads %.1f ms (%llu; staging %.1f, deswizzle %.1f, decode %.1f), image creation %.1f ms, fence waits %.1f ms, executor idle %.1f ms (a queue parked %.1f ms); pipeline lookups %.1f ms (shader modules %.1f ms)",
        ms,hitch.translate*1000,hitch.pipeline_wait*1000,(unsigned long long)pending,up,(unsigned long long)n,st,sw,dc,made,fence,idle,parked,hitch.lookup*1000,hitch.module*1000);
    hitch={};
}
u32 TakePrewarmHits(){return prewarm_hits.exchange(0);}
u64 ShaderCacheBytes(u64 key){
    char path[64];std::snprintf(path,sizeof path,"/%016llx.bin",(unsigned long long)key);
    std::error_code ec;const auto size=std::filesystem::file_size(ShaderCacheDirectory()+path,ec);return ec?0:(u64)size;
}
std::string ShaderCacheDir(){return ShaderCacheDirectory();}
u32 ShaderCacheVersionValue(){return ShaderCacheVersion;}
bool PrewarmBuild(const plist::Record& record){return PrewarmPipeline(record);}
void ReportCompileTime() {
    static u64 total_hits=0;if(const u32 hits=TakePrewarmHits()){total_hits+=hits;RT_LOG("prewarmed pipelines first used: %u since the last report (%llu this run)",hits,(unsigned long long)total_hits);}
    if(compile_time.created||compile_time.translate>0||compile_time.cached)RT_LOG("compile time: %.0f ms shader translation (%u variants from the disk cache), %.0f ms for %u graphics pipelines",
        compile_time.translate*1000,compile_time.cached,compile_time.pipelines*1000,compile_time.created);
    compile_time={};
}
void ReportDrawStats(u64 window_texture) {
    static const bool enabled = std::getenv("OOT_DRAW_STATS") != nullptr;
    if (!enabled) return;
    Texture* t = window_texture ? HostTexture(window_texture) : nullptr;
    RT_LOG("draw stats: %llu draws, %llu skipped (render disabled), %llu clears; window texture %#llx %ux%u written=%d",
           (unsigned long long)stats.draws, (unsigned long long)stats.skipped, (unsigned long long)stats.clears,
           (unsigned long long)window_texture, t ? t->width : 0, t ? t->height : 0, t ? (int)t->gpu_written : -1);
    stats = {};
}
// Queued uniform updates overlapping memory another operation reads or writes are written first.
void UniformUpdatesBefore(u64 cpu,u64 size){if(size)WriteUniformUpdatesOverlapping(cpu,size);}
void ReportUniformOverlays(){if(uniform_updates_deferred)RT_LOG("uniform updates: %.1f deferred, %.1f overlay copies per frame",uniform_updates_deferred/300.0,uniform_overlays_made/300.0);uniform_updates_deferred=uniform_overlays_made=0;}
} // namespace nvn
