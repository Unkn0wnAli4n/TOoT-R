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
#include <memory>
#include <string>
#include "../../native.h"
#include "occlusion_spv.h"
#include "indirect_clamp_spv.h"
#include "timestamp_scale_spv.h"
#include "commands.h"
#include "memory.h"
#include "program.h"
#include "texture.h"
#include "shader_compiler.h"
#include "batch.h"
using namespace rt;
namespace nvn {
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
    u32 sample_mask = ~0u;
    bool depth_bounds = false;
    float depth_bounds_range[2]{0, 1};
    u32 primitive = 4;
    float point_size = 1;
    // Root tables (NVN2 push constants): [0] graphics, [1] compute; 8 tables of 256 bytes.
    u32 root[2][8][64]{};
} state;
// GPU sample counters and predicates (occlusion.comp). Reports accumulate copied query results
// into a running total on the GPU and write it to guest memory; a conditional draw whose operands
// are still being produced gets its predicate computed on the GPU and runs under
// VK_EXT_conditional_rendering. The CPU never waits for occlusion results.
// Conditional draws by how their predicate was decided (since the last fps report).
struct PredicateStats { u64 cpu = 0, flushed = 0, bracketed = 0, dispatched = 0, processing = 0, processed_reports = 0, depth_tested = 0; } predicate_stats;
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
bool PredicateOperandsFinal(){
    // Reports and GPU writes earlier in the open batch can feed the predicate.
    static const bool sync_predicates=std::getenv("OOT_SYNC_PREDICATES")!=nullptr;
    // OOT_ASYNC_OCCLUSION=1: samples reports still in flight are not waited for; the predicate
    // uses the last completed result at that address (one or two frames old).
    static const bool async_occlusion=std::getenv("OOT_ASYNC_OCCLUSION")!=nullptr;
    if(HasPendingWrite(state.render_condition,24))return false;
    return !HasPendingReport(state.render_condition,24)||sync_predicates||async_occlusion;
}
// Values of reports whose value is known when recorded (counters other than samples passed
// report 0): predicates over them are decided on the CPU, even while the write is in flight.
std::unordered_map<u64,u64> known_reports;
bool EvaluatePredicateOnCpu(){
    const auto* operands=static_cast<const u8*>(ResolveAddress(state.render_condition,24));
    u64 initial,current;
    std::memcpy(&initial,operands,8);std::memcpy(&current,operands+16,8);
    if(const auto k=known_reports.find(state.render_condition);k!=known_reports.end())initial=k->second;
    if(const auto k=known_reports.find(state.render_condition+16);k!=known_reports.end())current=k->second;
    return (initial==current)==state.render_if_equal;
}
// Clears and other non-draw work: exact, waiting for the operands when needed.
bool RenderEnabled() {
    if (!state.render_enable) return false;
    static const bool ignore_conditions=std::getenv("OOT_NO_RENDER_CONDITION")!=nullptr; // (diagnostic)
    if (!state.render_condition||ignore_conditions) return true;
    if(!PredicateOperandsFinal())Flush();
    return EvaluatePredicateOnCpu();
}
// Draws: operands still in flight are evaluated on the GPU (sets draw_predicate).
bool DrawEnabled() {
    draw_predicate=~0u;draw_inverted=false;
    if (!state.render_enable) return false;
    if (!state.render_condition) return true;
    if(PredicateOperandsFinal()){++predicate_stats.cpu;return EvaluatePredicateOnCpu();}
    // A "draw if visible" draw that is depth tested (LESS/EQUAL/LEQUAL) changes no pixel when it is
    // in fact occluded: the depth test rejects all of it. Drawing it without waiting for the
    // occlusion result is therefore exact, and avoids a pipeline drain per tested object.
    static const bool exact_occlusion=std::getenv("OOT_EXACT_OCCLUSION")!=nullptr;
    const u32 depth_func=(state.depth>>4)&15;
    if(!exact_occlusion&&!state.render_if_equal&&(state.depth&1)&&state.targets.depth&&(depth_func==2||depth_func==3||depth_func==4)&&HasPendingReport(state.render_condition,24)){
        ++predicate_stats.depth_tested;return true;}
    if(!gpu_counters.ready||HasPendingWrite(state.render_condition,24)){++predicate_stats.flushed;Flush();return EvaluatePredicateOnCpu();}
    EnsureCounterRings();
    if(PredicateOperandsFinal())return EvaluatePredicateOnCpu(); // (a ring flush completed the reports)
    // The usual pattern: two adjacent reports bracket an occlusion test, and the draw depends on
    // whether samples passed between them, which processing stores in the second report's slot.
    const auto pair=gpu_counters.reports.find(state.render_condition+16);
    // Or (this game): the counter is reset before each occlusion test, and the report at +0 is
    // compared with a final zero at +16; samples since the reset are then the report's value.
    const auto single=gpu_counters.reports.find(state.render_condition);
    u64 second=1;
    if(single!=gpu_counters.reports.end()&&single->second.after_reset&&!HasPendingReport(state.render_condition+16,8)&&!HasPendingWrite(state.render_condition+16,8)){
        std::memcpy(&second,ResolveAddress(state.render_condition+16,8),8);
        if(const auto k=known_reports.find(state.render_condition+16);k!=known_reports.end())second=k->second;
    }
    const bool after_reset=single!=gpu_counters.reports.end()&&single->second.after_reset&&second==0;
    const bool bracketed=after_reset||(pair!=gpu_counters.reports.end()&&pair->second.follows_previous);
    if(!gpu_counters.pending.empty())ProcessReports();
    {static u32 shown2=0;if(std::getenv("OOT_PREDICATE_TRACE")&&shown2++<20){u64 a=0,b=0;const bool w=PendingWriteRange(state.render_condition,24,&a,&b);
        RT_LOG("predicate @%#llx: pending report %d, pending write %d (%#llx-%#llx), known %d/%d",(unsigned long long)state.render_condition,(int)HasPendingReport(state.render_condition,24),(int)w,(unsigned long long)a,(unsigned long long)b,
            (int)known_reports.count(state.render_condition),(int)known_reports.count(state.render_condition+16));}}
    {static u32 shown=0;if(std::getenv("OOT_PREDICATE_TRACE")&&shown++<30){
        const auto a=gpu_counters.reports.find(state.render_condition);
        RT_LOG("predicate @%#llx eq %d: report@+0 %s (slot %u follows %d), report@+16 %s (follows %d), last report %#llx",(unsigned long long)state.render_condition,(int)state.render_if_equal,
            a!=gpu_counters.reports.end()?"yes":"no",a!=gpu_counters.reports.end()?a->second.slot:0,a!=gpu_counters.reports.end()?(int)a->second.follows_previous:-1,
            pair!=gpu_counters.reports.end()?"yes":"no",pair!=gpu_counters.reports.end()?(int)pair->second.follows_previous:-1,(unsigned long long)gpu_counters.last_report);}}
    if(bracketed){++predicate_stats.bracketed;draw_predicate=after_reset?single->second.slot:pair->second.slot;draw_inverted=state.render_if_equal;}
    else{
        ++predicate_stats.dispatched;
        draw_predicate=gpu_counters.next_predicate++;
        CounterDispatch(2,draw_predicate,state.render_if_equal,state.render_condition,0);
    }
    if(gpu_counters.predicate_values){const u32 slot=draw_predicate;const bool inverted=draw_inverted;const u64 at=state.render_condition;const bool if_equal=state.render_if_equal;
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
    Immediate([&](VkCommandBuffer cmd){for(const auto& u:updates)vkCmdUpdateBuffer(cmd,u.buffer,u.offset,u.bytes.size(),u.bytes.data());});
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
u32 draw_base_vertex = 0, draw_base_instance = 0;
constexpr u32 StorageTableOffset(u32 hw) { return 0x110 + hw * 0x100 + (hw >= 5 ? 0x60 : 0); }
// The hardware stage and byte within its table of a bank-0 offset (false outside every table).
inline bool StorageTableEntry(u32 offset, u32& hw, u32& within) {
    for (hw = 0; hw < 7; ++hw) { const u32 base = StorageTableOffset(hw); if (offset >= base && offset < base + 0x100) { within = offset - base; return true; } }
    return false;
}
u32 frame_render_width=0; // widest compute render size this frame (dynamic resolution)
u32 dispatch_grid[3]{1, 1, 1}; // workgroup counts of the current dispatch (compute bank 0 +0x4..+0xc)
thread_local bool synthesizing = false; // SynthesizeDriverBank probes whole banks
u32 Cbuf(void* context, u32 bank, u32 offset) {
    auto& stage = *static_cast<StageState*>(context);
    static const bool trace = std::getenv("OOT_CBUF_TRACE") != nullptr;
    if (trace) RT_LOG("cbuf read bank %u offset %#x", bank, offset);
    // Compute bank 0 is packed per program (counts in control +0xa9/+0xb1/+0xb2/+0xb3/+0xb6): from
    // 0x10, the push constants (+0xa9 words, 16-byte aligned), A uniform-buffer entries and B storage-buffer entries of 16 bytes {address, size}, then C
    // texture handles and D image handles of 8 bytes, each for bindings 0.. in order.
    if (bank == 0 && &stage == &state.stages[5]) {
        const auto& control = stage.shader ? stage.shader->control : std::vector<u8>{};
        const auto count = [&](u32 at) -> u32 { return at < control.size() ? control[at] : 0; };
        const u32 a = count(0xb1), b = count(0xb2), t = count(0xb3), im = count(0xb6), push = count(0xa9) * 4;
        const auto entry = [&](const Buffer& buf, u32 word) -> u32 { return word == 0 ? (u32)buf.address : word == 1 ? (u32)(buf.address >> 32) : word == 2 ? (u32)buf.size : 0; };
        const auto half = [&](u64 v) { return (u32)(offset & 4 ? v >> 32 : v); };
        // 0x4/0x8/0xc: the dispatch's workgroup counts (shaders flatten ctaid with them).
        if (offset < 0x10) return offset == 0 ? 0 : dispatch_grid[offset / 4 - 1];
        u32 base = 0x10;
        // Push constants (root table 0, written by PushComputeConstants) come first: +0xa9 words.
        if (offset >= base && offset < base + push) return offset - base < 0x100 ? state.root[1][0][(offset - base) / 4] : 0;
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
            if(!synthesizing&&!reported[{at,offset}]){reported[{at,offset}]=true;RT_LOG("shader %#llx reads literal bank 1 at %#x beyond its %zu constant bytes (returned 0)",(unsigned long long)at,offset,stage.shader?stage.shader->constants.size():(size_t)0);}
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
        const bool compute = &stage == &state.stages[5];
        if (compute && table_hw != 0) return 0;
        auto& owner = compute ? stage : state.stages[HardwareToPublic[table_hw]];
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
        return state.root[&stage == &state.stages[5]][0][offset / 4];
    }
    if (bank < UniformBankBase) {
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
    if (bank == 0 && &stage == &state.stages[5]) return 0x400;
    if (bank == 0) return StorageTableEnd;
    if (bank == 2) return 0x220;
    return DriverBankSize;
}
void SynthesizeDriverBank(StageState& stage, u32 bank, u8* out) {
    std::memset(out, 0, DriverBankUsed(stage, bank));
    if (bank == 0 && &stage != &state.stages[5]) { // the shared SSBO tables, written directly
        constexpr u32 HardwareToPublic[] = {0, 3, 4, 2, 1, 6, 7};
        for (u32 hw = 0; hw < 7; ++hw) {
            const auto& owner = state.stages[HardwareToPublic[hw]];
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
    if(bank==RootBank){std::memcpy(out,state.root[&stage==&state.stages[5]][0],0x100);return;}
    if(bank==1){
        if(!stage.shader||stage.shader->constants.size()>DriverBankSize)Fatal("shader literal bank exceeds driver ring allocation");
        std::memcpy(out,stage.shader->constants.data(),stage.shader->constants.size());return;
    }
    u32 first = 0, last = DriverBankSize; // the only offsets synthesized below
    if (bank == 0 && &stage == &state.stages[5]) last = 0x400;
    else if (bank == 0) { first = StorageTableBase; last = StorageTableEnd; }
    else if (bank == 2) { first = 0x20; last = 0x220; }
    for (u32 offset = first; offset < last; offset += 4) {
        const bool known = (bank == 0 && &stage == &state.stages[5] && offset < 0x400) || (bank == 0 && &stage != &state.stages[5] && offset >= StorageTableBase && offset < StorageTableEnd) || (bank == 2 && offset >= 0x20 && offset < 0x220);
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
    }
    // Identical banks (most consecutive draws share them) reuse the ring copy written earlier in
    // this ring cycle; the ring then wraps (a full flush) far less often.
    alignas(16) static thread_local u8 bank_data[DriverBankSize];
    const u32 used = DriverBankUsed(stage, bank);
    SynthesizeDriverBank(stage, bank, bank_data);
    const bool compute_bank0 = bank == 0 && &stage == &state.stages[5]; // patched by indirect dispatch
    u64 hash = (0xcbf29ce484222325ull ^ bank) * 0x100000001b3ull ^ used;
    if (!compute_bank0) {
        const u64* words = reinterpret_cast<const u64*>(bank_data);
        for (u32 i = 0; i < used / 8; ++i) hash = (hash ^ words[i]) * 0x100000001b3ull;
        const auto found = ring_banks.find(hash);
        // (Compared against a CPU shadow: reading the host-visible ring itself is very slow.)
        if (found != ring_banks.end() && std::memcmp(ring_shadow.data() + found->second, bank_data, used) == 0)
            return {ring.buffer, found->second, used};
    }
    const u64 align = std::max<u64>(vk.properties.limits.minUniformBufferOffsetAlignment, 256);
    ring.head = (ring.head + align - 1) & ~(align - 1);
    if (ring.head + DriverBankSize > ring.size) Fatal("driver constant ring exhausted within one draw");
    std::memcpy(ring.mapped + ring.head, bank_data, used);
    if (compute_bank0) compute_bank0_offset = ring.head;
    else { if (ring_shadow.size() != ring.size) ring_shadow.resize(ring.size); std::memcpy(ring_shadow.data() + ring.head, bank_data, used); ring_banks[hash] = ring.head; }
    const VkDescriptorBufferInfo info{ring.buffer, ring.head, used};
    ring.head += used;
    return info;
}
// A ring copy of a uniform range with the pending updates applied (Buffer() when none overlaps).
VkDescriptorBufferInfo UniformOverlay(u64 cpu,u64 bytes){
    if(!UniformOverlayEnabled()||!UnwrittenUniformOverlap(cpu,bytes))return {};
    auto& vk=Vk();
    if(!ring.buffer||bytes>(1u<<20))return {};
    if(const auto c=uniform_overlay_cache.find({cpu,bytes});c!=uniform_overlay_cache.end())return {ring.buffer,c->second,bytes};
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
    return state.textures && id < state.textures->entries.size() && state.textures->entries[id].host;
}
bool SamplerRegistered(u32 handle) {
    const u32 id=handle>>20;
    return state.samplers&&id<state.samplers->entries.size()&&state.samplers->entries[id].valid;
}
TextureBinding& Binding(u32 handle) {
    const u32 id = handle & 0xfffff;
    if (!state.textures || id >= state.textures->entries.size() || !state.textures->entries[id].host) {
        for (u32 s = 0; s < 2; ++s) {
            RT_LOG("stage %u bound textures: %#llx %#llx %#llx %#llx", s, (unsigned long long)state.stages[s].textures[0],
                   (unsigned long long)state.stages[s].textures[1], (unsigned long long)state.stages[s].textures[2], (unsigned long long)state.stages[s].textures[3]);
            const auto& u = state.stages[s].uniforms[2];
            if (u.address) {
                const u32* w = static_cast<const u32*>(ResolveAddress(u.address, 0x40));
                RT_LOG("stage %u bank 2 words: %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x", s, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10], w[11]);
            }
        }
        Fatal("unregistered shader texture handle %#x", handle);
    }
    return state.textures->entries[id];
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
    if (!state.samplers || id >= state.samplers->entries.size() || !state.samplers->entries[id].valid) {
        // (the movie player samples its frames with sampler 0 without registering one: a linear clamp sampler,
        // as an unset sampler descriptor reads on the console)
        static VkSampler fallback = VK_NULL_HANDLE;
        if (!fallback) { VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; si.magFilter = si.minFilter = VK_FILTER_LINEAR;
            si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; si.maxLod = VK_LOD_CLAMP_NONE;
            VK_CHECK(vkCreateSampler(Vk().device, &si, nullptr, &fallback)); }
        static std::set<u32> told; if (told.insert(id).second && told.size() < 8) RT_LOG("sampler %u is not registered: a linear clamp sampler stands in", id);
        return fallback;
    }
    auto& e = state.samplers->entries[id]; if (e.sampler && e.settings_generation == gfx::Generation()) return e.sampler;
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
// (15:0) and task payload lines (31:24); +0x5c output topology (31:24, 3 = triangles); +0x60
// max vertices (31:16) and max primitives (15:0).
std::array<u32,3> MeshRegisters(const ShaderCode& code){
    const auto word=[&](u32 offset){u32 v=0;if(offset+4<=code.control.size())std::memcpy(&v,code.control.data()+offset,4);return v;};
    const u32 c58=word(0x58),c5c=word(0x5c),c60=word(0x60);
    const u32 threads=std::min(c58&0xffff,0x3fu),lines=c58>>24,topology=c5c>>24;
    // (Hookshot, mesh_registers_of: NVN's topology 3 (triangles) is the register's 4; points and lines go as they are.
    // Lines were a Fatal here: the shader compiler emits them, or the program is skipped as untranslatable.)
    if(code.stage==7&&topology!=3){static std::set<u64> told;if(told.insert(code.address).second)RT_LOG("mesh shader %#llx outputs topology %u (not triangles)",(unsigned long long)code.address,topology);}
    const u32 topology_register=topology==3?4u:(topology&0xfu);
    // Control +0x60: maximum vertices in the low half, maximum primitives in the high half (the
    // game's meshlets: 64 vertices / 126 triangles, grass 128 / 64). SET_MESH_SHADER_A holds
    // vertices at bits 4-12 and primitives at bits 13-22; swapping them clamped real meshlets
    // (rock depth passes drew garbage triangles: black shards).
    // (Hookshot, vk_shader.cpp mesh_registers_of: the pair is not always vertices and primitives in
    // that order - the Kokiri door's says 126/64 while its code writes 64 vertices and up to 128
    // primitives; held to 64, the rest of each meshlet was never output (holes, white door glow).
    // Both limits take the larger value. OOT_MESH_EXACT_LIMITS=1: the old split.)
    static const bool exact=std::getenv("OOT_MESH_EXACT_LIMITS")!=nullptr;
    const u32 most=std::min(std::max(c60&0xffff,c60>>16),0x100u);
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
    if(bank==0&&&stage==&state.stages[5]){
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
struct ShaderVariant { const ShaderCode* source; nvshader::Request request; nvshader::Result compiled; VkShaderModule module=VK_NULL_HANDLE; std::vector<std::array<u32,3>> texture_dependencies; };
struct CompileTime { double translate=0, pipelines=0; u32 created=0, cached=0; } compile_time; // seconds since the last report
// This frame's executor time in work that varies with new content (slow-frame diagnosis).
struct Hitch { double translate=0, pipeline_wait=0; } hitch;
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
constexpr u32 ShaderCacheVersion=7; // 2: subgroup operations in task/mesh programs; 3: NaN guard, fog scale (in the key); 4: guard clamp 65000; 5: NaN-safe min/max/saturate; 6: local memory starts at zero; 7: mesh stage buffer starts at zero
bool ShaderCacheEnabled(){static const bool on=std::getenv("OOT_NO_SHADER_CACHE")==nullptr;return on;}
std::string ShaderCacheDirectory(){
    static const std::string dir=[]{std::string d=rt::DataDir()+"/shader_cache";
        std::filesystem::create_directories(d);return d;}();
    return dir;
}
u64 ShaderCacheKey(const ShaderCode& code,const nvshader::Request& r){
    u64 h=0xcbf29ce484222325ull;const auto mix=[&](const void* p,size_t n){const u8* b=static_cast<const u8*>(p);for(size_t i=0;i<n;++i)h=(h^b[i])*0x100000001b3ull;};
    mix(&ShaderCacheVersion,4);{const char* fog=std::getenv("OOT_FOG_SCALE");const char* nan=std::getenv("OOT_NAN_GUARD");const char* lz=std::getenv("OOT_LOCAL_ZERO");const char* crs=std::getenv("OOT_LOCAL_CRS");const char* mz=std::getenv("OOT_MESH_NO_ZERO");const char* nc=std::getenv("OOT_MESH_NOCULL");const std::string k=std::string(code.stage==7?"u":"")+(fog?fog:"")+"/"+(nan?nan:"")+"/"+(lz?lz:"")+"/"+(crs?crs:"")+"/"+(mz?mz:"")+"/"+(nc?nc:"");mix(k.data(),k.size());}mix(code.code.data(),code.code.size());mix(code.control.data(),code.control.size());
    mix(&r.stage,4);mix(&r.binding_base,4);mix(r.input_types.data(),r.input_types.size());mix(r.color_types.data(),r.color_types.size());
    mix(&r.have_previous_stores,1);mix(r.previous_stores.data(),sizeof(u64)*r.previous_stores.size());mix(&r.topology,4);mix(&r.convert_depth,1);
    mix(&r.point_size,4);mix(r.mesh_registers.data(),4*r.mesh_registers.size());mix(&r.storage_alignment,8);mix(&r.buffer_float32_atomic_add,1);
    mix(&r.negate_y,1);mix(&r.force_early_depth,1);mix(&r.dual_source,1);
    return h;
}
struct ShaderCacheWriter{std::vector<u8> bytes;
    template<typename T> void put(const T& v){const u8* b=reinterpret_cast<const u8*>(&v);bytes.insert(bytes.end(),b,b+sizeof v);}
    template<typename T> void vec(const std::vector<T>& v){put((u32)v.size());const u8* b=reinterpret_cast<const u8*>(v.data());bytes.insert(bytes.end(),b,b+v.size()*sizeof(T));}};
struct ShaderCacheReader{const u8* p;const u8* end;bool ok=true;
    template<typename T> void get(T& v){if(p+sizeof v>end){ok=false;return;}std::memcpy(&v,p,sizeof v);p+=sizeof v;}
    template<typename T> void vec(std::vector<T>& v){u32 n=0;get(n);if(!ok||p+(u64)n*sizeof(T)>end){ok=false;return;}v.resize(n);std::memcpy(v.data(),p,(u64)n*sizeof(T));p+=(u64)n*sizeof(T);}};
void SerializeResult(ShaderCacheWriter& w,const nvshader::Result& r){
    w.put(u32{0x53434831});w.vec(r.spirv);w.vec(r.dependencies);w.put(r.stores);w.put(r.global_loads);w.put(r.global_stores);
    w.put((u32)r.resources.size());
    for(const auto& x:r.resources){w.put(x.kind);w.put(x.binding);w.put(x.count);w.put(x.bank);w.put(x.offset);w.put(x.secondary_bank);w.put(x.secondary_offset);
        w.put(x.shift);w.put(x.secondary_shift);w.put(x.size_shift);w.put(x.texture_type);w.put(x.image_format);w.put(x.indirect_offset);
        w.put(x.storage_index);w.put(x.storage_offset);w.put(x.storage_stride);w.put(x.sized_storage);w.put(x.secondary);w.put(x.depth);
        w.put(x.multisample);w.put(x.integer);w.put(x.written);w.vec(x.handle_program);}
}
bool DeserializeResult(ShaderCacheReader& rd,nvshader::Result& r){
    u32 magic=0;rd.get(magic);if(!rd.ok||magic!=0x53434831)return false;
    rd.vec(r.spirv);rd.vec(r.dependencies);rd.get(r.stores);rd.get(r.global_loads);rd.get(r.global_stores);
    u32 n=0;rd.get(n);if(!rd.ok||n>4096)return false;r.resources.resize(n);
    for(auto& x:r.resources){rd.get(x.kind);rd.get(x.binding);rd.get(x.count);rd.get(x.bank);rd.get(x.offset);rd.get(x.secondary_bank);rd.get(x.secondary_offset);
        rd.get(x.shift);rd.get(x.secondary_shift);rd.get(x.size_shift);rd.get(x.texture_type);rd.get(x.image_format);rd.get(x.indirect_offset);
        rd.get(x.storage_index);rd.get(x.storage_offset);rd.get(x.storage_stride);rd.get(x.sized_storage);rd.get(x.secondary);rd.get(x.depth);
        rd.get(x.multisample);rd.get(x.integer);rd.get(x.written);rd.vec(x.handle_program);}
    return rd.ok;
}
std::unordered_map<u64,std::vector<nvshader::Result>> shader_cache_entries; // key -> cached translations (loaded on first use)
std::unordered_set<u64> shader_cache_loaded;
const std::vector<nvshader::Result>& ShaderCacheLookup(u64 key){
    auto& list=shader_cache_entries[key];
    if(shader_cache_loaded.insert(key).second){
        char path[64];std::snprintf(path,sizeof path,"/%016llx.bin",(unsigned long long)key);
        if(FILE* f=std::fopen((ShaderCacheDirectory()+path).c_str(),"rb")){
            std::vector<u8> data;u8 buf[65536];size_t n;while((n=std::fread(buf,1,sizeof buf,f))>0)data.insert(data.end(),buf,buf+n);std::fclose(f);
            ShaderCacheReader rd{data.data(),data.data()+data.size()};
            while(rd.p<rd.end){nvshader::Result r;if(!DeserializeResult(rd,r))break;list.push_back(std::move(r));}
        }
    }
    return list;
}
// Programs reading global memory embed the address of this run's guest address table: they are translated with
// a placeholder there, cached as such, and get the real address just before their module is made
// (they were translated again every run: 100-900 ms of hitches).
constexpr u64 AddressTablePlaceholder=0x5eedf00dcafe0008ull;
void PatchAddressTable(std::vector<u32>& spirv){
    const u32 lo=(u32)AddressTablePlaceholder,hi=(u32)(AddressTablePlaceholder>>32);const u64 real=GlobalAddressTable(false);
    for(size_t i=0;i+1<spirv.size();++i)if(spirv[i]==lo&&spirv[i+1]==hi){spirv[i]=(u32)real;spirv[i+1]=(u32)(real>>32);}
}
void ShaderCacheStore(u64 key,const nvshader::Result& r){
    ShaderCacheWriter w;SerializeResult(w,r);
    char path[64];std::snprintf(path,sizeof path,"/%016llx.bin",(unsigned long long)key);
    if(FILE* f=std::fopen((ShaderCacheDirectory()+path).c_str(),"ab")){std::fwrite(w.bytes.data(),1,w.bytes.size(),f);std::fclose(f);}
    shader_cache_entries[key].push_back(r);
}
ShaderVariant& CompileShader(const ShaderCode& code, u32 binding_base, const ShaderVariant* previous) {
    auto& stage = state.stages[ShaderPublicStage[code.stage]];
    nvshader::Request request; request.stage=code.stage; request.binding_base=binding_base; request.code=&code.code; request.control=&code.control;
    request.context=&stage; request.read_constant=Cbuf; request.mapped_address=[](void*,u64 a){return LookupAddress(a).pool != nullptr;};
    request.read_pointer=ReadPointer;
    request.constant_pointer_available=ConstantPointerAvailable;
    request.texture_type=TextureType; request.texture_format=TextureFormat;
    request.classify_constant=ClassifyConstant; request.describe_texture=DescribeTexture;
    request.global_address_table=AddressTablePlaceholder; request.storage_alignment=Vk().properties.limits.minStorageBufferOffsetAlignment;
    request.buffer_float32_atomic_add=Vk().buffer_float32_atomic_add;
    // Input types describe vertex fetch, so they apply to the vertex stage only; later stages take
    // float inputs from the previous stage (gated by its store mask).
    request.input_types.fill(code.stage==0?5:0);
    for(u32 i=0;i<32;++i) if(code.stage==0&&(state.attributes[i]&0x40)) {
        request.input_types[i]=AttributeType(GetFormat(state.attributes[i]>>21&0x3ff).vk);
    }
    if(code.stage==1)for(u32 i=0;i<state.color_count;++i)if(state.targets.colors[i]) {
        auto* t=HostTexture(state.targets.colors[i]);
        request.color_types[i]=AttributeType(state.targets.color_views[i].format?
            GetFormat(state.targets.color_views[i].format).vk:t->info.vk);
    }
    if(previous) {request.have_previous_stores=true;request.previous_stores=previous->compiled.stores;}
    request.topology=state.primitive<=1?0:state.primitive<=3?1:3;
    // OpenGL-style clip depth (-w..w) is remapped to Vulkan's 0..w by the last pre-raster stage.
    const u32 last_pre_raster=code.stage>=6?7:state.stages[2].shader?2:state.stages[4].shader?4:0;
    if(code.stage==6||code.stage==7)request.mesh_registers=MeshRegisters(code);
    request.convert_depth=DepthNearIsMinusW()&&ShaderPublicStage[code.stage]==last_pre_raster;
    if(state.primitive==0&&ShaderPublicStage[code.stage]==last_pre_raster)request.point_size=std::max(state.point_size,1.f);
    // The variant that matched last time usually matches again: matches move to the front.
    auto& candidates=variants_by_code[&code];
    for (size_t index=0;index<candidates.size();++index) {
        auto* variant=candidates[index];
        const auto& q=variant->request;
        if(variant->source!=&code || q.binding_base!=binding_base || q.input_types!=request.input_types || q.color_types!=request.color_types || q.previous_stores!=request.previous_stores || q.topology!=request.topology || q.convert_depth!=request.convert_depth || q.point_size!=request.point_size) continue;
        bool match=true;static const bool why=std::getenv("OOT_VARIANT_TRACE")!=nullptr;
        for(const auto& d:variant->compiled.dependencies){
            const u32 now=CurrentDependencyValue(stage,d);
            if(now!=d.value){match=false;if(why)RT_LOG("  variant miss %#llx: kind %u cbuf %u+%#x was %#x now %#x",(unsigned long long)code.address,(u32)d.kind,d.bank,d.offset,d.value,now);break;}}
        if(match){if(index)std::swap(candidates[0],candidates[index]);return *variant;}
    }
    auto variant=std::make_unique<ShaderVariant>(); variant->source=&code; variant->request=request;
    // Content hash: stable across runs (UI programs load at different addresses each time).
    const u64 code_hash=[&]{u64 h=0xcbf29ce484222325ull;for(u8 b:code.code)h=(h^b)*0x100000001b3ull;return h;}();
    RT_LOG("compiling native shader stage %u at %#llx (variant %zu, code hash %016llx)",code.stage,(unsigned long long)code.address,shaders.size(),(unsigned long long)code_hash);
    if(const char* dir=std::getenv("OOT_SHADER_DUMP")){
        char path[512];std::snprintf(path,sizeof path,"%s/%llx-stage%u.bin",dir,(unsigned long long)code.address,code.stage);
        if(FILE* f=std::fopen(path,"wb")){std::fwrite(code.code.data(),1,code.code.size(),f);std::fclose(f);}
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
    if(variant->compiled.global_loads||variant->compiled.global_stores)PatchAddressTable(variant->compiled.spirv);
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; info.codeSize=variant->compiled.spirv.size()*4;info.pCode=variant->compiled.spirv.data();
    VK_CHECK(vkCreateShaderModule(Vk().device,&info,nullptr,&variant->module));
    if(std::getenv("OOT_SHADER_DUMP")) {char path[512];std::snprintf(path,sizeof path,"%s/%llx-v%zu.spv",std::getenv("OOT_SHADER_DUMP"),(unsigned long long)code.address,shaders.size());FILE* file=std::fopen(path,"wb");if(file){std::fwrite(info.pCode,1,info.codeSize,file);std::fclose(file);}}
    auto* result=variant.get(); variants_by_code[&code].push_back(result);shaders.push_back(std::move(variant));return *result;
}

// Background pipeline compilation.
class WorkerPool {
public:
    explicit WorkerPool(unsigned count){for(unsigned i=0;i<count;++i)threads.emplace_back([this]{Loop();});}
    void Run(std::function<void()> job){{std::lock_guard lock{mutex};jobs.push_back(std::move(job));}ready.notify_one();}
private:
    void Loop(){for(;;){std::function<void()> job;{std::unique_lock lock{mutex};ready.wait(lock,[this]{return !jobs.empty();});job=std::move(jobs.front());jobs.pop_front();}job();}}
    std::mutex mutex;std::condition_variable ready;std::deque<std::function<void()>> jobs;std::vector<std::thread> threads;
};
WorkerPool& PipelineWorkers(){static WorkerPool pool{std::max(2u,std::min(6u,std::thread::hardware_concurrency()/3))};return pool;}
std::mutex compile_time_mutex;
struct Pipeline {
    std::atomic<VkPipeline> pipeline{VK_NULL_HANDLE}; // null while a worker compiles it
    VkPipelineLayout layout=VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptors=VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool=VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set=VK_NULL_HANDLE;
    u64 descriptor_hash=0,descriptor_flushes=~0ull; // contents and batch of descriptor_set (reused while unchanged)
    bool push=false; // push-descriptor layout: draws push their descriptors (no set)
    std::vector<ShaderVariant*> shaders;
    std::vector<VkDescriptorSetLayoutBinding> bindings;
};
std::map<std::string,std::unique_ptr<Pipeline>> pipelines;

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
Pipeline& GetPipeline(bool mesh) {
    std::vector<ShaderVariant*> active;
    ShaderVariant* previous=nullptr;u32 binding=0;
    const std::initializer_list<u32> order=mesh?std::initializer_list<u32>{6u,7u,1u}:std::initializer_list<u32>{0u,3u,4u,2u,1u};
    for(u32 i:order) if(state.stages[i].shader) {
        auto& v=CompileShader(*state.stages[i].shader,binding,previous);previous=&v;active.push_back(&v);
        for(auto& r:v.compiled.resources)binding=std::max(binding,r.binding+(r.kind==nvshader::ResourceKind::UniformBuffer||r.kind==nvshader::ResourceKind::StorageBuffer?r.count:1));
    }
    static Pipeline untranslatable;
    for(auto* v:active)if(!v->module)return untranslatable;
    if(!mesh&&!state.stages[0].shader)Fatal("draw has no vertex shader");
    if(mesh&&!state.stages[7].shader)Fatal("mesh draw has no mesh shader");
    std::string key;
    const auto add=[&](const auto& value){key.append(reinterpret_cast<const char*>(&value),sizeof value);};
    // Only the state bits pipeline creation reads below: the raw NVN words also carry unrelated
    // (often uninitialized) bits, which would make every draw look like a new pipeline.
    for(u32 i=0;i<32;++i){const u32 a=state.attributes[i];add(a&0x40?a&(31u|0x40u|0x3fffu<<7|0x3ffu<<21):0u);}
    {std::array<bool,16> used{};for(u32 i=0;i<32;++i)if(state.attributes[i]&0x40)used[state.attributes[i]&31]=true;
     for(u32 i=0;i<16;++i){add(used[i]?state.streams[i][0]:0u);add(used[i]?state.streams[i][1]:0u);}}
    add(state.color_count);
    for(u32 i=0;i<state.color_count;++i){const bool blended=(state.color>>(8+i))&1;add(blended);add(blended?state.blends[i]>>24:0ull);add(state.channel>>(i*4)&15);}
    add(state.polygon&(0x1fu|0xfu<<15));add(WindowOriginLowerLeft());
    add(state.depth&0xf7);add((state.depth>>2)&1?state.depth>>32:0ull);
    add(state.depth_bounds);if(state.depth_bounds)add(state.depth_bounds_range);
    add(state.primitive);add(state.restart);add(state.depth_clamp);add(state.discard);add(state.sample_mask);add(mesh);
    std::array<VkFormat,8> colors{};VkFormat depth=VK_FORMAT_UNDEFINED,stencil=VK_FORMAT_UNDEFINED;u32 samples=1;
    for(u32 i=0;i<state.color_count;++i)if(state.targets.colors[i]){auto* t=HostTexture(state.targets.colors[i]);colors[i]=state.targets.color_views[i].format?GetFormat(state.targets.color_views[i].format).vk:t->info.vk;samples=t->samples;}
    if(state.targets.depth){auto* t=HostTexture(state.targets.depth);if(t->info.depth)depth=t->info.vk;if(t->info.stencil)stencil=t->info.vk;samples=t->samples;}
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
    auto snapshot=std::make_shared<State>(state);
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
        VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};pi.pNext=&ri;pi.stageCount=stages.size();pi.pStages=stages.data();pi.pVertexInputState=mesh?nullptr:&vi;pi.pInputAssemblyState=mesh?nullptr:&ia;pi.pViewportState=&vp;pi.pRasterizationState=&rs;pi.pMultisampleState=&ms;pi.pDepthStencilState=&ds;pi.pColorBlendState=&cb;pi.pDynamicState=&dy;pi.layout=target->layout;
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
        PipelineWorkers().Run(std::move(job));
        const auto waited_from=std::chrono::steady_clock::now();
        const auto deadline=waited_from+std::chrono::microseconds(3000);
        while(!target->pipeline.load(std::memory_order_acquire)&&std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
        hitch.pipeline_wait+=std::chrono::duration<double>(std::chrono::steady_clock::now()-waited_from).count();
    }
    }
    auto* result=pipeline.get();pipelines.emplace(std::move(key),std::move(pipeline));return *result;
}
u64 ShaderStorageAddress(const ShaderVariant& shader,u32 index) {
    auto& stage=state.stages[ShaderPublicStage[shader.source->stage]];
    for(const auto& resource:shader.compiled.resources)if(resource.kind==nvshader::ResourceKind::StorageBuffer) {
        if(index--!=0)continue;
        u64 address=Cbuf(&stage,resource.bank,resource.offset)|(u64)Cbuf(&stage,resource.bank,resource.offset+4)<<32;
        if(resource.indirect_offset!=~0u)address=ReadPointer(nullptr,address+resource.indirect_offset);
        return address;
    }
    Fatal("texture handle expression uses an unknown storage descriptor");
}
u32 ResourceHandle(const nvshader::Resource& r,ShaderVariant& shader,u32 element) {
    auto& stage=state.stages[ShaderPublicStage[shader.source->stage]];
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
            })+element;
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
    for(u32 i=0;i<state.color_count;++i)if(state.targets.colors[i])DumpRaw(*HostTexture(state.targets.colors[i]),n,dir);
    for(u32 i=0;i<state.color_count;++i)if(state.targets.colors[i]){char name[96];std::snprintf(name,sizeof name,"/%03u-%s-c%u-%llx.ppm",n,what,i,(unsigned long long)state.targets.colors[i]);DumpTexture(*HostTexture(state.targets.colors[i]),dir+std::string(name));}
    ++n;
}
void LogDraw(const Command& command) {
    if(state.stages[6].shader){auto& st=state.stages[6];const auto word=[&](u32 u,u32 off)->long long{const auto& b=st.uniforms[u];return b.address&&off+4<=b.size?(long long)ReadWord(b.address+off):-1;};
        RT_LOG("  task constants: c3[0x420]=%lld c3[0x428]=%lld c4[0x5c]=%#llx (c3 @%#llx c4 @%#llx)",word(0,0x420),word(0,0x428),word(1,0x5c),(unsigned long long)st.uniforms[0].address,(unsigned long long)st.uniforms[1].address);}
    const auto* x=command.x;
    for(u32 st=0;st<2;++st)for(u32 i=0;i<16;++i){const auto& b=state.stages[st].storage[i];if(!b.address)continue;const auto range=LookupAddress(b.address);
        std::string words;if(range.pool){const float* f=static_cast<const float*>(ResolveAddress(b.address,0x40));for(u32 k=0;k<16;++k){char t[24];std::snprintf(t,sizeof t," %g",f[k]);words+=t;}}
        RT_LOG("  stage %u storage %u @%#llx size %#llx mapped %d:%s",st,i,(unsigned long long)b.address,(unsigned long long)b.size,range.pool!=nullptr,words.c_str());}
    for(u32 st=0;st<2;++st)for(u32 i=0;i<8;++i){const auto& u=state.stages[st].uniforms[i];if(!u.address)continue;const auto range=LookupAddress(u.address);
        std::string words;if(range.pool){const float* f=static_cast<const float*>(ResolveAddress(u.address,std::min<u64>(u.size,0x60)));for(u32 k=0;k<std::min<u64>(u.size,0x60)/4;++k){char b[24];std::snprintf(b,sizeof b," %g",f[k]);words+=b;}}
        RT_LOG("  stage %u uniform %u @%#llx size %#llx:%s",st,i,(unsigned long long)u.address,(unsigned long long)u.size,words.c_str());
        if(range.pool&&u.size<=0x100){const u64 a=u.address,n=std::min<u64>(u.size,0x40);bool zero=true;const u8* p=static_cast<const u8*>(ResolveAddress(a,n));for(u64 k=0;k<n;++k)zero&=p[k]==0;
            if(zero)AfterFlush([a,n,st,i,pool_flags=range.pool->flags,mirrored=range.pool->device_buffer!=VK_NULL_HANDLE]{const float* f=static_cast<const float*>(ResolveAddress(a,n));
                RT_LOG("  (stage %u uniform %u @%#llx pool flags %#x%s, after its batch: %g %g %g %g %g %g %g %g)",st,i,(unsigned long long)a,pool_flags,mirrored?" device copy":"",f[0],f[1],f[2],f[3],f[4],f[5],f[6],f[7]);});}}
    RT_LOG("  device modes: origin lower-left %d, depth near -w %d",WindowOriginLowerLeft(),DepthNearIsMinusW());
    RT_LOG("  raster: discard %d sample_mask %#x depth_clamp %d depth_bounds %d restart %d samples-target; depth bias factor %g units %g clamp %g",state.discard,state.sample_mask,state.depth_clamp,state.depth_bounds,state.restart,state.depth_bias[0],state.depth_bias[1],state.depth_bias[2]);
    for(u32 i=0;i<32;++i){const u32 a=state.attributes[i];if(a&0x40)RT_LOG("  attribute %u: raw %#x stream %u offset %u format %u",i,a,a&31,(a>>7)&0x3fff,a>>21&0x3ff);}
    for(u32 i=0;i<2;++i){const auto& v=state.vertices[i];if(!v.address)continue;const float* f=static_cast<const float*>(ResolveAddress(v.address,48));
        RT_LOG("  vertex stream %u @%#llx stride %u: %g %g %g %g %g %g",i,(unsigned long long)v.address,state.streams[i][0],f[0],f[1],f[2],f[3],f[4],f[5]);
        const u8* raw=static_cast<const u8*>(ResolveAddress(v.address,96));std::string hex;for(u32 k=0;k<96;++k){char b[4];std::snprintf(b,sizeof b,"%02x",raw[k]);hex+=b;if(k%16==15)hex+=' ';}
        RT_LOG("  vertex bytes: %s",hex.c_str());}
    std::string shaders;for(u32 i=0;i<std::size(state.stages);++i)if(i!=5&&state.stages[i].shader){char b[48];std::snprintf(b,sizeof b," s%u=%#llx",i,(unsigned long long)state.stages[i].shader->address);shaders+=b;}
    std::string targets;for(u32 i=0;i<state.color_count;++i)if(state.targets.colors[i]){auto* t=HostTexture(state.targets.colors[i]);char b[64];std::snprintf(b,sizeof b," c%u=%#llx(%ux%u f%u)",i,(unsigned long long)state.targets.colors[i],t->width,t->height,t->format);targets+=b;}
    RT_LOG("draw op %u args %llu %llu %llu %llu %llu %llu %llu |%s |%s depth=%#llx vp(%g %g %g %g) sc(%d %d %u %u) blend0=%#llx color=%#x chan=%#x poly=%#x ds=%#llx",
        command.op,(unsigned long long)x[0],(unsigned long long)x[1],(unsigned long long)x[2],(unsigned long long)x[3],(unsigned long long)x[4],(unsigned long long)x[5],(unsigned long long)x[6],
        shaders.c_str(),targets.c_str(),(unsigned long long)state.targets.depth,state.viewport.x,state.viewport.y,state.viewport.width,state.viewport.height,
        state.scissor.offset.x,state.scissor.offset.y,state.scissor.extent.width,state.scissor.extent.height,(unsigned long long)state.blends[0],state.color,state.channel,state.polygon,(unsigned long long)state.depth);
}
// A 64 KiB zero-filled buffer bound in place of unbound uniform/storage slots.
VkDescriptorBufferInfo ZeroBuffer(){
    static VkBuffer buffer=VK_NULL_HANDLE;
    if(!buffer){auto& vk=Vk();
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=0x10000;bi.usage=VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT|VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        VK_CHECK(TracedCreateBuffer(&bi,&buffer,__FILE__,__LINE__));VkMemoryRequirements req;vkGetBufferMemoryRequirements(vk.device,buffer,&req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=req.size;ai.memoryTypeIndex=FindMemoryType(req.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory memory;VK_CHECK(vkAllocateMemory(vk.device,&ai,nullptr,&memory));VK_CHECK(vkBindBufferMemory(vk.device,buffer,memory,0));
        void* p;VK_CHECK(vkMapMemory(vk.device,memory,0,0x10000,0,&p));std::memset(p,0,0x10000);}
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
        auto& stage=state.stages[ShaderPublicStage[shader->source->stage]];
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
        struct Probe { StageState* stage; const ShaderVariant* shader; bool pending=false; } probe{&stage,shader};
        for(const auto& r:shader->compiled.resources){
            if(r.handle_program.empty())continue;
            for(u32 element=0;element<std::max(r.count,1u)&&!probe.pending;++element)
                nvshader::EvaluateHandle(r.handle_program,&probe,
                    [](void* p,u32 bank,u32 offset){auto& q=*static_cast<Probe*>(p);
                        if(bank>=UniformBankBase&&bank-UniformBankBase<q.stage->uniforms.size()){const auto& u=q.stage->uniforms[bank-UniformBankBase];
                            if(u.address&&HasPendingWrite(u.address+offset,4))q.pending=true;}
                        return Cbuf(q.stage,bank,offset);},
                    [](void* p,u32 index,u32 offset){auto& q=*static_cast<Probe*>(p);
                        const u64 alignment=Vk().properties.limits.minStorageBufferOffsetAlignment;
                        const u64 address=(ShaderStorageAddress(*q.shader,index)&~(alignment-1))+offset;
                        if(HasPendingWrite(address,4))q.pending=true;
                        return ReadWord(address);});
            if(probe.pending){ReportHandleWait("handle program",shader->source->address,0,4);return true;}
        }
    }
    return false;
}
// Resources of the draw/dispatch whose descriptors were written last: textures it reads and
// whether it writes storage (buffers or images). Draw uses them to keep a render pass open.
std::vector<Texture*> descriptor_reads;
bool descriptor_writes=false;
// The descriptor writes of the last WriteDescriptors (pushed by a push-descriptor pipeline's draw).
thread_local std::vector<VkWriteDescriptorSet> descriptor_write_list;
// Writes of descriptor_write_list that come from a uniform binding (the draw fast path patches
// them), and whether the last resolution depends only on bindings the fast path tracks.
struct UniformSlot { u32 write; StageState* stage; u32 bank; };
thread_local std::vector<UniformSlot> uniform_write_slots;
thread_local bool descriptor_fast_eligible = false;
// Guest ranges [start, end) of the buffers bound by the last WriteDescriptors, and the images it
// writes: compute dispatches use them to skip barriers between independent dispatches.
std::vector<std::pair<u64,u64>> descriptor_buffer_reads,descriptor_buffer_writes;
std::vector<Texture*> descriptor_image_writes;
// Fills the pipeline's (reused) descriptor set from the current state.
bool WriteDescriptors(Pipeline& pipeline){
    auto& vk=Vk();static const bool gpu_trace=std::getenv("OOT_GPU_TRACE")!=nullptr;
    // Texture handles and buffer pointers read from guest memory on the CPU must see GPU writes
    // recorded earlier in the open batch: submit first only when such a write is pending.
    if(HandleMemoryPending(pipeline))Flush();
    descriptor_reads.clear();descriptor_writes=false;
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
    if(ring.buffer&&ring.head+32*DriverBankSize+(4u<<20)>ring.size){Flush();ring.head=0;ring_banks.clear();uniform_overlay_cache.clear();}
    // Uploads below may flush, retiring this batch's descriptor pools: then write a fresh set.
    for(u64 flushes=~0ull;flushes!=FlushCount();){
    flushes=FlushCount();
    descriptor_reads.clear();descriptor_writes=false;
    descriptor_buffer_reads.clear();descriptor_buffer_writes.clear();descriptor_image_writes.clear();
    uniform_write_slots.clear();descriptor_fast_eligible=true;
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
    for(auto* shader:pipeline.shaders){auto& stage=state.stages[ShaderPublicStage[shader->source->stage]];
        for(const auto& r:shader->compiled.resources){
            std::vector<VkDescriptorBufferInfo> buffers;std::vector<VkDescriptorImageInfo> images;
            // (Fast path: handles or pointers read through memory, and root-bank uniforms, are not tracked.)
            if(!r.handle_program.empty()||r.storage_index!=~0u||r.indirect_offset!=~0u||r.bank==RootBank||
               (r.kind==nvshader::ResourceKind::StorageBuffer&&r.bank>=UniformBankBase))descriptor_fast_eligible=false;
            const bool uniform_slot=r.kind==nvshader::ResourceKind::UniformBuffer&&r.bank>=UniformBankBase&&r.bank!=RootBank;
                if(r.kind==nvshader::ResourceKind::UniformBuffer&&(r.bank<UniformBankBase||(r.bank==RootBank&&!stage.uniforms[RootBank-UniformBankBase].address))){buffers.push_back(DriverBankDescriptor(stage,r.bank));}
                else if(r.kind==nvshader::ResourceKind::UniformBuffer||r.kind==nvshader::ResourceKind::StorageBuffer){
                Buffer b;
                if(r.kind==nvshader::ResourceKind::UniformBuffer){b=stage.uniforms.at(r.bank-UniformBankBase);}
                else {b.address=Cbuf(&stage,r.bank,r.offset)|(u64)Cbuf(&stage,r.bank,r.offset+4)<<32;
                    if(r.indirect_offset!=~0u)b.address=ReadPointer(nullptr,b.address+r.indirect_offset);
                    if(r.sized_storage)b.size=Cbuf(&stage,r.bank,r.offset+8);
                    static const bool exact=std::getenv("OOT_EXACT_STORAGE_SIZE")==nullptr?false:true;
                    if(!exact)b.size=0;} // (bound to the end of its memory: diagnostic default, see below)
                auto range=GpuLookupAddress(b.address);
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
                }else if(UniformOverlayEnabled())WriteUniformUpdatesOverlapping(range.cpu_address,extent); // (read as storage: memory must have them)
                buffers.push_back({range.buffer,range.offset-delta,std::min(b.size,range.available)+delta});
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
                        static VkSampler null_sampler=VK_NULL_HANDLE;
                        if(!null_sampler){VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};si.maxLod=VK_LOD_CLAMP_NONE;VK_CHECK(vkCreateSampler(vk.device,&si,nullptr,&null_sampler));}
                        images.push_back({r.kind==nvshader::ResourceKind::Texture?null_sampler:VK_NULL_HANDLE,VK_NULL_HANDLE,VK_IMAGE_LAYOUT_GENERAL});
                        continue;
                    }
                    if(!HandleRegistered(handle)){
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
                    descriptor_reads.push_back(&t);if(r.kind==nvshader::ResourceKind::Image&&r.written){descriptor_writes=true;descriptor_image_writes.push_back(&t);}
                    if(LogThisFrame()&&t.format==1&&t.storage&&t.storage_size){std::vector<u8> bytes(t.storage_size);std::memcpy(bytes.data(),ResolveAddress(t.storage,bytes.size()),bytes.size());u64 nz=0;for(u8 b:bytes)nz+=b!=0;RT_LOG("  (R8 texture %#llx: %llu/%zu nonzero guest bytes, uploaded frame %llu, upload seq %u)",(unsigned long long)binding.texture,(unsigned long long)nz,bytes.size(),(unsigned long long)t.upload_frame,t.upload_sequence);}
                    if(LogThisFrame()){const u32* sw=t.swizzle;RT_LOG("  stage %u texture resource bank %u offset %#x element %u: handle %#x -> texture %#llx %ux%u fmt %u written %d storage %#llx swizzle %u %u %u %u view %d (%u %u %u %u)",shader->source->stage,r.bank,r.offset,element,handle,(unsigned long long)binding.texture,t.width,t.height,t.format,(int)t.gpu_written,(unsigned long long)t.storage,sw[0],sw[1],sw[2],sw[3],binding.view.present,binding.view.swizzle[0],binding.view.swizzle[1],binding.view.swizzle[2],binding.view.swizzle[3]);}
                    if(r.kind==nvshader::ResourceKind::Image){static std::map<std::pair<u64,u32>,bool> seen;if(!seen[{shader->source->address,t.format}]){seen[{shader->source->address,t.format}]=true;
                        RT_LOG("image binding: shader %#llx bank %u+%#x texture fmt %u %ux%u, shader image format %u written %d",(unsigned long long)shader->source->address,r.bank,r.offset,t.format,t.width,t.height,r.image_format,(int)r.written);}}
                    TextureView view=binding.view;view.target=ShaderViewTarget(r.texture_type,t)+1;
                    if(r.kind==nvshader::ResourceKind::Image){view.swizzle[0]=2;view.swizzle[1]=3;view.swizzle[2]=4;view.swizzle[3]=5;} // storage views: identity
                    images.push_back({r.kind==nvshader::ResourceKind::Texture?Sampler(handle):VK_NULL_HANDLE,ImageView(t,view),VK_IMAGE_LAYOUT_GENERAL});}
                if(r.storage_index!=~0u){static std::map<std::pair<u64,u32>,bool> seen;
                    if(!seen[{shader->source->address,r.binding}]){seen[{shader->source->address,r.binding}]=true;
                        RT_LOG("storage texture array shader %#llx binding %u: %u/%u valid, storage %u+%#x stride %u",(unsigned long long)shader->source->address,r.binding,array_valid,r.count,r.storage_index,r.storage_offset,r.storage_stride);}}
            }else Fatal("texel buffer descriptor pending");
            queue_write(r,std::move(buffers),std::move(images));
            if(uniform_slot)uniform_write_slots.push_back({(u32)writes.size()-1,&stage,r.bank});
        }
    }
    // One update for the whole set (unless uploads above flushed: then the loop writes a new set).
    if(flushes!=FlushCount())continue; // (uploads above flushed: resolve again for the new batch)
    if(pipeline.push){pipeline.descriptor_set=VK_NULL_HANDLE;break;} // (the draw pushes descriptor_write_list)
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
// Indirect dispatch arguments, validated on the GPU (indirect_clamp.comp) into a slot of a
// device buffer: dispatched from there, and copied to the compute driver bank's grid words.
struct IndirectClamp{VkPipeline pipeline=VK_NULL_HANDLE;VkPipelineLayout layout=VK_NULL_HANDLE;VkBuffer buffer=VK_NULL_HANDLE;VkDeviceMemory memory=VK_NULL_HANDLE;
    VkDeviceAddress address=0;u32 next=0;static constexpr u32 Slots=4096;} indirect_clamp;
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
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT,0,16};
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
    for(u32 i=0;i<state.color_count&&!key;++i)if(state.targets.colors[i])if(auto* t=HostTexture(state.targets.colors[i]))key=t->storage;
    if(!key&&state.targets.depth)if(auto* t=HostTexture(state.targets.depth))key=t->storage;
    if(!key)return false;
    const bool recurring=RecurringWork(key);
    if(frame_skipping&&recurring){++frame_skip_dropped;return true;}
    return false;
}
u64 mesh_indirect_slot=0; // nonzero while Draw records a DrawMeshTasksIndirect (slot + 1)
struct FastDraw {
    bool valid=false; Pipeline* pipeline=nullptr; std::array<Texture*,8> colors{}; Texture* depth=nullptr; VkExtent2D extent{};
    std::array<VkRenderingAttachmentInfo,8> attachments{}; VkRenderingAttachmentInfo da{}; std::vector<u64> key; u64 flushes=0, pools=0; u32 texture_writes=0;
} fast_draw;
u64 fast_draw_count=0;
bool TryFastDraw(FastDraw& fd){
    if(FlushCount()!=fd.flushes){++fast_draw_fails[0];return false;}
    if(PoolGeneration()!=fd.pools||TextureWriteSequence()!=fd.texture_writes){++fast_draw_fails[1];return false;} // (texture writes: re-upload via the full path)
    if(!PassOpenWithKey(fd.key)){++fast_draw_fails[2];return false;}
    // The variants still apply (their specialization constants may live in the rebound buffers).
    for(auto* shader:fd.pipeline->shaders){auto& stage=state.stages[ShaderPublicStage[shader->source->stage]];
        if(stage.shader!=shader->source){++fast_draw_fails[3];return false;}
        for(const auto& d:shader->compiled.dependencies)if(CurrentDependencyValue(stage,d)!=d.value){++fast_draw_fails[4];return false;}}
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
// OOT_SKIP_SHADER_FILE=<path> (diagnostic): draws and dispatches using the programs listed there (hex
// addresses) are dropped; the file is re-read every 30 frames (bisecting an artifact while running).
bool SkippedByFile(u64 address){
    static const char* skip_file=std::getenv("OOT_SKIP_SHADER_FILE");
    if(!skip_file)return false;
    static std::vector<u64> live;static u64 read_at=~0ull;
    if(frame_number/30!=read_at){read_at=frame_number/30;live.clear();
        if(FILE* f=std::fopen(skip_file,"r")){char word[64];while(std::fscanf(f,"%63s",word)==1)live.push_back(std::strtoull(word,nullptr,16));std::fclose(f);}}
    return !live.empty()&&std::find(live.begin(),live.end(),address)!=live.end();
}
void Draw(const Command& command) {
    // OOT_SKIP_SHADER=<hex address>[,...]: drop draws using these programs (finding which draws
    // produce an artifact).
    static const std::vector<u64> skipped_shaders=[]{std::vector<u64> v;if(const char* e=std::getenv("OOT_SKIP_SHADER"))for(const char* p=e;*p;){char* end;v.push_back(std::strtoull(p,&end,16));p=*end?end+1:end;}return v;}();
    if(!skipped_shaders.empty())for(const auto& st:state.stages)if(st.shader&&std::find(skipped_shaders.begin(),skipped_shaders.end(),st.shader->address)!=skipped_shaders.end())return;
    for(const auto& st:state.stages)if(st.shader&&SkippedByFile(st.shader->address))return;
    if(SkipDraw())return;
    static const bool gpu_trace=std::getenv("OOT_GPU_TRACE")!=nullptr;
    if(gpu_trace){
        std::string stages;
        for(u32 i=0;i<8;++i)if(state.stages[i].shader){char word[64];std::snprintf(word,sizeof word," s%u=%llx",i,(unsigned long long)state.stages[i].shader->address);stages+=word;}
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
    const u64 indirect_slot=mesh?mesh_indirect_slot:0; // (DrawMeshTasksIndirect: arguments in indirect_clamp's slot - 1)
    // Fast path: draws that differ from the previous one only in uniform/vertex buffer bindings
    // (most of the game's per-object draws) reuse its pipeline, attachments and descriptor writes,
    // patching only the uniform buffer descriptors. OOT_NO_FAST_DRAW=1 disables it.
    static const bool fast_disabled=std::getenv("OOT_NO_FAST_DRAW")!=nullptr;
    auto& fd=fast_draw;
    if(gpu_counters.ready&&samples.enabled)EnsureCounterRings(); // (may flush: then the fast path does not apply)
    {   // (bank 0 draw parameters, see DrawBaseVertexOffset)
        const Op op=(Op)command.op;const auto& a=command.x;
        const u32 base_vertex=op==Op::DrawElementsBaseVertex||op==Op::DrawElementsInstanced?(u32)a[4]:0;
        const u32 base_instance=op==Op::DrawElementsInstanced?(u32)a[5]:op==Op::DrawArraysInstanced?(u32)a[3]:0;
        if(base_vertex!=draw_base_vertex||base_instance!=draw_base_instance){draw_base_vertex=base_vertex;draw_base_instance=base_instance;fast_draw_allowed=false;}
    }
    const bool fast=!fast_disabled&&!mesh&&!stats_pool&&fast_draw_allowed&&fd.valid&&draw_predicate==~0u&&TryFastDraw(fd);
    if(fast)++fast_draw_count;
    Pipeline* const pipeline_ptr=fast?fd.pipeline:&GetPipeline(mesh);
    auto& pipeline=*pipeline_ptr;auto& vk=Vk();
    std::array<Texture*,8> colors{};Texture* depth=nullptr;VkExtent2D extent{0,0};
    std::array<VkRenderingAttachmentInfo,8> attachments{};VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    std::vector<u64> key;
    if(fast){colors=fd.colors;depth=fd.depth;extent=fd.extent;attachments=fd.attachments;da=fd.da;key=fd.key;}
    else{
        fd.valid=false; // (this draw rewrites the descriptor lists the saved setup refers to)
    // Shaders with native global memory accesses read the guest address table: bring it up to
        // date with the pools (rewriting it waits for earlier work, so only when they need it).
        for(const auto* shader:pipeline.shaders)if(shader->compiled.global_loads||shader->compiled.global_stores){GlobalAddressTable();break;}
        if(!pipeline.pipeline.load(std::memory_order_acquire)){++stats.pending;return;} // untranslatable, or still compiling
        // Anything that may flush runs before the descriptor set is written (below): a flush retires
        // the batch and resets the descriptor pool that set comes from.
        if(gpu_counters.ready&&samples.enabled)EnsureCounterRings();
        if(!gpu_counters.ready&&samples.enabled&&samples.next==SampleCounter::Capacity){Flush();samples.next=0;}
        depth=state.targets.depth?HostTexture(state.targets.depth):nullptr;
        for(u32 i=0;i<state.color_count;++i)if(state.targets.colors[i]){colors[i]=HostTexture(state.targets.colors[i]);extent={std::max(1u,colors[i]->width>>state.targets.color_views[i].levels_base),std::max(1u,colors[i]->height>>state.targets.color_views[i].levels_base)};}
        if(!extent.width&&depth)extent={depth->width,depth->height};if(!extent.width){
            // Attachment-less draws (storage writes, queries) rasterize over the viewport area.
            extent={(u32)std::max(1.f,std::fabs(state.viewport.x)+std::fabs(state.viewport.width)),(u32)std::max(1.f,std::fabs(state.viewport.y)+std::fabs(state.viewport.height))};}
        for(u32 i=0;i<state.color_count;++i){auto& a=attachments[i];a.sType=VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;a.imageLayout=VK_IMAGE_LAYOUT_GENERAL;a.loadOp=VK_ATTACHMENT_LOAD_OP_LOAD;a.storeOp=VK_ATTACHMENT_STORE_OP_STORE;if(colors[i])a.imageView=AttachmentView(*colors[i],state.targets.color_views[i]);}
        da.imageLayout=VK_IMAGE_LAYOUT_GENERAL;da.loadOp=VK_ATTACHMENT_LOAD_OP_LOAD;da.storeOp=VK_ATTACHMENT_STORE_OP_STORE;if(depth)da.imageView=AttachmentView(*depth,state.targets.depth_view);
        if(!WriteDescriptors(pipeline))return;
        // Allocated only for a draw that is recorded: every slot in the counter range must be reset
        // and written, or GPU accumulation would sum stale results of earlier frames.
        // With GPU counters, occlusion samples are counted per render pass segment (see SplitSegment);
        // the per-draw queries are the CPU fallback.
        if(gpu_trace)RT_LOG("GPU submit");
        // The pass key: everything vkCmdBeginRendering depends on.
        key={extent.width|(u64)extent.height<<32,state.color_count,(u64)da.imageView,depth?(u64)depth->info.depth|(u64)depth->info.stencil<<1:0};
        for(u32 i=0;i<state.color_count;++i)key.push_back((u64)attachments[i].imageView);
        // Sampling an attachment of the open pass (feedback) needs the pass ended and a barrier.
        for(auto* t:descriptor_reads){if(t==depth){EndPass();break;}bool hit=false;for(auto* c:colors)hit|=c==t;if(hit){EndPass();break;}}
    }
    const u32 sample_query=gpu_counters.ready?~0u:AllocateSampleQuery();const VkDescriptorSet set=pipeline.descriptor_set;

    const auto begin=[&](VkCommandBuffer cmd){
        // (The barrier opening the pass orders memory; only a layout change needs an image barrier.)
        for(auto* color:colors)if(color){if(color->layout!=VK_IMAGE_LAYOUT_GENERAL)Transition(cmd,*color,VK_IMAGE_LAYOUT_GENERAL);color->gpu_written=true;}
        if(depth){if(depth->layout!=VK_IMAGE_LAYOUT_GENERAL)Transition(cmd,*depth,VK_IMAGE_LAYOUT_GENERAL);depth->gpu_written=true;}
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};ri.renderArea.extent=extent;ri.layerCount=1;ri.colorAttachmentCount=state.color_count;ri.pColorAttachments=attachments.data();if(depth&&depth->info.depth)ri.pDepthAttachment=&da;if(depth&&depth->info.stencil)ri.pStencilAttachment=&da;
        if(stats_pool)vkCmdResetQueryPool(cmd,stats_pool,0,1);
        if(sample_query!=~0u)vkCmdResetQueryPool(cmd,samples.pool,sample_query,1);
        vkCmdBeginRendering(cmd,&ri);};
    const u32 predicate=draw_predicate;const bool inverted=draw_inverted;draw_predicate=~0u;
    const auto record=[&](VkCommandBuffer cmd){
        // State already set on this command buffer by the previous draw is not set again (unless
        // other work was recorded in between). OOT_NO_DRAW_STATE_CACHE=1 sets everything per draw.
        static const bool no_cache=std::getenv("OOT_NO_DRAW_STATE_CACHE")!=nullptr;
        struct DrawStateCache{VkCommandBuffer cmd=VK_NULL_HANDLE;u64 serial=~0ull;VkPipeline pipeline=VK_NULL_HANDLE;VkPipelineLayout layout=VK_NULL_HANDLE;VkDescriptorSet set=VK_NULL_HANDLE;
            VkViewport viewport{};VkRect2D scissor{};float blend[4]{};float bias[3]{};u32 stencil[6]{};VkBuffer vb[16]{};VkDeviceSize vo[16]{};bool valid=false;};
        static DrawStateCache cache;
        if(no_cache||cache.cmd!=cmd||cache.serial!=OtherWorkSerial()){cache=DrawStateCache{};cache.cmd=cmd;}
        const VkPipeline bound=pipeline.pipeline;
        if(cache.pipeline!=bound){vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,bound);cache.pipeline=bound;}
        if(cache.layout!=pipeline.layout){ // (push constants are the same for every draw; set again for a new layout)
            float push[32]{};push[8]=1;vkCmdPushConstants(cmd,pipeline.layout,VK_SHADER_STAGE_ALL,0,sizeof push,push);cache.layout=pipeline.layout;cache.set=VK_NULL_HANDLE;}
        if(pipeline.push){if(!descriptor_write_list.empty())vk.push_descriptor_set(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline.layout,0,(u32)descriptor_write_list.size(),descriptor_write_list.data());cache.set=VK_NULL_HANDLE;}
        else if(cache.set!=set){vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline.layout,0,1,&set,0,nullptr);cache.set=set;}
        auto viewport=state.viewport;viewport.minDepth=std::clamp(viewport.minDepth,0.f,1.f);viewport.maxDepth=std::clamp(viewport.maxDepth,0.f,1.f);
        // Upper-left origin: NVN maps clip-space +y to the top row, which Vulkan does with a negative-height viewport.
        if(!WindowOriginLowerLeft()){viewport.y+=viewport.height;viewport.height=-viewport.height;}
        if(!cache.valid||std::memcmp(&viewport,&cache.viewport,sizeof viewport)){vkCmdSetViewport(cmd,0,1,&viewport);cache.viewport=viewport;}
        auto scissor=state.scissor;scissor.offset.x=std::max(scissor.offset.x,0);scissor.offset.y=std::max(scissor.offset.y,0);scissor.extent.width=std::min(scissor.extent.width,extent.width);scissor.extent.height=std::min(scissor.extent.height,extent.height);
        if(!cache.valid||std::memcmp(&scissor,&cache.scissor,sizeof scissor)){vkCmdSetScissor(cmd,0,1,&scissor);cache.scissor=scissor;}
        if(!cache.valid||std::memcmp(state.blend_color,cache.blend,sizeof cache.blend)){vkCmdSetBlendConstants(cmd,state.blend_color);std::memcpy(cache.blend,state.blend_color,sizeof cache.blend);}
        if(!cache.valid||std::memcmp(state.depth_bias,cache.bias,sizeof cache.bias)){vkCmdSetDepthBias(cmd,state.depth_bias[1],state.depth_bias[2],state.depth_bias[0]);std::memcpy(cache.bias,state.depth_bias,sizeof cache.bias);}
        {const u32 stencil[6]{state.stencil_compare[0],state.stencil_write[0],state.stencil_ref[0],state.stencil_compare[1],state.stencil_write[1],state.stencil_ref[1]};
         if(!cache.valid||std::memcmp(stencil,cache.stencil,sizeof stencil)){
            for(u32 i=0;i<2;++i){VkStencilFaceFlags face=i?VK_STENCIL_FACE_BACK_BIT:VK_STENCIL_FACE_FRONT_BIT;vkCmdSetStencilCompareMask(cmd,face,state.stencil_compare[i]);vkCmdSetStencilWriteMask(cmd,face,state.stencil_write[i]);vkCmdSetStencilReference(cmd,face,state.stencil_ref[i]);}
            std::memcpy(cache.stencil,stencil,sizeof stencil);}}
        // Only the streams this draw's attributes read. A binding left over from earlier draws may
        // name memory the game has since released (it crashed here: "unmapped vertex buffer"); an
        // attribute reading unmapped memory gets zeros (the zero buffer; reads past it are robust).
        {std::array<bool,32> used{};for(u32 a=0;a<32;++a)if(state.attributes[a]&0x40)used[state.attributes[a]&31]=true;
        for(u32 i=0;i<16;++i)if(used[i]){
            BufferRange range{};if(state.vertices[i].address)range=GpuLookupAddress(state.vertices[i].address);
            VkBuffer buffer=range.buffer;VkDeviceSize offset=range.offset;
            if(!range.pool){const auto zero=ZeroBuffer();buffer=zero.buffer;offset=0;
                static std::set<std::pair<u64,u64>> told;const u64 fs=state.stages[1].shader?state.stages[1].shader->address:0;
                if(told.insert({state.vertices[i].address,fs}).second&&told.size()<64)RT_LOG("vertex stream %u at %#llx is not mapped (draw with fragment shader %#llx): reads zeros",i,(unsigned long long)state.vertices[i].address,(unsigned long long)fs);}
            if(!cache.valid||cache.vb[i]!=buffer||cache.vo[i]!=offset){vkCmdBindVertexBuffers(cmd,i,1,&buffer,&offset);cache.vb[i]=buffer;cache.vo[i]=offset;}}}
        cache.valid=true;cache.serial=OtherWorkSerial();
        // DrawElements*(mode, index type, count, index buffer[, base vertex[, base instance, instances]])
        const auto* x=command.x;const Op op=(Op)command.op;
        if(predicate!=~0u){VkConditionalRenderingBeginInfoEXT ci{VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT};ci.buffer=gpu_counters.predicates;ci.offset=predicate*4ull;if(inverted)ci.flags=VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT;gpu_counters.begin(cmd,&ci);}
        if(stats_pool)vkCmdBeginQuery(cmd,stats_pool,0,0);
        if(sample_query!=~0u)vkCmdBeginQuery(cmd,samples.pool,sample_query,VK_QUERY_CONTROL_PRECISE_BIT);
        const bool indexed=op==Op::DrawElements||op==Op::DrawElementsInstanced||op==Op::DrawElementsBaseVertex;
        const bool instanced=op==Op::DrawElementsInstanced||op==Op::DrawArraysInstanced;
        if(mesh){
            if(x[0])Fatal("nonzero first mesh task %llu requires workgroup base support",(unsigned long long)x[0]);
            static const auto draw_mesh=reinterpret_cast<PFN_vkCmdDrawMeshTasksEXT>(vkGetDeviceProcAddr(vk.device,"vkCmdDrawMeshTasksEXT"));
            static const auto draw_mesh_indirect=reinterpret_cast<PFN_vkCmdDrawMeshTasksIndirectEXT>(vkGetDeviceProcAddr(vk.device,"vkCmdDrawMeshTasksIndirectEXT"));
            if(!draw_mesh||!draw_mesh_indirect)Fatal("Vulkan mesh draw entry point unavailable");
            if(indirect_slot)draw_mesh_indirect(cmd,indirect_clamp.buffer,indirect_slot-1,1,16);
            else draw_mesh(cmd,(u32)x[1],1,1);
        }
        else if(indexed){const auto range=GpuLookupAddress(x[3]);if(!range.pool){static int told=0;if(told++<16)RT_LOG("draw skipped: index buffer %#llx is not mapped",(unsigned long long)x[3]);return;}const VkIndexType types[]={VK_INDEX_TYPE_UINT8_KHR,VK_INDEX_TYPE_UINT16,VK_INDEX_TYPE_UINT32};if(x[1]>2)Fatal("invalid index type %llu",(unsigned long long)x[1]);vkCmdBindIndexBuffer(cmd,range.buffer,range.offset,types[x[1]]);
            const s32 base_vertex=op==Op::DrawElements?0:(s32)x[4];vkCmdDrawIndexed(cmd,(u32)x[2],instanced?(u32)x[6]:1,0,base_vertex,instanced?(u32)x[5]:0);}
        else vkCmdDraw(cmd,x[2],instanced?x[4]:1,x[1],instanced?x[3]:0);
        if(stats_pool)vkCmdEndQuery(cmd,stats_pool,0);
        if(sample_query!=~0u)vkCmdEndQuery(cmd,samples.pool,sample_query);
        if(predicate!=~0u)gpu_counters.end(cmd);};
    const u64 profile_key=state.stages[1].shader?state.stages[1].shader->address:0;
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
        for(u32 st=0;st<8;++st)if(st!=5&&state.stages[st].shader)for(const auto& u:state.stages[st].uniforms)if(u.address)access.reads.push_back({u.address,u.address+std::max<u64>(u.size,256)});
        for(const auto& v:state.vertices)if(v.address)access.reads.push_back({v.address,v.address+(v.size?v.size:(1u<<20))});
        if((command.op==(u32)Op::DrawElements||command.op==(u32)Op::DrawElementsInstanced||command.op==(u32)Op::DrawElementsBaseVertex)&&command.x[3])
            access.reads.push_back({command.x[3],command.x[3]+(1u<<20)});
        }
        PassDraw(key,[&](VkCommandBuffer cmd){
            begin(cmd);
            if(count_samples){ // the pass instance's first occlusion segment
                const u32 q=samples.next++;vkCmdBeginQuery(cmd,samples.pool,q,VK_QUERY_CONTROL_PRECISE_BIT);gpu_counters.segment_query=q;
                SetPassEndHook([q](VkCommandBuffer c){vkCmdEndQuery(c,samples.pool,q);gpu_counters.segment_query=~0u;});}
        },record,profile_key,tracking&&!mesh_indirect_slot?&access:nullptr);
        if(!fast){ // what the next draw may reuse
            fd.valid=pipeline.push&&descriptor_fast_eligible&&!descriptor_writes&&predicate==~0u&&!mesh;
            if(fd.valid){fd.pipeline=&pipeline;fd.colors=colors;fd.depth=depth;fd.extent=extent;fd.attachments=attachments;fd.da=da;fd.key=key;
                fd.flushes=FlushCount();fd.pools=PoolGeneration();fd.texture_writes=TextureWriteSequence();}
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
    if(LogThisFrame()&&std::getenv("OOT_LUMA_TRACE"))for(u32 i=0;i<state.color_count;++i)if(state.targets.colors[i]){auto* t=HostTexture(state.targets.colors[i]);
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
std::map<VkShaderModule,std::unique_ptr<Pipeline>> compute_pipelines;
// indirect: guest address of {x, y, z} workgroup counts written by GPU work in the open batch;
// they are read on the GPU (vkCmdDispatchIndirect, and copied into the driver bank's +0x4..+0xc).
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
    const int choice=gfx::Current().ai_upscaler; // (the settings window: 1 off, 2 on; 0 as below; read every call, so it is live)
    if(choice)return choice==1;
    static const bool slow=[]{const char* e=std::getenv("OOT_NETWORKS");if(e)return std::atoi(e)==0;
        const u32 vendor=Vk().properties.vendorID;const bool s=vendor==0x1002||vendor==0x8086;
        if(s)RT_LOG("upscaling networks: a linear upscale instead (this GPU's matrix multiplies are emulated slowly)");return s;}();
    return slow;
}
void NetworkStandIn(ShaderVariant& v){
    std::vector<Texture*> images,textures;
    for(const auto& r:v.compiled.resources){
        if(r.kind!=nvshader::ResourceKind::Image&&r.kind!=nvshader::ResourceKind::Texture)continue;
        for(u32 e=0;e<r.count;++e){const u32 handle=ResourceHandle(r,v,e);if(!handle)continue;
            Texture* t=Binding(handle).host.get();if(!t)continue;(r.kind==nvshader::ResourceKind::Image?images:textures).push_back(t);}
    }
    const auto area=[](const Texture* t){return (u64)t->width*t->height;};
    const auto has=[](const std::vector<Texture*>& list,const Texture* t){for(auto* x:list)if(x==t||(x->storage&&x->storage==t->storage))return true;return false;};
    Texture* out=nullptr;for(auto* t:images)if(t->depth<=1&&!has(textures,t)&&(!out||area(t)>area(out)))out=t;
    if(!out)return; // (an earlier pass: feature maps in buffers)
    Texture* in=nullptr;for(auto* t:textures)if(t->depth<=1&&t->target==out->target&&!has(images,t)&&!t->info.depth&&(!in||area(t)>area(in)))in=t;
    if(!in||area(in)>=area(out))return;
    EnsureUploaded(*in);EnsureImage(*out);
    auto& vk=Vk();VkFormatProperties fi,fo;vkGetPhysicalDeviceFormatProperties(vk.physical,in->info.vk,&fi);vkGetPhysicalDeviceFormatProperties(vk.physical,out->info.vk,&fo);
    if(!(fi.optimalTilingFeatures&VK_FORMAT_FEATURE_BLIT_SRC_BIT)||!(fi.optimalTilingFeatures&VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)||!(fo.optimalTilingFeatures&VK_FORMAT_FEATURE_BLIT_DST_BIT)){
        static int told=0;if(told++<8)RT_LOG("upscaling network stand-in: formats %d -> %d cannot be blitted",(int)in->info.vk,(int)out->info.vk);return;}
    {static int shown=0;if(shown++<8)RT_LOG("upscaling network stand-in: %ux%u fmt %u -> %ux%u fmt %u",in->width,in->height,in->format,out->width,out->height,out->format);}
    Texture* src=in;Texture* dst=out;
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
    if(state.root[1][0][2]<8192&&state.root[1][0][3]<8192&&state.root[1][0][2]>16)frame_render_width=std::max(frame_render_width,state.root[1][0][2]+1); // (compute root: render size - 1)
    auto& stage=state.stages[5];if(!stage.shader)Fatal("compute dispatch without a compute shader");
    if(SkippedByFile(stage.shader->address))return;
    struct Phases{std::chrono::steady_clock::time_point t0=std::chrono::steady_clock::now(),last=t0;double ms[6]{};u64 shader=0;
        void mark(int i){const auto now=std::chrono::steady_clock::now();ms[i]+=std::chrono::duration<double,std::milli>(now-last).count();last=now;}
        ~Phases(){const double total=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
            if(total>100){static std::atomic<u32> shown{0};if(shown++<30)RT_LOG("slow dispatch of shader %#llx: %.0f ms (shader %.0f, address table %.0f, pipeline %.0f, descriptors %.0f, recording %.0f)",
                (unsigned long long)shader,total,ms[0],ms[1],ms[2],ms[3],ms[4]);}}} phases;phases.shader=stage.shader->address;
    auto& v=CompileShader(*stage.shader,0,nullptr);auto& vk=Vk();phases.mark(0);
    if(!v.module)return; // not translatable
    if(IsNetwork(*stage.shader)&&SlowNetworks()){NetworkStandIn(v);return;}
    if(v.compiled.global_loads||v.compiled.global_stores)GlobalAddressTable(); // (pools may have changed)
    phases.mark(1);
    auto& slot=compute_pipelines[v.module];
    if(!slot){
        slot=std::make_unique<Pipeline>();slot->shaders={&v};
        for(const auto& r:v.compiled.resources){const bool buffer=r.kind==nvshader::ResourceKind::UniformBuffer||r.kind==nvshader::ResourceKind::StorageBuffer;
            slot->bindings.push_back({r.binding,DescriptorType(r.kind),buffer?1:r.count,VK_SHADER_STAGE_COMPUTE_BIT,nullptr});}
        VkDescriptorSetLayoutCreateInfo di{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};di.bindingCount=slot->bindings.size();di.pBindings=slot->bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(vk.device,&di,nullptr,&slot->descriptors));
        VkPushConstantRange push{VK_SHADER_STAGE_ALL,0,128};
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};li.setLayoutCount=1;li.pSetLayouts=&slot->descriptors;li.pushConstantRangeCount=1;li.pPushConstantRanges=&push;
        VK_CHECK(vkCreatePipelineLayout(vk.device,&li,nullptr,&slot->layout));
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};ci.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        ci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;ci.stage.module=v.module;ci.stage.pName="main";ci.layout=slot->layout;
        // AMD (RADV's optimizer): OoT's largest compute programs (300k+ SPIR-V words) took up to 3.5 minutes to build on the
        // Steam Deck, the game frozen meanwhile. Programs past OOT_NOOPT_WORDS (40000; 0: never) are built without the
        // optimizer first (from Hookshot f414a76), and optimized on a worker, which takes over when done.
        static const u64 noopt_words=std::getenv("OOT_NOOPT_WORDS")?std::strtoull(std::getenv("OOT_NOOPT_WORDS"),nullptr,10):40000;
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
                VkPipeline optimized=VK_NULL_HANDLE;
                if(vkCreateComputePipelines(vk.device,PipelineCache(),1,&ci,nullptr,&optimized)!=VK_SUCCESS)return;
                const VkPipeline old=target->pipeline.exchange(optimized);
                RT_LOG("compute pipeline for shader %#llx optimized on a worker in %.0f ms",(unsigned long long)address,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());
                AfterFlush([old]{vkDestroyPipeline(Vk().device,old,nullptr);}); // (after work recorded with it)
            });
        }
        {const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
         if(ms>20){u64 h=0xcbf29ce484222325ull;for(u32 w:v.compiled.spirv)h=(h^w)*0x100000001b3ull;
             RT_LOG("compute pipeline for shader %#llx built in %.0f ms (SPIR-V %zu words, hash %016llx%s%s)",(unsigned long long)stage.shader->address,ms,v.compiled.spirv.size(),(unsigned long long)h,
                    v.compiled.global_loads||v.compiled.global_stores?", global memory":"",quick?", without the optimizer":"");}}
    }
    auto& pipeline=*slot;phases.mark(2);
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
            const u64 push[2]{indirect_source,c.address+slot};
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
        float push[32]{};push[8]=1;vkCmdPushConstants(cmd,pipeline.layout,VK_SHADER_STAGE_ALL,0,sizeof push,push);
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
        stage.shader->control.size()>0xa9?stage.shader->control[0xa9]:0,state.root[1][0][0],state.root[1][0][1],state.root[1][0][2],state.root[1][0][3],state.root[1][0][4],state.root[1][0][5],state.root[1][0][6],state.root[1][0][7]);
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
    std::memcpy(reinterpret_cast<u8*>(state.root[which][table])+offset,data,std::min<size_t>(size,0x100-offset));
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
        const auto before=std::array<const ShaderCode*,9>{state.stages[0].shader,state.stages[1].shader,state.stages[2].shader,state.stages[3].shader,state.stages[4].shader,state.stages[5].shader,state.stages[6].shader,state.stages[7].shader,state.stages[8].shader};
        struct Check{const std::array<const ShaderCode*,9>& b;~Check(){for(u32 i=0;i<9;++i)if(state.stages[i].shader!=b[i]){fast_draw_allowed=false;break;}}} check{before}; // (the same program bound again keeps the fast path)
        auto* program=HostProgram(x[0]);
        for(u32 i=0;i<9;++i) if(x[1]&(1u<<i)) state.stages[i].shader=nullptr;
        if(program) for(const auto& code:program->shaders) {
            if(code.stage>7)Fatal("binding NVN shader stage %u",code.stage);
            const u32 stage=ShaderPublicStage[code.stage];if(x[1]&(1u<<stage))state.stages[stage].shader=&code;}
        break;
    }
    case Op::BindUniformBuffer: if(x[0]>=9||x[1]>=18) Fatal("invalid uniform binding"); state.stages[x[0]].uniforms[x[1]]={x[2],x[3]};break;
    case Op::BindStorageBuffer: if(x[0]>=9||x[1]>=16) Fatal("invalid storage binding"); if(state.stages[x[0]].storage[x[1]].address!=x[2]||state.stages[x[0]].storage[x[1]].size!=x[3]){state.stages[x[0]].storage[x[1]]={x[2],x[3]};fast_draw_allowed=false;}break;
    case Op::BindVertexBuffer: state.vertices.at(x[0])={x[1],x[2]};break;
    case Op::BindVertexBuffers: for(u32 i=0;i<x[1];++i)std::memcpy(&state.vertices.at(x[0]+i),data+i*16,16);break;
    case Op::BindVertexAttribState: {decltype(state.attributes) next{};std::memcpy(next.data(),data,std::min(command.data.size(),sizeof next));
        if(next!=state.attributes){state.attributes=next;fast_draw_allowed=false;}break;} // (a different vertex format changes the pipeline)
    case Op::BindVertexStreamState: {decltype(state.streams) next{};std::memcpy(next.data(),data,std::min(command.data.size(),sizeof next));
        if(next!=state.streams){state.streams=next;fast_draw_allowed=false;}break;}
    case Op::BindTexture: if(state.stages.at(x[0]).textures.at(x[1])!=x[2]){state.stages[x[0]].textures[x[1]]=x[2];fast_draw_allowed=false;}break;
    case Op::BindTextures: for(u32 i=0;i<x[2];++i)std::memcpy(&state.stages.at(x[0]).textures.at(x[1]+i),data+i*8,8);break;
    case Op::BindImage: state.stages.at(x[0]).images.at(x[1])=x[2];break;
    case Op::BindImages: for(u32 i=0;i<x[2];++i)std::memcpy(&state.stages.at(x[0]).images.at(x[1]+i),data+i*8,8);break;
    case Op::SetTexturePool: state.textures=HostTexturePool(x[0]);break;
    case Op::SetSamplerPool: state.samplers=HostSamplerPool(x[0]);break;
    case Op::SetRenderTargets: if(std::memcmp(&state.targets,data,sizeof state.targets)||state.color_count!=x[0]){std::memcpy(&state.targets,data,sizeof state.targets);state.color_count=x[0];fast_draw_allowed=false;}break;
    case Op::SetViewport: state.viewport.x=(s32)x[0];state.viewport.y=(s32)x[1];state.viewport.width=(s32)x[2];state.viewport.height=(s32)x[3];break;
    case Op::SetViewports: if(x[0]==0 && x[1])std::memcpy(&state.viewport,data,16);break;
    case Op::SetDepthRange: state.viewport.minDepth=command.f[0];state.viewport.maxDepth=command.f[1];break;
    case Op::SetDepthRanges: if(x[0]==0&&x[1])std::memcpy(&state.viewport.minDepth,data,8);break;
    case Op::SetScissor: state.scissor={{(s32)x[0],(s32)x[1]},{(u32)x[2],(u32)x[3]}};break;
    case Op::SetScissors: if(x[0]==0&&x[1])std::memcpy(&state.scissor,data,16);break;
    case Op::BindBlendState: {u64 blend;std::memcpy(&blend,data,8);if(state.blends[blend&7]!=blend){state.blends[blend&7]=blend;fast_draw_allowed=false;}break;}
    case Op::BindColorState: {u32 v;std::memcpy(&v,data,4);if(v!=state.color){state.color=v;fast_draw_allowed=false;}break;}
    case Op::BindChannelMaskState: {u32 v;std::memcpy(&v,data,4);if(v!=state.channel){state.channel=v;fast_draw_allowed=false;}break;}
    case Op::BindDepthStencilState: std::memcpy(&state.depth,data,8);break;
    case Op::BindPolygonState: {u32 v;std::memcpy(&v,data,4);if(v!=state.polygon){state.polygon=v;fast_draw_allowed=false;}break;}
    case Op::BindMultisampleState: break;
    case Op::SetBlendColor: std::memcpy(state.blend_color,data,16);break;
    case Op::SetDepthBias: std::memcpy(state.depth_bias,command.f,12);break;
    case Op::SetLineWidth: if(state.line_width!=command.f[0]){state.line_width=command.f[0];fast_draw_allowed=false;}break;
    case Op::SetStencilMask: for(u32 i=0;i<2;++i)if(x[0]&(1u<<i))state.stencil_write[i]=x[1];break;
    case Op::SetStencilValueMask: for(u32 i=0;i<2;++i)if(x[0]&(1u<<i))state.stencil_compare[i]=x[1];break;
    case Op::SetStencilRef: for(u32 i=0;i<2;++i)if(x[0]&(1u<<i))state.stencil_ref[i]=x[1];break;
    case Op::SetPrimitiveRestart: if(state.restart!=(bool)x[0]){state.restart=x[0];fast_draw_allowed=false;}break;
    case Op::SetRenderEnable: state.render_enable=x[0];state.render_condition=0;break;
    case Op::SetRenderEnableConditional:
        if(x[0]>1)Fatal("invalid conditional render mode %llu",(unsigned long long)x[0]);
        state.render_enable=true;state.render_condition=x[1];state.render_if_equal=x[0]==0;break;
    case Op::SetConservativeRasterEnable: if(x[0])Fatal("conservative rasterization pending");break;
    case Op::SetTiledCacheAction: case Op::SetTiledCacheTileSize: break; // console tile-cache optimization
    case Op::ClearColor:case Op::ClearColori:case Op::ClearColorui: {
        if(!DrawEnabled())break;
        const u32 predicate=draw_predicate;const bool inverted=draw_inverted;draw_predicate=~0u;
        ++stats.clears;
        const u64 object=state.targets.colors[x[0]];if(!object)break;auto& t=*HostTexture(object);
        if(LogThisFrame()){float f[4];std::memcpy(f,data,16);RT_LOG("clear color %llu target %#llx %ux%u fmt %u to (%g %g %g %g)",(unsigned long long)x[0],(unsigned long long)object,t.width,t.height,t.format,f[0],f[1],f[2],f[3]);}
        if(x[2]!=15)Fatal("partial color clear requires attachment clear");
        if(predicate!=~0u){PredicatedClear(predicate,inverted,&t,state.targets.color_views[x[0]],VK_IMAGE_ASPECT_COLOR_BIT,data,0,0);t.gpu_written=true;break;}
        Immediate([&](VkCommandBuffer cmd){Transition(cmd,t,VK_IMAGE_LAYOUT_GENERAL);VkClearColorValue color;std::memcpy(&color,data,16);
            const auto& view=state.targets.color_views[x[0]];VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT,view.levels_base,1,view.layers_base,view.layers_count?view.layers_count:1};
            vkCmdClearColorImage(cmd,t.image,t.layout,&color,1,&range);});t.gpu_written=true;break;
    }
    case Op::ClearDepthStencil: {
        if(!DrawEnabled())break;
        const u32 predicate=draw_predicate;const bool inverted=draw_inverted;draw_predicate=~0u;
        if(!state.targets.depth)break;auto& t=*HostTexture(state.targets.depth);
        const auto& view=state.targets.depth_view;
        VkImageAspectFlags aspect=(x[0]?VK_IMAGE_ASPECT_DEPTH_BIT:0)|(x[2]?VK_IMAGE_ASPECT_STENCIL_BIT:0);aspect &= AspectOf(t);if(!aspect)break;
        if(predicate!=~0u){PredicatedClear(predicate,inverted,&t,view,aspect,nullptr,command.f[0],(u32)x[1]);t.gpu_written=true;break;}
        Immediate([&](VkCommandBuffer cmd){Transition(cmd,t,VK_IMAGE_LAYOUT_GENERAL);VkClearDepthStencilValue value{command.f[0],(u32)x[1]};
            VkImageSubresourceRange range{aspect,view.levels_base,1,view.layers_base,view.layers_count?view.layers_count:1};vkCmdClearDepthStencilImage(cmd,t.image,t.layout,&value,1,&range);});t.gpu_written=true;break;
    }
    case Op::SetDepthClamp: state.depth_clamp=x[0];break;
    case Op::SetPointSize: if(state.point_size!=command.f[0]){state.point_size=command.f[0];fast_draw_allowed=false;}break;
    case Op::SetRasterizerDiscard: state.discard=x[0];break;
    case Op::SetSampleMask: state.sample_mask=(u32)x[0];break;
    case Op::SetDepthBounds: state.depth_bounds=x[0];state.depth_bounds_range[0]=command.f[0];state.depth_bounds_range[1]=command.f[1];break;
    case Op::SetAlphaRef: break; // alpha test is done by the game's shaders on this platform
    case Op::BindUniformBuffers: for(u32 i=0;i<x[2];++i){if(x[0]>=9||x[1]+i>=18)Fatal("invalid uniform binding");std::memcpy(&state.stages[x[0]].uniforms[x[1]+i],data+i*16,16);}break;
    case Op::BindStorageBuffers: for(u32 i=0;i<x[2];++i){if(x[0]>=9||x[1]+i>=16)Fatal("invalid storage binding");std::memcpy(&state.stages[x[0]].storage[x[1]+i],data+i*16,16);}break;
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
        auto& t=*HostTexture(x[0]);TextureCopy region;std::memcpy(&region,data,sizeof region);
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
            if(trace){const u64 a=x[0];const u64 shader=state.stages[5].shader?state.stages[5].shader->address:0;AfterFlush([a,shader]{const u32* g=static_cast<const u32*>(ResolveAddress(a,12));
                static std::map<std::tuple<u64,u32,u32,u32>,u32> seen;if(seen[{shader,g[0],g[1],g[2]}]++==0)RT_LOG("indirect dispatch shader %#llx: %u %u %u",(unsigned long long)shader,g[0],g[1],g[2]);});}
            break;}}Flush();}
        const u32* g=static_cast<const u32*>(ResolveAddress(x[0],12));if(g[0]&&g[1]&&g[2])Dispatch(g[0],g[1],g[2]);break;}
    case Op::PushGraphicsConstants: PushRoot(0,command,command.data.data(),command.data.size());break;
    case Op::PushComputeConstants: PushRoot(1,command,command.data.data(),command.data.size());break;
    case Op::PushGraphicsConstantsDeferred: case Op::PushComputeConstantsDeferred: {
        const size_t size=std::min<u64>(x[2],0x100);
        PushRoot(command.op==(u32)Op::PushComputeConstantsDeferred,command,static_cast<const u8*>(ResolveAddress(x[3],size)),size);break;}
    case Op::DrawMeshTasks:
        state.primitive=4;if(DrawEnabled()){Draw(command);++stats.draws;}else ++stats.skipped;break;
    case Op::DrawMeshTasksIndirect: {
        if(!DrawEnabled()){++stats.skipped;break;}
        // NVN's two-word command {task count, first task} is usually written by GPU culling work
        // earlier in the same batch (foliage): it is converted on the GPU into Vulkan's
        // {x, y, z} group counts (indirect_clamp.comp, mesh mode) and drawn from there. Reading it
        // on the CPU at recording time saw zero tasks and dropped every grass draw.
        static const bool cpu_args=std::getenv("OOT_MESH_INDIRECT_CPU")!=nullptr;
        if(cpu_args){
            if(HasPendingWrite(x[0],8))Flush();
            u32 args[2];std::memcpy(args,ResolveAddress(x[0],sizeof args),sizeof args);
            Command direct=command;direct.op=(u32)Op::DrawMeshTasks;direct.x[0]=args[1];direct.x[1]=args[0];
            state.primitive=4;if(args[0]){Draw(direct);++stats.draws;}break;
        }
        InitializeIndirectClamp();
        const u64 source=GuestDeviceAddress(x[0]);
        if(LogThisFrame()){const u64 address=x[0];const bool pending=HasPendingWrite(address,8);
            AfterFlush([address,pending]{u32 a[2];std::memcpy(a,ResolveAddress(address,8),8);
                RT_LOG("mesh indirect args @%#llx (GPU-pending %d) after its batch: count %u first %u",(unsigned long long)address,(int)pending,a[0],a[1]);});}
        auto& c=indirect_clamp;const u64 slot=u64(c.next++%IndirectClamp::Slots)*16;
        ImmediateLight([&](VkCommandBuffer cmd){
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
            const u64 push[2]{source,(c.address+slot)|1};
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,c.pipeline);
            vkCmdPushConstants(cmd,c.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof push,push);
            vkCmdDispatch(cmd,1,1,1);
            barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,0,1,&barrier,0,nullptr,0,nullptr);
        });
        Command direct=command;direct.op=(u32)Op::DrawMeshTasks;direct.x[0]=0;direct.x[1]=0;
        mesh_indirect_slot=slot+1;
        state.primitive=4;Draw(direct);++stats.draws;
        mesh_indirect_slot=0;
        break;
    }
    case Op::DrawElementsBaseVertex:
    case Op::DrawArrays:case Op::DrawElements:case Op::DrawArraysInstanced:case Op::DrawElementsInstanced:
        state.primitive=x[0];if(DrawEnabled()){Draw(command);++stats.draws;}else ++stats.skipped;break;
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
void EnsureCounterRings(){
    auto& g=gpu_counters;
    if(samples.next+256<SampleCounter::Capacity&&g.next_predicate+4096<GpuCounters::PredicateCapacity&&g.entries_used+4096<GpuCounters::EntryCapacity)return;
    Flush();
    vkResetQueryPool(Vk().device,samples.pool,0,SampleCounter::Capacity);
    samples.next=0;g.processed=0;g.entries_used=0;g.next_predicate=0;g.segment_query=~0u;g.reports.clear();
}
// Ends the open pass's occlusion segment and starts the next inside the same pass, so every query
// before the returned boundary belongs to work recorded so far.
u32 SplitSegment(){
    auto& g=gpu_counters;
    EnsureCounterRings();
    if(g.segment_query==~0u)return samples.next;
    const u32 ended=g.segment_query,next=samples.next++;
    const bool split=RecordInPass([&](VkCommandBuffer cmd){
        vkCmdEndQuery(cmd,samples.pool,ended);
        vkCmdBeginQuery(cmd,samples.pool,next,VK_QUERY_CONTROL_PRECISE_BIT);});
    if(!split){g.segment_query=~0u;return samples.next;} // (the pass already ended)
    g.segment_query=next;
    SetPassEndHook([next](VkCommandBuffer cmd){vkCmdEndQuery(cmd,samples.pool,next);gpu_counters.segment_query=~0u;});
    return next;
}
void ResetRenderCounter(u32 counter) {
    if(counter!=1)return;
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
    if(!RecordInPass(record))ImmediateLight(record);
    r.writes.push_back({q,address,value!=nullptr,value?*value:0});
    NoteWriteBack(address,16);
    return true;
}
// (Before submission, outside render passes.)
struct TimestampScale{VkPipeline pipeline=VK_NULL_HANDLE;VkPipelineLayout layout=VK_NULL_HANDLE;VkBuffer buffer=VK_NULL_HANDLE;VkDeviceMemory memory=VK_NULL_HANDLE;
    VkDeviceAddress address=0;u64 period=0;static constexpr u32 Slots=ReportTimes::Capacity;} timestamp_scale;
bool TimestampScaling(){ // (ticks of another length than 1 ns: the conversion's pipeline and tick buffer, made once)
    auto& vk=Vk();auto& t=timestamp_scale;
    static const bool needed=vk.properties.limits.timestampPeriod!=1.0f;
    if(!needed||t.pipeline)return needed;
    t.period=(u64)std::llround(vk.properties.limits.timestampPeriod*65536.0);
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
    return (u64)((s64)now-r.offset);
}
void ReportRenderCounter(u32 counter,u64 address) {
    auto* out=static_cast<u64*>(ResolveAddress(address,16));
    static const bool force_visible=std::getenv("OOT_SAMPLES_VISIBLE")!=nullptr; // (diagnostic: every sample count large)
    if((frame_skipping||force_visible)&&counter==1){ // (no draws ran: report the object visible so the next rendered frame draws it)
        out[0]=1u<<20;out[1]=GuestTimestampNs();known_reports[address]=1u<<20;gpu_counters.reports.erase(address);return;}
    if(counter==1){NotePendingReport(address);known_reports.erase(address);}
    else known_reports[address]=0;
    const bool on_gpu=gpu_counters.ready;
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
std::shared_ptr<void> SaveRenderState() { return std::make_shared<State>(state); }
void RestoreRenderState(const std::shared_ptr<void>& saved) { state = *static_cast<const State*>(saved.get()); }
bool FrameLogged() { return LogThisFrame(); }
u64 CurrentFrameNumber() { return frame_number; }
bool CurrentFrameSkipped() { return frame_skipping; }
bool FrameSkipEnabled() { return FrameSkipOn(); }
void NextFrame() {
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
    frame_skipping=FrameSkipOn()&&skip_level&&frame_number%(skip_level+1)!=0;
    if(frame_number%300==0){ReportGpuProfile();{static u64 lf=0,ld=0;static const bool st=std::getenv("OOT_SET_STATS")!=nullptr;if(st&&stats.draws>ld)RT_LOG("fast draws: %.0f%% of %.0f draws per frame",100.0*(fast_draw_count-lf)/(stats.draws-ld),(stats.draws-ld)/300.0);if(st){std::string b;for(u32 i=0;i<128;++i)if(fast_draw_breakers[i]>300){char t[32];std::snprintf(t,sizeof t," op%u:%.1f",i,fast_draw_breakers[i]/300.0);b+=t;}
            RT_LOG("  fast path ended by (per frame):%s; declined: flush %.1f pools %.1f pass %.1f shader %.1f constants %.1f",b.c_str(),fast_draw_fails[0]/300.0,fast_draw_fails[1]/300.0,fast_draw_fails[2]/300.0,fast_draw_fails[3]/300.0,fast_draw_fails[4]/300.0);
            std::memset(fast_draw_breakers,0,sizeof fast_draw_breakers);std::memset(fast_draw_fails,0,sizeof fast_draw_fails);}lf=fast_draw_count;ld=stats.draws;}static const bool rate=std::getenv("OOT_BINDING_RATE")!=nullptr;if(rate)ReportBindingRate();}
    { static std::map<u32,u32> widths; if(frame_render_width>1)++widths[frame_render_width]; frame_render_width=0;
      if(frame_number%300==0&&!widths.empty()){std::string t;for(auto& [w,n]:widths){char b[32];std::snprintf(b,sizeof b," %u:%u",w,n);t+=b;}RT_LOG("render widths (frames):%s",t.c_str());widths.clear();} }
    static const char* trigger = std::getenv("OOT_DRAW_LOG_TRIGGER");
    if (trigger && std::remove(trigger) == 0) triggered_frame = (s64)frame_number;
    if (LogThisFrame()) RT_LOG("==== logged frame %llu", (unsigned long long)frame_number);
}
void ReportPredicateStats() {
    RT_LOG("conditional draws: %llu drawn as depth tested, %llu decided on the CPU, %llu after a flush, %llu bracketed GPU, %llu with a predicate dispatch; %llu report passes for %llu reports",
        (unsigned long long)predicate_stats.depth_tested,(unsigned long long)predicate_stats.cpu,(unsigned long long)predicate_stats.flushed,(unsigned long long)predicate_stats.bracketed,(unsigned long long)predicate_stats.dispatched,
        (unsigned long long)predicate_stats.processing,(unsigned long long)predicate_stats.processed_reports);
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
    if(ms>threshold)RT_LOG("slow frame %.0f ms: shader translation %.1f ms, pipeline waits %.1f ms (%llu draws skipped), texture uploads %.1f ms (%llu; staging %.1f, deswizzle %.1f, decode %.1f), fence waits %.1f ms, executor idle %.1f ms (a queue parked %.1f ms)",
        ms,hitch.translate*1000,hitch.pipeline_wait*1000,(unsigned long long)pending,up,(unsigned long long)n,st,sw,dc,fence,idle,parked);
    hitch={};
}
void ReportCompileTime() {
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
