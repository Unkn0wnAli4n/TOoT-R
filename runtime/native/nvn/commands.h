// Recorded NVN command lists and their execution on Vulkan.
#pragma once
#include <atomic>
#include <chrono>

#include <memory>
#include <variant>
#include <vector>

#include "vk.h"
#include "objects.h"

namespace nvn {

enum class Op : u32 {
    DrawArrays, DrawElements, DrawArraysInstanced, DrawElementsInstanced,
    DrawArraysIndirect, DrawElementsIndirect, DispatchCompute, DispatchComputeIndirect,
    BindProgram, BindUniformBuffer, BindStorageBuffer, BindVertexBuffer, BindVertexBuffers,
    BindVertexAttribState, BindVertexStreamState, BindTexture, BindTextures, BindImage, BindImages,
    SetTexturePool, SetSamplerPool, SetRenderTargets, SetViewport, SetViewports, SetScissor, SetScissors,
    SetDepthRange, SetDepthRanges, SetBlendColor, SetStencilMask, SetDepthBias, SetPointSize, SetLineWidth,
    BindBlendState, BindChannelMaskState, BindColorState, BindDepthStencilState, BindMultisampleState, BindPolygonState,
    ClearColor, ClearColori, ClearColorui, ClearDepthStencil, CopyBufferToBuffer, CopyBufferToTexture,
    CopyTextureToBuffer, CopyTextureToTexture, UpdateUniformBuffer, Barrier, CallCommands, CopyCommands,
    FenceSync, WaitSync, SignalEvent, WaitEvent, ReportCounter, ResetCounter, SetTiledCacheAction,
    SetTiledCacheTileSize, SetConservativeRasterEnable, SetPrimitiveRestart, SetRenderEnable,
    SetCopyRowStride, SetCopyImageStride,
    SetStencilValueMask, SetStencilRef,
    SetDepthClamp, SetAlphaRef, SetSampleMask, SetRasterizerDiscard, SetDepthBounds,
    BindUniformBuffers, BindStorageBuffers, ClearTexture, ClearTexturei, ClearTextureui, ClearBuffer,
    DrawElementsBaseVertex, Ignored, DrawMeshTasks,
    PushGraphicsConstants, PushComputeConstants, PushGraphicsConstantsDeferred, PushComputeConstantsDeferred, SetRenderEnableConditional, ReportValue,
    BindTexturesDeferred, BindImagesDeferred, BindUniformBuffersDeferred, BindStorageBuffersDeferred,
    SetGeometryProcessingMode, DrawMeshTasksIndirect,
    QueuePresent, // (host: nvnQueuePresentTexture in queue order)
};

constexpr size_t WindowBuilderSize = 0x70;

// One recorded command: an opcode plus captured arguments (pointer arguments are copied at
// record time, as NVN only requires them to stay valid during the call).
struct Command {
    u32 op = 0;
    u64 x[8] = {};
    float f[8] = {};
    std::vector<u8> data;
};

struct CommandList {
    std::vector<Command> commands;
    std::vector<u32> calls; // indices of the CallCommands entries (set when recording ends)
    bool indexed = false;   // `calls` is complete
    mutable std::atomic<u32> submissions{0}; // (diagnostic, OOT_FLAG_TRACE: lists with flag decisions submitted again)
    bool has_flags = false;
};
// A texture copy/clear target: view plus NVNcopyRegion {x, y, z, width, height, depth}.
struct TextureCopy { TextureView view; s32 region[6]; };
struct RenderTargets { u64 colors[8]{}; TextureView color_views[8]{}; u64 depth{}; TextureView depth_view; };
bool ExecuteRenderCommand(const Command& command);
// False once a command since the last full draw may change what a draw resolves (render.cpp).
bool CurrentFrameSkipped(); // frame skip: this frame is neither rendered nor shown
bool FrameSkipEnabled();
inline thread_local bool fast_draw_allowed = false; // (per thread: draws are resolved on several, see batch.h resolve_worker)
inline u64 fast_draw_breakers[128]{}; // (OOT_SET_STATS) commands that ended a run of fast-path draws
inline u64 fast_draw_fails[8]{};
inline u64 slow_draw_why[136]{};       // (OOT_SET_STATS) why a draw took the full path: [op] the command that changed state, 128.. see render.cpp      // reasons TryFastDraw declined
// Before a command executes: whether the draw fast path (render.cpp) survives it.
inline void NoteFastDrawBreaker(const Command& command) {
    // Draw fast path (render.cpp): only these commands keep the previous draw's setup valid.
    switch ((Op)command.op) {
    case Op::DrawArrays: case Op::DrawElements: case Op::DrawArraysInstanced: case Op::DrawElementsInstanced:
    case Op::DrawElementsBaseVertex: case Op::BindUniformBuffer: case Op::BindUniformBuffers:
    case Op::BindVertexBuffer: case Op::BindVertexBuffers: case Op::BindVertexAttribState: case Op::BindVertexStreamState:
    case Op::SetViewport: case Op::SetViewports: case Op::SetScissor: case Op::SetScissors: case Op::SetDepthRange: case Op::SetDepthRanges:
    case Op::ReportCounter: case Op::ResetCounter: break;
    // These invalidate only when they change a value (their handlers in render.cpp check).
    case Op::BindProgram: case Op::BindTexture: case Op::SetRenderTargets: case Op::BindBlendState: case Op::BindColorState:
    case Op::BindPolygonState: case Op::BindDepthStencilState: case Op::SetLineWidth: case Op::BindStorageBuffer: case Op::BindChannelMaskState:
    case Op::SetPrimitiveRestart: case Op::SetPointSize: case Op::SetDepthClamp:
    case Op::SetDepthBounds: case Op::SetSampleMask: case Op::SetRasterizerDiscard: break;
    // Dynamic stencil state is emitted by Draw for both full and fast draws.
    case Op::SetStencilMask: case Op::SetStencilValueMask: case Op::SetStencilRef: break;
    case Op::SetBlendColor: case Op::SetDepthBias: break; // (dynamic state, set per draw)
    // No effect on what a draw resolves: render enable is evaluated per draw; barriers are
    // already implied by submission order here.
    case Op::SetRenderEnable: case Op::SetRenderEnableConditional: case Op::Barrier:
    case Op::SetTiledCacheAction: case Op::SetTiledCacheTileSize: case Op::SetAlphaRef:
    case Op::SetGeometryProcessingMode: break;
    default: if (fast_draw_allowed) ++fast_draw_breakers[std::min<u32>(command.op, 127)]; fast_draw_allowed = false;
    }
}
// Command-list segments whose draws are resolved on several threads at once and committed in order (render.cpp).
// Only segments of state changes, draws and counter reports qualify (commands.cpp, ScanSegment).
struct ParallelSegment { const CommandList* list; size_t begin, end; };
void ExecuteSegmentsParallel(const std::vector<ParallelSegment>& segments, const std::vector<u64>& report_addresses);
void ExecuteSegmentSerial(const CommandList& list, size_t begin, size_t end); // (such a segment, the usual way)
unsigned ParallelResolveThreads(); // helper threads (0: off)
bool ParallelDrawsUsable();        // now (no diagnostic mode that needs one draw at a time)
// Renderer state (bound shaders, targets, buffers): each queue has its own.
std::shared_ptr<void> SaveRenderState();
void RestoreRenderState(const std::shared_ptr<void>& saved);
void ResetRenderCounter(u32 counter);
bool SamplesRunOpen(); // (between a samples reset and its report: no submission of the executor's own may fall there)
void ReportRenderCounter(u32 counter, u64 address, bool flag_applied = false); // (flag_applied: its word was put to 0 when the command was recorded)
// GPU-written timestamp (and value) for the report at this point of the command stream; false:
// the host writes them when the batch completes. GuestTimestampNs: the guest's timestamp clock.
bool RecordReportTimestamp(u64 report_address, const u32* value);
u64 GuestTimestampNs();
extern u64 last_present_ns; // render.cpp

// Executes submitted command lists in order on the host queue.
class CommandExecutor {
public:
    void Submit(u64 queue, u64 handle);
    void Submit(u64 queue, std::shared_ptr<const CommandList> list);
    // A queue-level operation (FenceSync, WaitSync) in the queue's order.
    void QueueOperation(u64 queue, Op op, u64 argument, u64 argument2 = 0);
    void Resume(); // parked queues whose events are now satisfied run
    void Drain();  // waits until everything submitted was executed (parked queues excepted)
    void Flush();
    void Finish();
};
CommandExecutor& Executor();
bool TimelineOn(); double TimelineMs(); // (OOT_TIMELINE)
// A fence's sync was signalled (its batch retired): threads in nvnSyncWait look again.
void NotifySyncSignalled();
// Sleeps until some sync is signalled or `timeout` passes; true if `state` is signalled.
bool WaitSyncSignalled(const u32* state, std::chrono::nanoseconds timeout);
// True when a thread of their own executes submissions and retires batches (nothing for a waiting game thread to drive).
bool BackgroundGpuThreads();
extern u64 current_queue; // queue of the submission being executed

// Command list storage: a handle names a list until a new list is recorded at the same handle.
std::shared_ptr<const CommandList> LookupCommandList(u64 handle);
CommandList& Recording(u64 object);

void ReportSlowFrame(double frame_ms); // render.cpp
void UniformUpdatesBefore(u64 cpu, u64 size); // render.cpp (uniform overlays)
void ReportUniformOverlays();
bool FrameLogged(); // render.cpp: the current frame is being logged (OOT_DRAW_LOG*)
void NoteCpuEvent(u32* value);
bool IsCpuEvent(u32* value);
} // namespace nvn
