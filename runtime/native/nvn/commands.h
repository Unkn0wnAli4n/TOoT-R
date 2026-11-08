// Recorded NVN command lists and their execution on Vulkan.
#pragma once

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
};
// A texture copy/clear target: view plus NVNcopyRegion {x, y, z, width, height, depth}.
struct TextureCopy { TextureView view; s32 region[6]; };
struct RenderTargets { u64 colors[8]{}; TextureView color_views[8]{}; u64 depth{}; TextureView depth_view; };
bool ExecuteRenderCommand(const Command& command);
// False once a command since the last full draw may change what a draw resolves (render.cpp).
bool CurrentFrameSkipped(); // frame skip: this frame is neither rendered nor shown
bool FrameSkipEnabled();
inline bool fast_draw_allowed = false;
inline u64 fast_draw_breakers[128]{}; // (OOT_SET_STATS) commands that ended a run of fast-path draws
inline u64 fast_draw_fails[8]{};      // reasons TryFastDraw declined
// Renderer state (bound shaders, targets, buffers): each queue has its own.
std::shared_ptr<void> SaveRenderState();
void RestoreRenderState(const std::shared_ptr<void>& saved);
void ResetRenderCounter(u32 counter);
void ReportRenderCounter(u32 counter, u64 address);
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
extern u64 current_queue; // queue of the submission being executed

// Command list storage: a handle names a list until a new list is recorded at the same handle.
std::shared_ptr<const CommandList> LookupCommandList(u64 handle);
CommandList& Recording(u64 object);

void ReportSlowFrame(double frame_ms); // render.cpp
void UniformUpdatesBefore(u64 cpu, u64 size); // render.cpp (uniform overlays)
void ReportUniformOverlays();
bool FrameLogged(); // render.cpp: the current frame is being logged (OOT_DRAW_LOG*)
} // namespace nvn
