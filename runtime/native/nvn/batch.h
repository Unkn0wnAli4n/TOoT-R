// GPU work batching. Recorded work accumulates in one command buffer and is submitted when
// results are needed (end of a submitted NVN command list, CPU readback, presentation), instead
// of one submit-and-wait per draw.
#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <vector>

#include "vk.h"

namespace nvn {

// Records into the open batch (opening one if needed). Work runs in recording order.
void Immediate(const std::function<void(VkCommandBuffer)>& record, u64 profile_key = 0);
// What the next BeginWork orders (barrier statistics): 0 other, 1 compute dispatch, 2 render pass.
inline thread_local int work_kind = 0;
void ReportGpuProfile();
// Draws into a dynamic-rendering instance shared by consecutive draws with the same `key`
// (attachments and render area). A new instance starts (after a full barrier) when the key
// differs or none is open; `begin` records its start. Any other recorded work, EndPass or Flush
// ends the instance. Within an instance rasterization order keeps attachment accesses ordered.
// Whether the open batch has a render pass instance open with this key (PassDraw would reuse it).
bool PassOpenWithKey(const std::vector<u64>& key);
// What a draw accesses (textures by host object, guest memory ranges): a render pass whose first draw
// conflicts with nothing recorded since the last full barrier begins without one (no pipeline drain).
struct PassAccess {
    std::vector<const void*> read_textures, written_textures; // (attachments are written)
    std::vector<std::pair<u64, u64>> reads, writes;
    bool global_read = false, global_write = false;           // (shader accesses through pointers: anywhere)
};
void PassDraw(const std::vector<u64>& key, const std::function<void(VkCommandBuffer)>& begin,
              const std::function<void(VkCommandBuffer)>& draw, u64 profile_key = 0, const PassAccess* access = nullptr);
// Work recorded without a full barrier whose accesses are not described: the next pass orders after it.
void NoteUnorderedWork();
void EndPass();
// Records work that carries its own narrow barriers (the item it depends on and what depends on it
// are ordered by `record` itself): no full barrier before it, and reopening the same render pass
// afterwards needs only an attachment barrier.
void ImmediateLight(const std::function<void(VkCommandBuffer)>& record);
// Changes whenever any work is recorded or the batch is submitted.
u64 RecordSerial();
// Changes whenever anything other than a render-pass draw is recorded, or a batch is submitted:
// state a draw set on the command buffer is still in effect while it is unchanged.
u64 OtherWorkSerial();
// Records `record` with no barrier before it, only if nothing was recorded since RecordSerial()
// returned `serial` and no render pass is open. The caller guarantees the work has no hazard
// against what was recorded since the last full barrier. False: nothing was recorded.
bool ImmediateUnordered(const std::function<void(VkCommandBuffer)>& record, u64 serial);
// True while a PassDraw begin callback reopens attachments after only light work.
bool ReopeningAfterLightWork();
// Records into the open render pass instance without ending it; false (nothing recorded) if none.
bool RecordInPass(const std::function<void(VkCommandBuffer)>& record);
// Runs (recording into the pass) just before the open pass instance ends; replaced by each call.
void SetPassEndHook(std::function<void(VkCommandBuffer)> hook);
// Runs before an open batch is submitted (may record more work).
void SetPreSubmitHook(std::function<void()> hook);
// Submits the open batch, waits for all submitted work, then runs the deferred actions in order.
void Flush();
// Submits the open batch without waiting (completed batches retire as later calls find them done).
void Submit();
// Number of batches submitted so far; WaitForBatches(n) waits until the first n have completed.
u64 SubmittedBatches();
void WaitForBatches(u64 count);
// Retires batches that have completed (non-blocking).
void Poll();
// Retires (waiting) and submits as needed until `done` holds or no work remains.
void WaitUntil(const std::function<bool()>& done);
// Runs after the work recorded so far completes (CPU-visible signals, resource release).
void AfterFlush(std::function<void()> action);
// Defer until the next flush even when resources are prepared before recording GPU work.
void AfterBatch(std::function<void()> action);
// A descriptor set for `layout`, valid until the next flush.
VkDescriptorSet AllocateBatchSet(VkDescriptorSetLayout layout);
// Held while a command list executes so batches from other threads do not interleave.
// (recursive; its depth tells whether a GPU wait may release it: only an outermost holder does)
// The executor holds it for whole command lists (milliseconds), and took it again at once: the game's thread, which
// needs it for a moment at each readback, submit and frame pacing call, stood behind it for most of a frame (16 ms of
// 30 in Castle Town). Threads that find it taken count as waiters, and the executor lets them in between commands
// (Yield). OOT_NO_BATCH_YIELD=1: as before.
struct BatchLock {
    std::recursive_mutex m;
    int depth = 0; // (only read and written by the holder)
    std::atomic<int> waiters{0};
    void lock() { if (!m.try_lock()) { waiters.fetch_add(1, std::memory_order_relaxed); m.lock(); waiters.fetch_sub(1, std::memory_order_relaxed); } ++depth; }
    // (waits without asking the executor to yield: it takes the lock between command lists, as before Yield)
    void lock_patient() { m.lock(); ++depth; }
    bool try_lock() { if (!m.try_lock()) return false; ++depth; return true; }
    void unlock() { --depth; m.unlock(); }
    // (the outermost holder, between two pieces of its work) Threads waiting for the lock take it first.
    void Yield();
};
BatchLock& BatchMutex();
// Write-back copies of GPU storage writes (memory.cpp): waits until the copy of batch `serial` is done.
void WaitWriteback(u64 serial);
VkSemaphore WritebackTimeline();
// Guest memory written by work recorded in the open batch (reports, buffer copies/fills/updates,
// shader storage writes). CPU reads of such memory need a Flush first; others do not.
void NotePendingWrite(u64 address, u64 size, bool write_back = true); // write_back: copy GPU result to guest memory
// A GPU write whose data CPU readers already see (e.g. uniform updates, kept in an overlay): only
// copied back to guest memory, never waited for.
void NoteWriteBack(u64 address, u64 size);
// Guest time blocked on the GPU (per-frame report): 0 nvnSyncWait, 1 readbacks (invalidates), 2 nvnQueueFinish.
extern std::atomic<u64> cpu_wait_ns[3];
void DropDeviceAhead(u64 address, u64 size); // (a pool goes)
bool HasPendingWrite(u64 address, u64 size);
bool PendingWriteRange(u64 address, u64 size, u64* start, u64* end); // the pending range hit (diagnostics)
// An invalidate: GPU work writing [address, address + size), and its write-back, completes first.
void WaitForWrites(u64 address, u64 size);
// Counter reports (occlusion results) pending in the open batch, tracked separately: a predicate
// waiting only on those can be treated as "visible" instead of stalling for the result.
void NotePendingReport(u64 address);
bool HasPendingReport(u64 address, u64 size);
// Batches submitted so far (diagnostics).
u64 FlushCount();
void ReportBatchStats();
// Total GPU execution time of submitted batches (first to last command), in nanoseconds.
u64 GpuBusyNanoseconds();
u64 FenceWaitNanoseconds(); // CPU time spent waiting for batches
u64 WritebackNanoseconds(); // GPU time of the batches' final write-back copies

} // namespace nvn
