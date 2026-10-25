//
// plume
//
// Copyright (c) 2024 renderbag and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file for details.
//

// A RenderInterface that runs another one on a single thread, for a backend whose every call has to come from the
// thread that owns the device: WebGPU in a browser, where the device is a JavaScript object that can't leave the
// worker (or the page) that made it. Any number of threads may call it, and the calls reach the backend on the device
// thread in the order they were made.
//
// - A call that returns something (create*(), map(), a swap chain's resize(), ...) runs on the device thread while the
//   caller waits. map() of a buffer whose data the backend delivers asynchronously waits on the calling side until
//   isMapReady(), so the device thread keeps returning to its event loop meanwhile, and so does
//   createGraphicsPipeline() for a pipeline the backend builds asynchronously, until its isReady().
// - createTexture() and an upload buffer's map() don't wait: the texture is made on the device thread after whatever
//   is queued before, and a caller that needs its backend object first (recording it in a command list) waits then; an
//   upload buffer is mapped at memory of its own on the calling side, and unmap() queues a copy of what was written
//   for the backend's buffer (and an execution one of each buffer left mapped), so the caller can write again at once.
//   Not one of 16 MB or more (an image's or a texture pack's, mapped now and then): that memory would be a second copy
//   of the buffer in the heap beside the backend's, so it maps the backend's on the device thread as other buffers do.
// - A READBACK buffer's isMapReady() doesn't wait for the device thread: after a copy into the buffer is replayed, the
//   device thread checks it at each replay and present until the data is there, maps it and publishes the answer,
//   which isMapReady() reads. Its map() waits for that answer, not behind everything queued on the device thread.
// - A call that returns nothing (unmap(), setName(), descriptor writes, executeCommandLists(), present(), destroying
//   an object) is queued and the caller goes on. The queue is first in, first out, so whatever happened before a call
//   on the calling side, under the caller's own locks or on another thread it waited for, reaches the backend first.
// - A command list records on the calling thread, copying whatever its arguments point at, and is replayed into a
//   backend command list on the device thread when it's executed. Recording never waits for the device thread.
//   So an object deleted meanwhile is deleted on the device thread only once every command list that had recorded
//   something by the delete has been executed, begun again or deleted: a list never replays without an object it
//   recorded, or submits a texture already destroyed (WebGPU drops that command buffer). A game's hi-res texture pack
//   reloading a texture during a frame did that.
// - waitForCommandFence() waits for the device thread to replay the executions that signal the fence up to the last
//   wait for it, not the one just queued, so a caller records its next list while the device thread replays its last.
//   That is enough on a backend whose fences are no-ops (WebGPU's): a command list, an upload buffer's memory and a
//   READBACK buffer are safe to reuse without it (above), and an object is deleted only after the lists that used it.
// - A swap chain hands out textures of its own, and present() copies the one drawn into the backend's swap chain
//   texture: acquire, copy and present run as one task on the device thread. A browser shows the canvas between two
//   tasks of its event loop whenever it updates the rendering, so a canvas texture acquired in one task can be gone
//   by the next. That costs a copy of the window each present. wait() waits on the calling thread where the backend
//   can tell it there (waitOnCaller()), and otherwise polls the backend's isWaitReady() from it, as map() does
//   isMapReady(). needsResize(), getRefreshRate() and isHidden() don't wait for the device thread:
//   they answer what the backend said at the last present, resize() or refresh, and queue a refresh.
// - On the device thread itself, everything runs in place, after whatever is queued.
// - Raytracing and acceleration structures aren't forwarded: WebGPU has neither.
//
// A DeviceThread is how calls get there. CreateEmscriptenDeviceThread() proxies them with Emscripten's proxying queue
// to a thread that returns to its event loop, such as the page's main thread with the program itself in pthreads
// (-sPROXY_TO_PTHREAD, or threads it starts). Anything that runs the queued work in order on one thread will do.

#pragma once

#include "plume_render_interface.h"

#include <functional>

#if defined(__EMSCRIPTEN_PTHREADS__)
#include <pthread.h>
#endif

namespace plume {
    struct DeviceThread {
        virtual ~DeviceThread() { }

        // Runs work on the device thread after everything queued before it, and returns once it has run.
        virtual void runSync(const std::function<void()> &work) = 0;

        // Queues work to run on the device thread after everything queued before it, and returns.
        virtual void runAsync(std::function<void()> &&work) = 0;

        // Whether the calling thread is the device thread.
        virtual bool isCurrent() const = 0;
    };

    // Creates the backend on the device thread with createInterface, and returns an interface that runs it there.
    extern std::unique_ptr<RenderInterface> CreateDeviceThreadInterface(std::shared_ptr<DeviceThread> deviceThread, const std::function<std::unique_ptr<RenderInterface>()> &createInterface);

#if defined(__EMSCRIPTEN_PTHREADS__)
    // The work runs on thread when it returns to its event loop. Use emscripten_main_runtime_thread_id() for the page's
    // main thread, which is where a canvas and a device requested by the page are.
    extern std::shared_ptr<DeviceThread> CreateEmscriptenDeviceThread(pthread_t thread);
#endif
};
