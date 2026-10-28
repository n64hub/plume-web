//
// plume
//
// Copyright (c) 2024 renderbag and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file for details.
//

// See plume_device_thread.h for what runs where.

#include "plume_device_thread.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <set>
#include <thread>
#include <type_traits>

#if defined(__EMSCRIPTEN_PTHREADS__)
#include <emscripten/proxying.h>
#include "plume_web_waits.h"
#endif

namespace plume {
    // Runs f on the device thread and returns what it returned.
    template<typename F>
    static auto runSync(DeviceThread *thread, F &&f) -> decltype(f()) {
        using Result = decltype(f());
        if constexpr (std::is_void_v<Result>) {
            thread->runSync(f);
        }
        else {
            Result result{};
            thread->runSync([&]() { result = f(); });
            return result;
        }
    }

    // A command list reaches the backend only when it's executed, so an object deleted while a list that recorded it
    // waits to be executed would be gone when the list is replayed: a texture destroyed before the list's bind groups
    // are submitted, which WebGPU answers by dropping the command buffer. So deleting a backend object waits until every
    // command list that had recorded something by then has been executed, begun again or deleted: the lists that could
    // have recorded it. One that starts recording after can't have, and one begun and left empty holds nothing back.
    struct HeldDestroys {
        std::mutex mutex;
        uint64_t openCount = 0;
        std::multiset<uint64_t> open;
        std::deque<std::pair<uint64_t, std::function<void()>>> held;
    };

    static HeldDestroys heldDestroys;

    // Takes the destroys that no open list was opened before. With the mutex held.
    static std::vector<std::function<void()>> takeReleasedDestroys() {
        std::vector<std::function<void()>> released;
        const uint64_t firstOpen = heldDestroys.open.empty() ? UINT64_MAX : *heldDestroys.open.begin();
        while (!heldDestroys.held.empty() && (heldDestroys.held.front().first < firstOpen)) {
            released.emplace_back(std::move(heldDestroys.held.front().second));
            heldDestroys.held.pop_front();
        }

        return released;
    }

    // A command list is open from its first command until it's executed, begun again or deleted. openSerial is 0 when
    // it isn't.
    static void setCommandListOpen(uint64_t &openSerial, bool open) {
        std::vector<std::function<void()>> released;
        {
            std::lock_guard lock(heldDestroys.mutex);
            if (openSerial != 0) {
                heldDestroys.open.erase(heldDestroys.open.find(openSerial));
                openSerial = 0;
            }

            if (open) {
                openSerial = ++heldDestroys.openCount;
                heldDestroys.open.insert(openSerial);
            }

            released = takeReleasedDestroys();
        }

        for (std::function<void()> &destroy : released) {
            destroy();
        }
    }

    // Deletes a backend object on the device thread, after the command lists that could have recorded it are executed.
    template<typename T>
    static void destroyAsync(DeviceThread *thread, T *object) {
        if (object == nullptr) {
            return;
        }

        std::function<void()> destroy = [thread, object]() { thread->runAsync([object]() { delete object; }); };
        {
            std::lock_guard lock(heldDestroys.mutex);
            if (!heldDestroys.open.empty()) {
                heldDestroys.held.emplace_back(heldDestroys.openCount, std::move(destroy));
                return;
            }
        }

        destroy();
    }

    struct DTDevice;
    struct DTCommandQueue;
    struct DTPipelineLayout;

    struct DTBufferFormattedView : RenderBufferFormattedView {
        DeviceThread *thread = nullptr;
        RenderBufferFormattedView *inner = nullptr;

        DTBufferFormattedView(DeviceThread *thread, std::unique_ptr<RenderBufferFormattedView> inner) : thread(thread), inner(inner.release()) { }
        ~DTBufferFormattedView() override { destroyAsync(thread, inner); }
    };

    // Whether a READBACK buffer's data has arrived, answered on the device thread and read on any thread without waiting
    // for it. A question that waited for the answer waited for the device thread to get through everything queued
    // before it, the replay of every command list executed meanwhile, and a thread asking between other work (RT64 at
    // each VI, whether a frame's copy to RAM can land) spent milliseconds a frame there. So the device thread answers on
    // its own: after a command list that copies into the buffer is replayed, it checks the buffer at every replay and
    // present until the data is there, maps it then, and publishes the answer with the pointer. isMapReady() reads the
    // answer, and it counts only for the executions queued: one from before the last execution that copies into the
    // buffer was replayed is about the data before it. A caller that finds no answer asks once more, for a device thread
    // with nothing else to do.
    struct DTMapState : std::enable_shared_from_this<DTMapState> {
        // On the device thread: the backend's buffer (null once deleted), whether it's mapped, and the executions that
        // copy into it replayed.
        RenderBuffer *inner = nullptr;
        bool mapped = false;
        uint64_t replayed = 0;

        // The executions queued that copy into the buffer, and the last answer: the executions replayed before it,
        // shifted left once, and 1 if the data had arrived. The data's pointer when it had.
        std::atomic<uint64_t> executions = 0;
        std::atomic<uint64_t> answer = 0;
        std::atomic<bool> asking = false;
        std::atomic<void *> data = nullptr;

        // Each answer wakes a map() waiting for one.
        std::mutex answerMutex;
        std::condition_variable answerCondition;
        std::atomic<uint64_t> answerCount = 0;

        bool isReady() const {
            return answer == ((executions << 1) | 1);
        }

        void ask(DeviceThread *thread);
        bool check();

        // On the device thread.
        void map() {
            if (!mapped && (inner != nullptr)) {
                data = inner->map(0, nullptr);
                mapped = true;
            }
        }

        // On the device thread.
        void unmap(uint32_t subresource, const RenderRange *writtenRange) {
            if (mapped && (inner != nullptr)) {
                inner->unmap(subresource, writtenRange);
            }

            mapped = false;
        }

        // Waits for an answer after the one counted, or a millisecond. Where a timed wait ends late (plume_web_waits.h),
        // it spins out the millisecond instead: Firefox's GPU process calls a map back only after a submission, which is
        // what each question makes (isMapReady()), so a waiter that slept between questions asked one every 15.6 ms and
        // its data arrived a tick or two late.
        void waitForAnswer(uint64_t counted) {
#if defined(__EMSCRIPTEN_PTHREADS__)
            if (timedWaitsEndLate()) {
                const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
                while ((answerCount == counted) && ((std::chrono::steady_clock::now() - start) < std::chrono::milliseconds(1))) {
                }

                return;
            }
#endif
            std::unique_lock lock(answerMutex);
            answerCondition.wait_for(lock, std::chrono::milliseconds(1), [&]() { return answerCount != counted; });
        }

        uint64_t countAnswers() {
            return answerCount;
        }
    };

    // On a device thread: the READBACK buffers whose data hasn't arrived since their last copy was replayed.
    static thread_local std::vector<std::shared_ptr<DTMapState>> pendingReadbacks;

    static thread_local bool checkingReadbacks = false;

    static void checkPendingReadbacks() {
        // The backend's readback callback can't come from inside a check (its callbacks come from its event loop), but if
        // it did, the check in progress answers them all anyway.
        if (checkingReadbacks) {
            return;
        }

        checkingReadbacks = true;
        pendingReadbacks.erase(std::remove_if(pendingReadbacks.begin(), pendingReadbacks.end(), [](const std::shared_ptr<DTMapState> &state) { return state->check(); }), pendingReadbacks.end());
        checkingReadbacks = false;
    }

    void DTMapState::ask(DeviceThread *thread) {
        bool idle = false;
        if (asking.compare_exchange_strong(idle, true)) {
            thread->runAsync([state = shared_from_this()]() {
                state->asking = false;
                if (std::find(pendingReadbacks.begin(), pendingReadbacks.end(), state) == pendingReadbacks.end()) {
                    state->check();
                }

                checkPendingReadbacks();
            });
        }
    }

    // On the device thread: answers whether the data the replayed copies wrote has arrived, and maps the buffer if it
    // has. Returns whether the buffer needs no more checks.
    bool DTMapState::check() {
        const bool ready = (inner != nullptr) && inner->isMapReady();
        if (ready) {
            map();
        }

        answer = (replayed << 1) | (ready ? 1 : 0);
        {
            std::lock_guard lock(answerMutex);
            answerCount++;
        }

        answerCondition.notify_all();
        return ready || (inner == nullptr);
    }

    // Deletes a READBACK buffer's backend buffer, on the device thread, and leaves its state without one.
    struct DTMapRelease {
        std::shared_ptr<DTMapState> state;

        ~DTMapRelease() {
            state->unmap(0, nullptr);
            delete state->inner;
            state->inner = nullptr;
        }
    };

    // The upload buffers mapped on the calling side and not unmapped yet. Before each execution what they hold is copied
    // and written to the backend's, as a backend writes a buffer left mapped at each submission.
    struct DTBuffer;
    static std::mutex mappedUploadsMutex;
    static std::vector<DTBuffer *> mappedUploads;

    static void setUploadMapped(DTBuffer *buffer, bool mapped) {
        std::scoped_lock lock(mappedUploadsMutex);
        auto it = std::find(mappedUploads.begin(), mappedUploads.end(), buffer);
        if (mapped && (it == mappedUploads.end())) {
            mappedUploads.emplace_back(buffer);
        }
        else if (!mapped && (it != mappedUploads.end())) {
            mappedUploads.erase(it);
        }
    }

    struct DTBuffer : RenderBuffer {
        DeviceThread *thread = nullptr;
        RenderBuffer *inner = nullptr;
        std::shared_ptr<DTMapState> mapState;
        bool uploadHeap = false;
        uint64_t size = 0;

        // An upload buffer's memory on the calling side (map()), for one smaller than CopiedUploadMaxSize.
        static constexpr uint64_t CopiedUploadMaxSize = 16 * 1024 * 1024;
        std::unique_ptr<uint8_t[]> uploadCopy;
        std::once_flag uploadCopyMade;

        DTBuffer(DeviceThread *thread, std::unique_ptr<RenderBuffer> inner, const RenderBufferDesc &desc) : thread(thread), inner(inner.release()), size(desc.size) {
            uploadHeap = (desc.heapType == RenderHeapType::UPLOAD) || (desc.heapType == RenderHeapType::GPU_UPLOAD);
            if (desc.heapType == RenderHeapType::READBACK) {
                mapState = std::make_shared<DTMapState>();
                mapState->inner = this->inner;
            }
        }

        ~DTBuffer() override {
            if (uploadHeap) {
                setUploadMapped(this, false);
            }

            if (mapState != nullptr) {
                destroyAsync(thread, new DTMapRelease{ mapState });
            }
            else {
                destroyAsync(thread, inner);
            }
        }

        // Data the backend delivers asynchronously (WebGPU's READBACK buffers) arrives while the device thread is back
        // in its event loop, so this thread waits for the device thread's answer rather than holding the device thread.
        // It asks at most once a millisecond: the device thread answers at every replay and present by itself, and a
        // question asked again as soon as the last answer said no had this thread asking as fast as the device thread
        // answered, about 10 times a millisecond, each answer a look at the queue. In Chrome that filled the GPU
        // process's main thread with as many tasks for as long as the data took, which held up the data and the page's
        // presents with it: RT64's wait for a framebuffer's change count (Pilotwings 64's copies between framebuffers)
        // made frame gaps of 100-130 ms. On the device thread itself, the backend's map() is all there is.
        void *map(uint32_t subresource, const RenderRange *readRange) override {
            if (mapState != nullptr) {
                if (thread->isCurrent()) {
                    mapState->map();
                    return mapState->data;
                }

                std::chrono::steady_clock::time_point asked;
                for (;;) {
                    const uint64_t counted = mapState->countAnswers();
                    if (mapState->isReady()) {
                        return mapState->data;
                    }

                    const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
                    if ((now - asked) >= std::chrono::milliseconds(1)) {
                        asked = now;
                        mapState->ask(thread);
                    }

                    mapState->waitForAnswer(counted);
                }
            }

            // An upload buffer is mapped at memory of its own on the calling side, the same every time, without waiting
            // for the device thread (RT64 maps ~30 a frame, 1,700 a second in Pilotwings 64's menus, and each map that
            // waited was 0.1-0.2 ms behind the page's thread). unmap() copies what was written and queues the copy for
            // the backend's buffer, so the caller can write the next data at once, before the device thread has got to
            // the last: waitForCommandFence() doesn't wait for it. Not one of CopiedUploadMaxSize or more: that memory is a
            // second copy of the buffer in the heap beside the backend's, and RecompFrontend's for its images (1.5 times
            // the largest yet, an SVG drawn at 4K among them) took GoldenEye's page past its heap at boot. One that big
            // is mapped now and then, so it maps the backend's on the device thread below, which runs every unmap queued
            // before, and unmap() is queued: the caller writes into the backend's memory once the last data went out.
            if (uploadHeap && (subresource == 0) && (size < CopiedUploadMaxSize)) {
                std::call_once(uploadCopyMade, [&]() { uploadCopy.reset(new uint8_t[size]()); });
                setUploadMapped(this, true);
                return uploadCopy.get();
            }

            const std::thread::id caller = std::this_thread::get_id();
            for (;;) {
                void *data = nullptr;
                const bool ready = runSync(thread, [&]() {
                    if ((std::this_thread::get_id() != caller) && !inner->isMapReady()) {
                        return false;
                    }

                    data = inner->map(subresource, readRange);
                    return true;
                });

                if (ready) {
                    return data;
                }

                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }

        bool isMapReady() override {
            if (mapState != nullptr) {
                if (mapState->isReady()) {
                    return true;
                }

                mapState->ask(thread);
                return false;
            }

            return runSync(thread, [&]() { return inner->isMapReady(); });
        }

        void unmap(uint32_t subresource, const RenderRange *writtenRange) override {
            RenderBuffer *buffer = inner;
            const bool ranged = (writtenRange != nullptr);
            const RenderRange range = ranged ? *writtenRange : RenderRange();
            if (uploadHeap && (subresource == 0) && (uploadCopy != nullptr)) {
                setUploadMapped(this, false);
                const uint64_t begin = ranged ? std::min(range.begin, size) : 0;
                const uint64_t end = ranged ? std::min(std::max(range.end, begin), size) : size;
                thread->runAsync([buffer, begin, bytes = std::vector<uint8_t>(uploadCopy.get() + begin, uploadCopy.get() + end), ranged, range]() {
                    uint8_t *data = static_cast<uint8_t *>(buffer->map(0, nullptr));
                    memcpy(data + begin, bytes.data(), bytes.size());
                    buffer->unmap(0, ranged ? &range : nullptr);
                });
            }
            else if (mapState != nullptr) {
                thread->runAsync([state = mapState, subresource, ranged, range]() { state->unmap(subresource, ranged ? &range : nullptr); });
            }
            else {
                thread->runAsync([=]() { buffer->unmap(subresource, ranged ? &range : nullptr); });
            }
        }

        std::unique_ptr<RenderBufferFormattedView> createBufferFormattedView(RenderFormat format) override {
            std::unique_ptr<RenderBufferFormattedView> view = runSync(thread, [&]() { return inner->createBufferFormattedView(format); });
            return view ? std::make_unique<DTBufferFormattedView>(thread, std::move(view)) : nullptr;
        }

        void setName(const std::string &name) override {
            thread->runAsync([buffer = inner, name]() { buffer->setName(name); });
        }

        uint64_t getDeviceAddress() const override {
            return runSync(thread, [&]() { return inner->getDeviceAddress(); });
        }
    };

    struct DTTextureView : RenderTextureView {
        DeviceThread *thread = nullptr;
        RenderTextureView *inner = nullptr;

        DTTextureView(DeviceThread *thread, std::unique_ptr<RenderTextureView> inner) : thread(thread), inner(inner.release()) { }
        ~DTTextureView() override { destroyAsync(thread, inner); }
    };

    // A texture's backend object, which the device thread fills in when it makes it. A caller that needs it before then
    // waits (unwrap()), and a task queued meanwhile reads it when it runs, after the texture was made.
    struct DTTextureSlot {
        RenderTexture *texture = nullptr;
        std::atomic<bool> made = false;
    };

    static std::mutex texturesMadeMutex;
    static std::condition_variable texturesMade;

    // Deletes a texture's backend object, on the device thread.
    struct DTTextureRelease {
        std::shared_ptr<DTTextureSlot> slot;

        ~DTTextureRelease() {
            delete slot->texture;
            slot->texture = nullptr;
        }
    };

    struct DTTexture : RenderTexture {
        DeviceThread *thread = nullptr;
        std::shared_ptr<DTTextureSlot> slot = std::make_shared<DTTextureSlot>();
        bool ownsInner = true;

        // One the swap chain fills in, on the device thread.
        DTTexture(DeviceThread *thread) : thread(thread), ownsInner(false) {
            slot->made = true;
        }

        // Makes the texture on the device thread, after whatever is queued before, and returns without waiting. RT64's
        // texture cache makes two for each texture the game loads and waited for each: a screen with hundreds of new
        // ones (a menu in Pilotwings 64) held the game's display lists 110-170 ms, in Chrome and Firefox alike.
        DTTexture(DeviceThread *thread, std::function<std::unique_ptr<RenderTexture>()> &&make) : thread(thread) {
            thread->runAsync([slot = slot, make = std::move(make)]() {
                slot->texture = make().release();
                {
                    std::lock_guard lock(texturesMadeMutex);
                    slot->made.store(true, std::memory_order_release);
                }

                texturesMade.notify_all();
            });
        }

        ~DTTexture() override {
            if (ownsInner) {
                destroyAsync(thread, new DTTextureRelease{ std::move(slot) });
            }
        }

        bool isMade() const {
            return slot->made.load(std::memory_order_acquire);
        }

        // The backend object, once the device thread has made it.
        RenderTexture *get() const {
            if (!isMade()) {
                if (thread->isCurrent()) {
                    // Runs what is queued, which makes it.
                    thread->runSync([]() { });
                }
                else {
                    std::unique_lock lock(texturesMadeMutex);
                    texturesMade.wait(lock, [&]() { return isMade(); });
                }
            }

            return slot->texture;
        }

        std::unique_ptr<RenderTextureView> createTextureView(const RenderTextureViewDesc &desc) const override {
            std::unique_ptr<RenderTextureView> view = runSync(thread, [&]() { return slot->texture->createTextureView(desc); });
            return view ? std::make_unique<DTTextureView>(thread, std::move(view)) : nullptr;
        }

        void setName(const std::string &name) override {
            thread->runAsync([slot = slot, name]() {
                if (slot->texture != nullptr) {
                    slot->texture->setName(name);
                }
            });
        }
    };

    struct DTShader : RenderShader {
        DeviceThread *thread = nullptr;
        RenderShader *inner = nullptr;

        DTShader(DeviceThread *thread, std::unique_ptr<RenderShader> inner) : thread(thread), inner(inner.release()) { }
        ~DTShader() override { destroyAsync(thread, inner); }

        void setName(const std::string &name) override {
            thread->runAsync([shader = inner, name]() { shader->setName(name); });
        }
    };

    struct DTSampler : RenderSampler {
        DeviceThread *thread = nullptr;
        RenderSampler *inner = nullptr;

        DTSampler(DeviceThread *thread, std::unique_ptr<RenderSampler> inner) : thread(thread), inner(inner.release()) { }
        ~DTSampler() override { destroyAsync(thread, inner); }
    };

    struct DTPipeline : RenderPipeline {
        DeviceThread *thread = nullptr;
        RenderPipeline *inner = nullptr;

        DTPipeline(DeviceThread *thread, std::unique_ptr<RenderPipeline> inner) : thread(thread), inner(inner.release()) { }
        ~DTPipeline() override { destroyAsync(thread, inner); }

        void setName(const std::string &name) override {
            thread->runAsync([pipeline = inner, name]() { pipeline->setName(name); });
        }

        RenderPipelineProgram getProgram(const std::string &name) const override {
            return runSync(thread, [&]() { return inner->getProgram(name); });
        }

        bool isReady() override {
            return runSync(thread, [&]() { return inner->isReady(); });
        }

        // A pipeline the backend builds asynchronously (WebGPU's) is finished while the device thread is back in its
        // event loop, so this thread polls isReady() between calls rather than holding the device thread. The builds
        // take hundreds of milliseconds, so it polls rarely. On the device thread itself there is nothing to wait for.
        void waitUntilReady() {
            const std::thread::id caller = std::this_thread::get_id();
            while (!runSync(thread, [&]() { return (std::this_thread::get_id() == caller) || inner->isReady(); })) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    };

    struct DTPipelineLayout : RenderPipelineLayout {
        DeviceThread *thread = nullptr;
        RenderPipelineLayout *inner = nullptr;

        // For the size of a push constant write that passes 0.
        std::vector<RenderPushConstantRange> pushConstantRanges;

        DTPipelineLayout(DeviceThread *thread, std::unique_ptr<RenderPipelineLayout> inner, const RenderPipelineLayoutDesc &desc) : thread(thread), inner(inner.release()) {
            pushConstantRanges.assign(desc.pushConstantRanges, desc.pushConstantRanges + desc.pushConstantRangesCount);
        }

        ~DTPipelineLayout() override { destroyAsync(thread, inner); }
    };

    // The executions that signal a fence: those queued, those replayed on the device thread, and those the last
    // waitForCommandFence() came after.
    struct DTFenceState {
        std::mutex mutex;
        std::condition_variable replayedCondition;
        uint64_t queued = 0;
        uint64_t replayed = 0;
        uint64_t waited = 0;
    };

    struct DTCommandFence : RenderCommandFence {
        DeviceThread *thread = nullptr;
        RenderCommandFence *inner = nullptr;
        std::shared_ptr<DTFenceState> state = std::make_shared<DTFenceState>();

        DTCommandFence(DeviceThread *thread, std::unique_ptr<RenderCommandFence> inner) : thread(thread), inner(inner.release()) { }
        ~DTCommandFence() override { destroyAsync(thread, inner); }
    };

    struct DTCommandSemaphore : RenderCommandSemaphore {
        DeviceThread *thread = nullptr;
        RenderCommandSemaphore *inner = nullptr;

        DTCommandSemaphore(DeviceThread *thread, std::unique_ptr<RenderCommandSemaphore> inner) : thread(thread), inner(inner.release()) { }
        ~DTCommandSemaphore() override { destroyAsync(thread, inner); }
    };

    struct DTFramebuffer : RenderFramebuffer {
        DeviceThread *thread = nullptr;
        RenderFramebuffer *inner = nullptr;
        uint32_t width = 0;
        uint32_t height = 0;

        DTFramebuffer(DeviceThread *thread, std::unique_ptr<RenderFramebuffer> inner) : thread(thread), inner(inner.release()) {
            width = this->inner->getWidth();
            height = this->inner->getHeight();
        }

        ~DTFramebuffer() override { destroyAsync(thread, inner); }
        uint32_t getWidth() const override { return width; }
        uint32_t getHeight() const override { return height; }
    };

    struct DTQueryPool : RenderQueryPool {
        DeviceThread *thread = nullptr;
        RenderQueryPool *inner = nullptr;
        uint32_t count = 0;

        DTQueryPool(DeviceThread *thread, std::unique_ptr<RenderQueryPool> inner) : thread(thread), inner(inner.release()) {
            count = this->inner->getCount();
        }

        ~DTQueryPool() override { destroyAsync(thread, inner); }

        void queryResults() override {
            runSync(thread, [&]() { inner->queryResults(); });
        }

        const uint64_t *getResults() const override {
            return runSync(thread, [&]() { return inner->getResults(); });
        }

        uint32_t getCount() const override {
            return count;
        }
    };

    // The backend object behind a wrapper.

    template<typename Wrapper, typename T>
    static auto unwrapAs(T *object) -> decltype(static_cast<Wrapper *>(object)->inner) {
        return (object != nullptr) ? static_cast<Wrapper *>(object)->inner : nullptr;
    }

    template<typename Wrapper, typename T>
    static auto unwrapAs(const T *object) -> decltype(static_cast<const Wrapper *>(object)->inner) {
        return (object != nullptr) ? static_cast<const Wrapper *>(object)->inner : nullptr;
    }

    static RenderBuffer *unwrap(const RenderBuffer *object) { return unwrapAs<DTBuffer>(object); }
    static RenderBufferFormattedView *unwrap(const RenderBufferFormattedView *object) { return unwrapAs<DTBufferFormattedView>(object); }
    static RenderTexture *unwrap(const RenderTexture *object) { return (object != nullptr) ? static_cast<const DTTexture *>(object)->get() : nullptr; }
    static RenderTextureView *unwrap(const RenderTextureView *object) { return unwrapAs<DTTextureView>(object); }
    static RenderShader *unwrap(const RenderShader *object) { return unwrapAs<DTShader>(object); }
    static RenderSampler *unwrap(const RenderSampler *object) { return unwrapAs<DTSampler>(object); }
    static RenderPipeline *unwrap(const RenderPipeline *object) { return unwrapAs<DTPipeline>(object); }
    static RenderPipelineLayout *unwrap(const RenderPipelineLayout *object) { return unwrapAs<DTPipelineLayout>(object); }
    static RenderCommandFence *unwrap(const RenderCommandFence *object) { return unwrapAs<DTCommandFence>(object); }
    static RenderCommandSemaphore *unwrap(const RenderCommandSemaphore *object) { return unwrapAs<DTCommandSemaphore>(object); }
    static RenderFramebuffer *unwrap(const RenderFramebuffer *object) { return unwrapAs<DTFramebuffer>(object); }
    static RenderQueryPool *unwrap(const RenderQueryPool *object) { return unwrapAs<DTQueryPool>(object); }

    static RenderBufferReference unwrap(const RenderBufferReference &reference) {
        return RenderBufferReference(unwrap(reference.ref), reference.offset);
    }

    static std::vector<RenderCommandSemaphore *> unwrap(RenderCommandSemaphore **semaphores, uint32_t count) {
        std::vector<RenderCommandSemaphore *> unwrapped(count);
        for (uint32_t i = 0; i < count; i++) {
            unwrapped[i] = unwrap(semaphores[i]);
        }

        return unwrapped;
    }

    // A descriptor set layout with its immutable samplers swapped for the backend's.
    struct UnwrappedDescriptorSetDesc {
        RenderDescriptorSetDesc desc;
        std::vector<RenderDescriptorRange> ranges;
        std::vector<std::vector<const RenderSampler *>> samplers;

        UnwrappedDescriptorSetDesc(const RenderDescriptorSetDesc &source) : desc(source) {
            ranges.assign(source.descriptorRanges, source.descriptorRanges + source.descriptorRangesCount);
            samplers.resize(ranges.size());
            for (size_t i = 0; i < ranges.size(); i++) {
                if (ranges[i].immutableSampler != nullptr) {
                    for (uint32_t j = 0; j < ranges[i].count; j++) {
                        samplers[i].emplace_back(unwrap(ranges[i].immutableSampler[j]));
                    }

                    ranges[i].immutableSampler = samplers[i].data();
                }
            }

            desc.descriptorRanges = ranges.data();
        }
    };

    struct DTDescriptorSet : RenderDescriptorSet {
        DeviceThread *thread = nullptr;
        RenderDescriptorSet *inner = nullptr;

        DTDescriptorSet(DeviceThread *thread, std::unique_ptr<RenderDescriptorSet> inner) : thread(thread), inner(inner.release()) { }
        ~DTDescriptorSet() override { destroyAsync(thread, inner); }

        void setBuffer(uint32_t descriptorIndex, const RenderBuffer *buffer, uint64_t bufferSize, const RenderBufferStructuredView *bufferStructuredView, const RenderBufferFormattedView *bufferFormattedView) override {
            const bool structured = (bufferStructuredView != nullptr);
            const RenderBufferStructuredView structuredView = structured ? *bufferStructuredView : RenderBufferStructuredView();
            thread->runAsync([set = inner, descriptorIndex, innerBuffer = unwrap(buffer), bufferSize, structured, structuredView, formattedView = unwrap(bufferFormattedView)]() {
                set->setBuffer(descriptorIndex, innerBuffer, bufferSize, structured ? &structuredView : nullptr, formattedView);
            });
        }

        void setTexture(uint32_t descriptorIndex, const RenderTexture *texture, RenderTextureLayout textureLayout, const RenderTextureView *textureView) override {
            // A texture still being made is read from its slot when the write runs, which is after it's made.
            const DTTexture *wrapper = static_cast<const DTTexture *>(texture);
            std::shared_ptr<DTTextureSlot> slot = ((wrapper != nullptr) && !wrapper->isMade()) ? wrapper->slot : nullptr;
            RenderTexture *innerTexture = (slot == nullptr) ? unwrap(texture) : nullptr;
            thread->runAsync([set = inner, descriptorIndex, slot = std::move(slot), innerTexture, textureLayout, innerView = unwrap(textureView)]() {
                set->setTexture(descriptorIndex, (slot != nullptr) ? slot->texture : innerTexture, textureLayout, innerView);
            });
        }

        void setSampler(uint32_t descriptorIndex, const RenderSampler *sampler) override {
            thread->runAsync([set = inner, descriptorIndex, innerSampler = unwrap(sampler)]() {
                set->setSampler(descriptorIndex, innerSampler);
            });
        }

        void setAccelerationStructure(uint32_t descriptorIndex, const RenderAccelerationStructure *accelerationStructure) override {
            assert(false && "Acceleration structures aren't forwarded to the device thread.");
        }
    };

    static RenderDescriptorSet *unwrap(const RenderDescriptorSet *object) { return unwrapAs<DTDescriptorSet>(object); }

    // DTCommandList

    struct DTCommandList : RenderCommandList {
        typedef std::function<void(RenderCommandList *)> Command;
        typedef std::vector<Command> Recording;

        DeviceThread *thread = nullptr;
        RenderCommandList *inner = nullptr;
        std::shared_ptr<Recording> recording;
        std::shared_ptr<const Recording> recorded;
        std::vector<std::shared_ptr<DTMapState>> readbacks;
        std::vector<std::shared_ptr<DTMapState>> recordedReadbacks;
        uint64_t openSerial = 0;
        const DTPipelineLayout *computeLayout = nullptr;
        const DTPipelineLayout *graphicsLayout = nullptr;

        DTCommandList(DeviceThread *thread, std::unique_ptr<RenderCommandList> inner) : thread(thread), inner(inner.release()) { }
        ~DTCommandList() override {
            setCommandListOpen(openSerial, false);
            destroyAsync(thread, inner);
        }

        void record(Command &&command) {
            assert((recording != nullptr) && "The command list wasn't begun.");
            if (openSerial == 0) {
                setCommandListOpen(openSerial, true);
            }

            recording->emplace_back(std::move(command));
        }

        // A copy into a READBACK buffer, whose answers to isMapReady() count only once this list is executed.
        void recordReadback(const RenderBuffer *buffer) {
            if (buffer == nullptr) {
                return;
            }

            const std::shared_ptr<DTMapState> &state = static_cast<const DTBuffer *>(buffer)->mapState;
            if ((state != nullptr) && (std::find(readbacks.begin(), readbacks.end(), state) == readbacks.end())) {
                readbacks.emplace_back(state);
            }
        }

        // Replays the last recording into the backend's command list. On the device thread.
        static void replay(RenderCommandList *commandList, const Recording &recording) {
            commandList->begin();
            for (const Command &command : recording) {
                command(commandList);
            }

            commandList->end();
        }

        static std::vector<uint8_t> copyPushConstants(const DTPipelineLayout *layout, uint32_t rangeIndex, const void *data, uint32_t size) {
            if (size == 0) {
                assert((layout != nullptr) && (rangeIndex < layout->pushConstantRanges.size()) && "Push constants without a size need the pipeline layout set first.");
                size = layout->pushConstantRanges[rangeIndex].size;
            }

            const uint8_t *bytes = reinterpret_cast<const uint8_t *>(data);
            return std::vector<uint8_t>(bytes, bytes + size);
        }

        void begin() override {
            setCommandListOpen(openSerial, false);
            recording = std::make_shared<Recording>();
            readbacks.clear();
            computeLayout = nullptr;
            graphicsLayout = nullptr;
        }

        void end() override {
            recorded = std::move(recording);
            recording.reset();
            recordedReadbacks = std::move(readbacks);
            readbacks.clear();
        }

        void barriers(RenderBarrierStages stages, const RenderBufferBarrier *bufferBarriers, uint32_t bufferBarriersCount, const RenderTextureBarrier *textureBarriers, uint32_t textureBarriersCount) override {
            std::vector<RenderBufferBarrier> buffers(bufferBarriers, bufferBarriers + bufferBarriersCount);
            for (RenderBufferBarrier &barrier : buffers) {
                barrier.buffer = unwrap(barrier.buffer);
            }

            std::vector<RenderTextureBarrier> textures(textureBarriers, textureBarriers + textureBarriersCount);
            for (RenderTextureBarrier &barrier : textures) {
                barrier.texture = unwrap(barrier.texture);
            }

            record([=](RenderCommandList *c) { c->barriers(stages, buffers.data(), uint32_t(buffers.size()), textures.data(), uint32_t(textures.size())); });
        }

        void dispatch(uint32_t threadGroupCountX, uint32_t threadGroupCountY, uint32_t threadGroupCountZ) override {
            record([=](RenderCommandList *c) { c->dispatch(threadGroupCountX, threadGroupCountY, threadGroupCountZ); });
        }

        void traceRays(uint32_t width, uint32_t height, uint32_t depth, RenderBufferReference shaderBindingTable, const RenderShaderBindingGroupsInfo &shaderBindingGroupsInfo) override {
            assert(false && "Raytracing isn't forwarded to the device thread.");
        }

        void drawInstanced(uint32_t vertexCountPerInstance, uint32_t instanceCount, uint32_t startVertexLocation, uint32_t startInstanceLocation) override {
            record([=](RenderCommandList *c) { c->drawInstanced(vertexCountPerInstance, instanceCount, startVertexLocation, startInstanceLocation); });
        }

        void drawIndexedInstanced(uint32_t indexCountPerInstance, uint32_t instanceCount, uint32_t startIndexLocation, int32_t baseVertexLocation, uint32_t startInstanceLocation) override {
            record([=](RenderCommandList *c) { c->drawIndexedInstanced(indexCountPerInstance, instanceCount, startIndexLocation, baseVertexLocation, startInstanceLocation); });
        }

        void setPipeline(const RenderPipeline *pipeline) override {
            record([p = unwrap(pipeline)](RenderCommandList *c) { c->setPipeline(p); });
        }

        void setComputePipelineLayout(const RenderPipelineLayout *pipelineLayout) override {
            computeLayout = static_cast<const DTPipelineLayout *>(pipelineLayout);
            record([p = unwrap(pipelineLayout)](RenderCommandList *c) { c->setComputePipelineLayout(p); });
        }

        void setComputePushConstants(uint32_t rangeIndex, const void *data, uint32_t offset, uint32_t size) override {
            std::vector<uint8_t> bytes = copyPushConstants(computeLayout, rangeIndex, data, size);
            record([=](RenderCommandList *c) { c->setComputePushConstants(rangeIndex, bytes.data(), offset, uint32_t(bytes.size())); });
        }

        void setComputeDescriptorSet(RenderDescriptorSet *descriptorSet, uint32_t setIndex) override {
            record([s = unwrap(descriptorSet), setIndex](RenderCommandList *c) { c->setComputeDescriptorSet(s, setIndex); });
        }

        void setGraphicsPipelineLayout(const RenderPipelineLayout *pipelineLayout) override {
            graphicsLayout = static_cast<const DTPipelineLayout *>(pipelineLayout);
            record([p = unwrap(pipelineLayout)](RenderCommandList *c) { c->setGraphicsPipelineLayout(p); });
        }

        void setGraphicsPushConstants(uint32_t rangeIndex, const void *data, uint32_t offset, uint32_t size) override {
            std::vector<uint8_t> bytes = copyPushConstants(graphicsLayout, rangeIndex, data, size);
            record([=](RenderCommandList *c) { c->setGraphicsPushConstants(rangeIndex, bytes.data(), offset, uint32_t(bytes.size())); });
        }

        void setGraphicsDescriptorSet(RenderDescriptorSet *descriptorSet, uint32_t setIndex) override {
            record([s = unwrap(descriptorSet), setIndex](RenderCommandList *c) { c->setGraphicsDescriptorSet(s, setIndex); });
        }

        void setGraphicsRootDescriptor(RenderBufferReference bufferReference, uint32_t rootDescriptorIndex) override {
            record([r = unwrap(bufferReference), rootDescriptorIndex](RenderCommandList *c) { c->setGraphicsRootDescriptor(r, rootDescriptorIndex); });
        }

        void setRaytracingPipelineLayout(const RenderPipelineLayout *pipelineLayout) override {
            assert(false && "Raytracing isn't forwarded to the device thread.");
        }

        void setRaytracingPushConstants(uint32_t rangeIndex, const void *data, uint32_t offset, uint32_t size) override {
            assert(false && "Raytracing isn't forwarded to the device thread.");
        }

        void setRaytracingDescriptorSet(RenderDescriptorSet *descriptorSet, uint32_t setIndex) override {
            assert(false && "Raytracing isn't forwarded to the device thread.");
        }

        void setIndexBuffer(const RenderIndexBufferView *view) override {
            const bool hasView = (view != nullptr);
            RenderIndexBufferView unwrapped = hasView ? *view : RenderIndexBufferView();
            unwrapped.buffer = unwrap(unwrapped.buffer);
            record([=](RenderCommandList *c) { c->setIndexBuffer(hasView ? &unwrapped : nullptr); });
        }

        void setVertexBuffers(uint32_t startSlot, const RenderVertexBufferView *views, uint32_t viewCount, const RenderInputSlot *inputSlots) override {
            std::vector<RenderVertexBufferView> unwrapped(views, views + ((views != nullptr) ? viewCount : 0));
            for (RenderVertexBufferView &view : unwrapped) {
                view.buffer = unwrap(view.buffer);
            }

            std::vector<RenderInputSlot> slots;
            if (inputSlots != nullptr) {
                slots.assign(inputSlots, inputSlots + viewCount);
            }

            record([=](RenderCommandList *c) { c->setVertexBuffers(startSlot, unwrapped.empty() ? nullptr : unwrapped.data(), viewCount, slots.empty() ? nullptr : slots.data()); });
        }

        void setViewports(const RenderViewport *viewports, uint32_t count) override {
            std::vector<RenderViewport> copied(viewports, viewports + count);
            record([=](RenderCommandList *c) { c->setViewports(copied.data(), uint32_t(copied.size())); });
        }

        void setScissors(const RenderRect *scissorRects, uint32_t count) override {
            std::vector<RenderRect> copied(scissorRects, scissorRects + count);
            record([=](RenderCommandList *c) { c->setScissors(copied.data(), uint32_t(copied.size())); });
        }

        void setFramebuffer(const RenderFramebuffer *framebuffer) override {
            record([f = unwrap(framebuffer)](RenderCommandList *c) { c->setFramebuffer(f); });
        }

        void setDepthBias(float depthBias, float depthBiasClamp, float slopeScaledDepthBias) override {
            record([=](RenderCommandList *c) { c->setDepthBias(depthBias, depthBiasClamp, slopeScaledDepthBias); });
        }

        void clearColor(uint32_t attachmentIndex, RenderColor colorValue, const RenderRect *clearRects, uint32_t clearRectsCount) override {
            std::vector<RenderRect> rects(clearRects, clearRects + ((clearRects != nullptr) ? clearRectsCount : 0));
            record([=](RenderCommandList *c) { c->clearColor(attachmentIndex, colorValue, rects.empty() ? nullptr : rects.data(), uint32_t(rects.size())); });
        }

        void clearDepthStencil(bool clearDepth, bool clearStencil, float depthValue, uint32_t stencilValue, const RenderRect *clearRects, uint32_t clearRectsCount) override {
            std::vector<RenderRect> rects(clearRects, clearRects + ((clearRects != nullptr) ? clearRectsCount : 0));
            record([=](RenderCommandList *c) { c->clearDepthStencil(clearDepth, clearStencil, depthValue, stencilValue, rects.empty() ? nullptr : rects.data(), uint32_t(rects.size())); });
        }

        void copyBufferRegion(RenderBufferReference dstBuffer, RenderBufferReference srcBuffer, uint64_t size) override {
            recordReadback(dstBuffer.ref);
            record([dst = unwrap(dstBuffer), src = unwrap(srcBuffer), size](RenderCommandList *c) { c->copyBufferRegion(dst, src, size); });
        }

        void copyTextureRegion(const RenderTextureCopyLocation &dstLocation, const RenderTextureCopyLocation &srcLocation, uint32_t dstX, uint32_t dstY, uint32_t dstZ, const RenderBox *srcBox) override {
            recordReadback(dstLocation.buffer);
            RenderTextureCopyLocation dst = dstLocation;
            dst.texture = unwrap(dst.texture);
            dst.buffer = unwrap(dst.buffer);
            RenderTextureCopyLocation src = srcLocation;
            src.texture = unwrap(src.texture);
            src.buffer = unwrap(src.buffer);
            const bool hasBox = (srcBox != nullptr);
            const RenderBox box = hasBox ? *srcBox : RenderBox();
            record([=](RenderCommandList *c) { c->copyTextureRegion(dst, src, dstX, dstY, dstZ, hasBox ? &box : nullptr); });
        }

        void copyBuffer(const RenderBuffer *dstBuffer, const RenderBuffer *srcBuffer) override {
            recordReadback(dstBuffer);
            record([dst = unwrap(dstBuffer), src = unwrap(srcBuffer)](RenderCommandList *c) { c->copyBuffer(dst, src); });
        }

        void copyTexture(const RenderTexture *dstTexture, const RenderTexture *srcTexture) override {
            record([dst = unwrap(dstTexture), src = unwrap(srcTexture)](RenderCommandList *c) { c->copyTexture(dst, src); });
        }

        void resolveTexture(const RenderTexture *dstTexture, const RenderTexture *srcTexture) override {
            record([dst = unwrap(dstTexture), src = unwrap(srcTexture)](RenderCommandList *c) { c->resolveTexture(dst, src); });
        }

        void resolveTextureRegion(const RenderTexture *dstTexture, uint32_t dstX, uint32_t dstY, const RenderTexture *srcTexture, const RenderRect *srcRect, RenderResolveMode resolveMode) override {
            const bool hasRect = (srcRect != nullptr);
            const RenderRect rect = hasRect ? *srcRect : RenderRect();
            record([=, dst = unwrap(dstTexture), src = unwrap(srcTexture)](RenderCommandList *c) { c->resolveTextureRegion(dst, dstX, dstY, src, hasRect ? &rect : nullptr, resolveMode); });
        }

        void buildBottomLevelAS(const RenderAccelerationStructure *dstAccelerationStructure, RenderBufferReference scratchBuffer, const RenderBottomLevelASBuildInfo &buildInfo) override {
            assert(false && "Acceleration structures aren't forwarded to the device thread.");
        }

        void buildTopLevelAS(const RenderAccelerationStructure *dstAccelerationStructure, RenderBufferReference scratchBuffer, RenderBufferReference instancesBuffer, const RenderTopLevelASBuildInfo &buildInfo) override {
            assert(false && "Acceleration structures aren't forwarded to the device thread.");
        }

        void discardTexture(const RenderTexture *texture) override {
            record([t = unwrap(texture)](RenderCommandList *c) { c->discardTexture(t); });
        }

        void resetQueryPool(const RenderQueryPool *queryPool, uint32_t queryFirstIndex, uint32_t queryCount) override {
            record([q = unwrap(queryPool), queryFirstIndex, queryCount](RenderCommandList *c) { c->resetQueryPool(q, queryFirstIndex, queryCount); });
        }

        void writeTimestamp(const RenderQueryPool *queryPool, uint32_t queryIndex) override {
            record([q = unwrap(queryPool), queryIndex](RenderCommandList *c) { c->writeTimestamp(q, queryIndex); });
        }
    };

    // DTSwapChain

    struct DTSwapChain : RenderSwapChain {
        DeviceThread *thread = nullptr;
        RenderDevice *innerDevice = nullptr;
        RenderCommandQueue *innerQueue = nullptr;
        RenderSwapChain *inner = nullptr;
        RenderSwapChainDesc desc;
        RenderWindow window = {};

        // What the caller draws into, and what present() copies to the backend's swap chain texture with.
        std::vector<std::unique_ptr<DTTexture>> textures;
        std::vector<std::unique_ptr<RenderTexture>> innerTextures;
        std::unique_ptr<RenderCommandList> copyList;
        std::unique_ptr<RenderCommandSemaphore> acquireSemaphore;
        std::unique_ptr<RenderCommandSemaphore> copySemaphore;

        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t nextTextureIndex = 0;
        bool vsyncEnabled = true;

        // What needsResize(), getRefreshRate() and isHidden() answer: the backend's answers at the last present, resize or
        // refresh on the device thread. A caller that waited for them each frame waited behind everything queued before
        // (RT64's present thread asks them before every present), so it reads these and queues a refresh instead.
        mutable std::atomic<bool> innerNeedsResize = false;
        mutable std::atomic<uint32_t> innerRefreshRate = 0;
        mutable std::atomic<bool> innerHidden = false;
        mutable std::atomic<bool> refreshing = false;

        // The presents queued for the device thread that haven't reached the backend's present() yet, which waitOnCaller()
        // counts as in flight.
        std::atomic<uint32_t> presentsQueued = 0;

        // On the device thread.
        DTSwapChain(DeviceThread *thread, RenderDevice *innerDevice, RenderCommandQueue *innerQueue, std::unique_ptr<RenderSwapChain> inner, const RenderSwapChainDesc &desc)
            : thread(thread), innerDevice(innerDevice), innerQueue(innerQueue), inner(inner.release()), desc(desc)
        {
            window = this->inner->getWindow();
            vsyncEnabled = this->inner->isVsyncEnabled();
            copyList = innerQueue->createCommandList();
            acquireSemaphore = innerDevice->createCommandSemaphore();
            copySemaphore = innerDevice->createCommandSemaphore();
            for (uint32_t i = 0; i < std::max(desc.textureCount, 1U); i++) {
                textures.emplace_back(std::make_unique<DTTexture>(thread));
            }

            innerTextures.resize(textures.size());
            updateTextures();
            publish();
        }

        ~DTSwapChain() override {
            thread->runSync([&]() {
                textures.clear();
                innerTextures.clear();
                copyList.reset();
                acquireSemaphore.reset();
                copySemaphore.reset();
                delete inner;
            });
        }

        // On the device thread: makes the caller's textures the size of the backend's swap chain.
        void updateTextures() {
            const uint32_t newWidth = inner->getWidth();
            const uint32_t newHeight = inner->getHeight();
            if ((newWidth == width) && (newHeight == height)) {
                return;
            }

            width = newWidth;
            height = newHeight;
            for (size_t i = 0; i < textures.size(); i++) {
                innerTextures[i].reset();
                if (!isEmpty()) {
                    innerTextures[i] = innerDevice->createTexture(RenderTextureDesc::ColorTarget(width, height, desc.format));
                }

                textures[i]->slot->texture = innerTextures[i].get();
            }
        }

        // On the device thread.
        void publish() const {
            innerNeedsResize = inner->needsResize();
            innerRefreshRate = inner->getRefreshRate();
            innerHidden = inner->isHidden();
        }

        // Queues a publish(), unless one is queued already.
        void refresh() const {
            bool idle = false;
            if (refreshing.compare_exchange_strong(idle, true)) {
                thread->runAsync([this]() {
                    refreshing = false;
                    publish();
                });
            }
        }

        bool present(uint32_t textureIndex, RenderCommandSemaphore **waitSemaphores, uint32_t waitSemaphoreCount) override {
            assert(textureIndex < textures.size());
            std::vector<RenderCommandSemaphore *> waits = unwrap(waitSemaphores, waitSemaphoreCount);
            presentsQueued++;
            thread->runAsync([this, textureIndex, waits]() mutable {
                publish();
                RenderTexture *source = innerTextures[textureIndex].get();
                uint32_t innerIndex = 0;
                if ((source == nullptr) || !inner->acquireTexture(acquireSemaphore.get(), &innerIndex)) {
                    presentsQueued--;
                    return;
                }

                RenderTexture *target = inner->getTexture(innerIndex);
                copyList->begin();
                copyList->barriers(RenderBarrierStage::COPY, { RenderTextureBarrier(source, RenderTextureLayout::COPY_SOURCE), RenderTextureBarrier(target, RenderTextureLayout::COPY_DEST) });
                copyList->copyTexture(target, source);
                copyList->barriers(RenderBarrierStage::NONE, RenderTextureBarrier(target, RenderTextureLayout::PRESENT));
                copyList->end();

                waits.emplace_back(acquireSemaphore.get());
                RenderCommandSemaphore *signal = copySemaphore.get();
                const RenderCommandList *commandList = copyList.get();
                innerQueue->executeCommandLists(&commandList, 1, waits.data(), uint32_t(waits.size()), &signal, 1);
                inner->present(innerIndex, &signal, 1);
                presentsQueued--;
                checkPendingReadbacks();
            });

            return true;
        }

        // The backend learns that a present's work is done in a callback on the device thread (WebGPU's), so where it can
        // this thread waits for that itself (waitOnCaller()), with no trip to the device thread. Each trip queued behind
        // whatever that thread was doing, a page's thread in Firefox's rendering, and the 1 ms sleep between two ended 6-14
        // ms late in Firefox on a laptop: wait() took 33-38 ms a game frame there, and RT64 paced Banjo-Kazooie's 60 to 30.
        // Otherwise (the first present, which asks the device thread) it polls isWaitReady() between calls rather than
        // holding the device thread, as map() does isMapReady().
        void wait() override {
            if (inner->waitOnCaller(presentsQueued)) {
                return;
            }

            const std::thread::id caller = std::this_thread::get_id();
            for (;;) {
                const bool ready = runSync(thread, [&]() {
                    if ((std::this_thread::get_id() != caller) && !inner->isWaitReady()) {
                        return false;
                    }

                    inner->wait();
                    return true;
                });

                if (ready) {
                    return;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        bool isWaitReady() override {
            return runSync(thread, [&]() { return inner->isWaitReady(); });
        }

#if defined(__EMSCRIPTEN__)
        // On the calling thread: it waits there, for frames the backend learns of on the device thread.
        bool sleepUntil(std::chrono::steady_clock::time_point time) override {
            return inner->sleepUntil(time);
        }
#endif

        bool resize() override {
            return runSync(thread, [&]() {
                const bool resized = inner->resize();
                updateTextures();
                publish();
                return resized;
            });
        }

        // The answer a present or a refresh ago: a window resized is seen at most one of them late.
        bool needsResize() const override {
            refresh();
            return innerNeedsResize;
        }

        void setVsyncEnabled(bool vsyncEnabled) override {
            this->vsyncEnabled = vsyncEnabled;
            thread->runAsync([swapChain = inner, vsyncEnabled]() { swapChain->setVsyncEnabled(vsyncEnabled); });
        }

        bool isVsyncEnabled() const override {
            return vsyncEnabled;
        }

        uint32_t getWidth() const override {
            return width;
        }

        uint32_t getHeight() const override {
            return height;
        }

        RenderTexture *getTexture(uint32_t textureIndex) override {
            assert(textureIndex < textures.size());
            return textures[textureIndex].get();
        }

        uint32_t getTextureCount() const override {
            return uint32_t(textures.size());
        }

        bool acquireTexture(RenderCommandSemaphore *signalSemaphore, uint32_t *textureIndex) override {
            assert(textureIndex != nullptr);
            if (isEmpty()) {
                return false;
            }

            *textureIndex = nextTextureIndex;
            nextTextureIndex = (nextTextureIndex + 1) % textures.size();

            // Nothing waits for a texture of this swap chain, but the caller will wait for the semaphore.
            if (signalSemaphore != nullptr) {
                thread->runAsync([queue = innerQueue, signal = unwrap(signalSemaphore)]() mutable {
                    queue->executeCommandLists(nullptr, 0, nullptr, 0, &signal, 1);
                });
            }

            return true;
        }

        RenderWindow getWindow() const override {
            return window;
        }

        bool isEmpty() const override {
            return (width == 0) || (height == 0);
        }

        uint32_t getRefreshRate() const override {
            refresh();
            return innerRefreshRate;
        }

        bool isHidden() const override {
            refresh();
            return innerHidden;
        }
    };

    // DTCommandQueue

    struct DTCommandQueue : RenderCommandQueue {
        DeviceThread *thread = nullptr;
        RenderDevice *innerDevice = nullptr;
        RenderCommandQueue *inner = nullptr;

        DTCommandQueue(DeviceThread *thread, RenderDevice *innerDevice, std::unique_ptr<RenderCommandQueue> inner) : thread(thread), innerDevice(innerDevice), inner(inner.release()) { }
        ~DTCommandQueue() override { destroyAsync(thread, inner); }

        std::unique_ptr<RenderCommandList> createCommandList() override {
            std::unique_ptr<RenderCommandList> commandList = runSync(thread, [&]() { return inner->createCommandList(); });
            return commandList ? std::make_unique<DTCommandList>(thread, std::move(commandList)) : nullptr;
        }

        std::unique_ptr<RenderSwapChain> createSwapChain(const RenderSwapChainDesc &desc) override {
            return runSync(thread, [&]() -> std::unique_ptr<RenderSwapChain> {
                std::unique_ptr<RenderSwapChain> swapChain = inner->createSwapChain(desc);
                if (swapChain == nullptr) {
                    return nullptr;
                }

                return std::make_unique<DTSwapChain>(thread, innerDevice, inner, std::move(swapChain), desc);
            });
        }

        void executeCommandLists(const RenderCommandList **commandLists, uint32_t commandListCount, RenderCommandSemaphore **waitSemaphores, uint32_t waitSemaphoreCount, RenderCommandSemaphore **signalSemaphores, uint32_t signalSemaphoreCount, RenderCommandFence *signalFence) override {
            struct Execution {
                RenderCommandList *commandList;
                std::shared_ptr<const DTCommandList::Recording> recording;
            };

            std::vector<Execution> executions;
            std::vector<std::shared_ptr<DTMapState>> readbacks;
            for (uint32_t i = 0; i < commandListCount; i++) {
                const DTCommandList *commandList = static_cast<const DTCommandList *>(commandLists[i]);
                assert((commandList->recorded != nullptr) && "The command list wasn't ended.");
                executions.push_back({ commandList->inner, commandList->recorded });
                readbacks.insert(readbacks.end(), commandList->recordedReadbacks.begin(), commandList->recordedReadbacks.end());
            }

            // What the upload buffers left mapped hold now goes to the backend's before the lists, as at a submission.
            {
                std::scoped_lock lock(mappedUploadsMutex);
                for (DTBuffer *buffer : mappedUploads) {
                    thread->runAsync([inner = buffer->inner, bytes = std::vector<uint8_t>(buffer->uploadCopy.get(), buffer->uploadCopy.get() + buffer->size)]() {
                        memcpy(inner->map(0, nullptr), bytes.data(), bytes.size());
                    });
                }
            }

            std::shared_ptr<DTFenceState> fenceState;
            uint64_t fenceSerial = 0;
            if (signalFence != nullptr) {
                fenceState = static_cast<DTCommandFence *>(signalFence)->state;
                std::scoped_lock lock(fenceState->mutex);
                fenceSerial = ++fenceState->queued;
            }

            thread->runAsync([queue = inner, executions = std::move(executions), readbacks, waits = unwrap(waitSemaphores, waitSemaphoreCount), signals = unwrap(signalSemaphores, signalSemaphoreCount), fence = unwrap(signalFence), fenceState, fenceSerial]() mutable {
                // A buffer that was mapped when its last data arrived, and that the caller never mapped and unmapped.
                for (const std::shared_ptr<DTMapState> &state : readbacks) {
                    state->unmap(0, nullptr);
                }

                std::vector<const RenderCommandList *> innerLists;
                for (const Execution &execution : executions) {
                    DTCommandList::replay(execution.commandList, *execution.recording);
                    innerLists.emplace_back(execution.commandList);
                }

                queue->executeCommandLists(innerLists.data(), uint32_t(innerLists.size()), waits.data(), uint32_t(waits.size()), signals.data(), uint32_t(signals.size()), fence);
                if (fenceState != nullptr) {
                    {
                        std::scoped_lock lock(fenceState->mutex);
                        fenceState->replayed = std::max(fenceState->replayed, fenceSerial);
                    }

                    fenceState->replayedCondition.notify_all();
                }

                for (const std::shared_ptr<DTMapState> &state : readbacks) {
                    state->replayed++;
                    if (std::find(pendingReadbacks.begin(), pendingReadbacks.end(), state) == pendingReadbacks.end()) {
                        pendingReadbacks.emplace_back(state);
                    }
                }

                checkPendingReadbacks();
            });

            // Once the replay is queued: an answer counts only after it.
            for (const std::shared_ptr<DTMapState> &state : readbacks) {
                state->executions++;
            }

            // After the replay is queued, so whatever the lists recorded is deleted once they're submitted.
            for (uint32_t i = 0; i < commandListCount; i++) {
                setCommandListOpen(const_cast<DTCommandList *>(static_cast<const DTCommandList *>(commandLists[i]))->openSerial, false);
            }
        }

        // Waits for the device thread to replay the executions that signal the fence up to the last wait for it, not the
        // one just queued: so a caller records its next list while the device thread replays its last, one execution in
        // flight. What a wait made safe to reuse is safe either way: a command list records anew at once, an upload
        // buffer's memory is the caller's own (DTBuffer::map()), a READBACK buffer's map() waits for its own data,
        // objects are deleted only after the lists that recorded them, and the backend's fences are no-ops (WebGPU's).
        // RT64 waited after each list, and each wait was a trip to the page's thread behind everything queued: on a
        // phone, 5.5-6.7 ms of each of its interpolated frames, and 9% of its display list thread's time.
        void waitForCommandFence(RenderCommandFence *fence) override {
            if ((fence == nullptr) || thread->isCurrent()) {
                runSync(thread, [&]() { inner->waitForCommandFence(unwrap(fence)); });
                return;
            }

            DTFenceState &state = *static_cast<DTCommandFence *>(fence)->state;
            std::unique_lock lock(state.mutex);
            const uint64_t target = state.waited;
            state.replayedCondition.wait(lock, [&]() { return state.replayed >= target; });
            state.waited = state.queued;
        }
    };

    // What makes a texture later on the device thread: the description with its clear value copied, since the caller's
    // is gone by then.
    template<typename F>
    static std::function<std::unique_ptr<RenderTexture>()> makeTexture(const RenderTextureDesc &desc, F &&create) {
        const bool cleared = (desc.optimizedClearValue != nullptr);
        const RenderClearValue clearValue = cleared ? *desc.optimizedClearValue : RenderClearValue();
        return [desc, cleared, clearValue, create = std::forward<F>(create)]() {
            RenderTextureDesc copy = desc;
            copy.optimizedClearValue = cleared ? &clearValue : nullptr;
            return create(copy);
        };
    }

    // DTPool

    struct DTPool : RenderPool {
        DeviceThread *thread = nullptr;
        RenderPool *inner = nullptr;

        DTPool(DeviceThread *thread, std::unique_ptr<RenderPool> inner) : thread(thread), inner(inner.release()) { }
        ~DTPool() override { destroyAsync(thread, inner); }

        std::unique_ptr<RenderBuffer> createBuffer(const RenderBufferDesc &desc) override {
            std::unique_ptr<RenderBuffer> buffer = runSync(thread, [&]() { return inner->createBuffer(desc); });
            return buffer ? std::make_unique<DTBuffer>(thread, std::move(buffer), desc) : nullptr;
        }

        std::unique_ptr<RenderTexture> createTexture(const RenderTextureDesc &desc) override {
            return std::make_unique<DTTexture>(thread, makeTexture(desc, [pool = inner](const RenderTextureDesc &desc) { return pool->createTexture(desc); }));
        }
    };

    // DTDevice

    struct DTDevice : RenderDevice {
        DeviceThread *thread = nullptr;
        RenderDevice *inner = nullptr;
        RenderDeviceCapabilities capabilities;
        RenderDeviceDescription description;

        DTDevice(DeviceThread *thread, std::unique_ptr<RenderDevice> inner) : thread(thread), inner(inner.release()) {
            capabilities = this->inner->getCapabilities();
            description = this->inner->getDescription();
#if defined(__EMSCRIPTEN__)
            // A READBACK buffer's data arrives in the backend's event loop, after the replay or present that last checked
            // it, so the device thread answers its waiters when it does. Otherwise they learned it only when they asked
            // again, after a wait of a millisecond, which on a page in Firefox on Windows ends on the system timer's
            // 15.6 ms tick (plume_web_waits.h): each frame copied to RAM waited a
            // tick or two, and Conker's page, which copies several a frame, presented 25-29 a second on a laptop.
            thread->runAsync([device = this->inner]() { device->setReadbackCallback(&checkPendingReadbacks); });
#endif
        }

        ~DTDevice() override { destroyAsync(thread, inner); }

        template<typename Wrapper, typename T, typename F>
        std::unique_ptr<T> create(F &&f) {
            std::unique_ptr<T> object = runSync(thread, f);
            return object ? std::make_unique<Wrapper>(thread, std::move(object)) : nullptr;
        }

        std::unique_ptr<RenderDescriptorSet> createDescriptorSet(const RenderDescriptorSetDesc &desc) override {
            return create<DTDescriptorSet, RenderDescriptorSet>([&]() {
                UnwrappedDescriptorSetDesc unwrapped(desc);
                return inner->createDescriptorSet(unwrapped.desc);
            });
        }

        std::unique_ptr<RenderShader> createShader(const void *data, uint64_t size, const char *entryPointName, RenderShaderFormat format) override {
            return create<DTShader, RenderShader>([&]() { return inner->createShader(data, size, entryPointName, format); });
        }

        std::unique_ptr<RenderSampler> createSampler(const RenderSamplerDesc &desc) override {
            return create<DTSampler, RenderSampler>([&]() { return inner->createSampler(desc); });
        }

        std::unique_ptr<RenderPipeline> createComputePipeline(const RenderComputePipelineDesc &desc) override {
            RenderComputePipelineDesc unwrapped = desc;
            unwrapped.pipelineLayout = unwrap(desc.pipelineLayout);
            unwrapped.computeShader = unwrap(desc.computeShader);
            return create<DTPipeline, RenderPipeline>([&]() { return inner->createComputePipeline(unwrapped); });
        }

        std::unique_ptr<RenderPipeline> createGraphicsPipeline(const RenderGraphicsPipelineDesc &desc) override {
            RenderGraphicsPipelineDesc unwrapped = desc;
            unwrapped.pipelineLayout = unwrap(desc.pipelineLayout);
            unwrapped.vertexShader = unwrap(desc.vertexShader);
            unwrapped.geometryShader = unwrap(desc.geometryShader);
            unwrapped.pixelShader = unwrap(desc.pixelShader);
            std::unique_ptr<RenderPipeline> pipeline = create<DTPipeline, RenderPipeline>([&]() { return inner->createGraphicsPipeline(unwrapped); });
            if (pipeline != nullptr) {
                static_cast<DTPipeline *>(pipeline.get())->waitUntilReady();
            }

            return pipeline;
        }

        std::unique_ptr<RenderPipeline> createRaytracingPipeline(const RenderRaytracingPipelineDesc &desc, const RenderPipeline *previousPipeline) override {
            assert(false && "Raytracing isn't forwarded to the device thread.");
            return nullptr;
        }

        std::unique_ptr<RenderCommandQueue> createCommandQueue(RenderCommandListType type) override {
            std::unique_ptr<RenderCommandQueue> queue = runSync(thread, [&]() { return inner->createCommandQueue(type); });
            return queue ? std::make_unique<DTCommandQueue>(thread, inner, std::move(queue)) : nullptr;
        }

        std::unique_ptr<RenderBuffer> createBuffer(const RenderBufferDesc &desc) override {
            std::unique_ptr<RenderBuffer> buffer = runSync(thread, [&]() { return inner->createBuffer(desc); });
            return buffer ? std::make_unique<DTBuffer>(thread, std::move(buffer), desc) : nullptr;
        }

        std::unique_ptr<RenderTexture> createTexture(const RenderTextureDesc &desc) override {
            return std::make_unique<DTTexture>(thread, makeTexture(desc, [device = inner](const RenderTextureDesc &desc) { return device->createTexture(desc); }));
        }

        std::unique_ptr<RenderAccelerationStructure> createAccelerationStructure(const RenderAccelerationStructureDesc &desc) override {
            assert(false && "Acceleration structures aren't forwarded to the device thread.");
            return nullptr;
        }

        std::unique_ptr<RenderPool> createPool(const RenderPoolDesc &desc) override {
            return create<DTPool, RenderPool>([&]() { return inner->createPool(desc); });
        }

        std::unique_ptr<RenderPipelineLayout> createPipelineLayout(const RenderPipelineLayoutDesc &desc) override {
            std::unique_ptr<RenderPipelineLayout> layout = runSync(thread, [&]() {
                std::vector<UnwrappedDescriptorSetDesc> sets(desc.descriptorSetDescs, desc.descriptorSetDescs + desc.descriptorSetDescsCount);
                std::vector<RenderDescriptorSetDesc> setDescs;
                for (const UnwrappedDescriptorSetDesc &set : sets) {
                    setDescs.emplace_back(set.desc);
                }

                RenderPipelineLayoutDesc unwrapped = desc;
                unwrapped.descriptorSetDescs = setDescs.empty() ? nullptr : setDescs.data();
                return inner->createPipelineLayout(unwrapped);
            });

            return layout ? std::make_unique<DTPipelineLayout>(thread, std::move(layout), desc) : nullptr;
        }

        std::unique_ptr<RenderCommandFence> createCommandFence() override {
            return create<DTCommandFence, RenderCommandFence>([&]() { return inner->createCommandFence(); });
        }

        std::unique_ptr<RenderCommandSemaphore> createCommandSemaphore() override {
            return create<DTCommandSemaphore, RenderCommandSemaphore>([&]() { return inner->createCommandSemaphore(); });
        }

        std::unique_ptr<RenderFramebuffer> createFramebuffer(const RenderFramebufferDesc &desc) override {
            std::vector<const RenderTexture *> colorAttachments(desc.colorAttachmentsCount);
            std::vector<const RenderTextureView *> colorAttachmentViews(desc.colorAttachmentsCount);
            for (uint32_t i = 0; i < desc.colorAttachmentsCount; i++) {
                colorAttachments[i] = unwrap(desc.colorAttachments[i]);
                colorAttachmentViews[i] = (desc.colorAttachmentViews != nullptr) ? unwrap(desc.colorAttachmentViews[i]) : nullptr;
            }

            RenderFramebufferDesc unwrapped = desc;
            unwrapped.colorAttachments = colorAttachments.empty() ? nullptr : colorAttachments.data();
            unwrapped.colorAttachmentViews = (desc.colorAttachmentViews != nullptr) ? colorAttachmentViews.data() : nullptr;
            unwrapped.depthAttachment = unwrap(desc.depthAttachment);
            unwrapped.depthAttachmentView = unwrap(desc.depthAttachmentView);

            // Made on the device thread: the wrapper reads the framebuffer's size.
            return runSync(thread, [&]() -> std::unique_ptr<RenderFramebuffer> {
                std::unique_ptr<RenderFramebuffer> framebuffer = inner->createFramebuffer(unwrapped);
                return framebuffer ? std::make_unique<DTFramebuffer>(thread, std::move(framebuffer)) : nullptr;
            });
        }

        std::unique_ptr<RenderQueryPool> createQueryPool(uint32_t queryCount) override {
            return runSync(thread, [&]() -> std::unique_ptr<RenderQueryPool> {
                std::unique_ptr<RenderQueryPool> queryPool = inner->createQueryPool(queryCount);
                return queryPool ? std::make_unique<DTQueryPool>(thread, std::move(queryPool)) : nullptr;
            });
        }

        void setBottomLevelASBuildInfo(RenderBottomLevelASBuildInfo &buildInfo, const RenderBottomLevelASMesh *meshes, uint32_t meshCount, bool preferFastBuild, bool preferFastTrace) override {
            assert(false && "Acceleration structures aren't forwarded to the device thread.");
        }

        void setTopLevelASBuildInfo(RenderTopLevelASBuildInfo &buildInfo, const RenderTopLevelASInstance *instances, uint32_t instanceCount, bool preferFastBuild, bool preferFastTrace) override {
            assert(false && "Acceleration structures aren't forwarded to the device thread.");
        }

        void setShaderBindingTableInfo(RenderShaderBindingTableInfo &tableInfo, const RenderShaderBindingGroups &groups, const RenderPipeline *pipeline, RenderDescriptorSet **descriptorSets, uint32_t descriptorSetCount) override {
            assert(false && "Raytracing isn't forwarded to the device thread.");
        }

        const RenderDeviceCapabilities &getCapabilities() const override {
            return capabilities;
        }

        const RenderDeviceDescription &getDescription() const override {
            return description;
        }

        RenderSampleCounts getSampleCountsSupported(RenderFormat format) const override {
            return runSync(thread, [&]() { return inner->getSampleCountsSupported(format); });
        }

#if defined(__EMSCRIPTEN__)
        // The backend answers from any thread (RenderDevice::isLost()), so this doesn't wait for the device thread.
        bool isLost() const override {
            return inner->isLost();
        }
#endif

        bool beginCapture() override {
            return runSync(thread, [&]() { return inner->beginCapture(); });
        }

        bool endCapture() override {
            return runSync(thread, [&]() { return inner->endCapture(); });
        }
    };

    // DTInterface

    struct DTInterface : RenderInterface {
        std::shared_ptr<DeviceThread> thread;
        RenderInterface *inner = nullptr;
        std::vector<std::string> deviceNames;
        RenderInterfaceCapabilities capabilities;

        DTInterface(std::shared_ptr<DeviceThread> thread, std::unique_ptr<RenderInterface> inner) : thread(thread), inner(inner.release()) {
            deviceNames = this->inner->getDeviceNames();
            capabilities = this->inner->getCapabilities();
        }

        ~DTInterface() override {
            // Waits, so that nothing queued outlives the thread.
            thread->runSync([&]() { delete inner; });
        }

        std::unique_ptr<RenderDevice> createDevice(const std::string &preferredDeviceName) override {
            return runSync(thread.get(), [&]() -> std::unique_ptr<RenderDevice> {
                std::unique_ptr<RenderDevice> device = inner->createDevice(preferredDeviceName);
                return device ? std::make_unique<DTDevice>(thread.get(), std::move(device)) : nullptr;
            });
        }

        const std::vector<std::string> &getDeviceNames() const override {
            return deviceNames;
        }

        const RenderInterfaceCapabilities &getCapabilities() const override {
            return capabilities;
        }
    };

    std::unique_ptr<RenderInterface> CreateDeviceThreadInterface(std::shared_ptr<DeviceThread> deviceThread, const std::function<std::unique_ptr<RenderInterface>()> &createInterface) {
        return runSync(deviceThread.get(), [&]() -> std::unique_ptr<RenderInterface> {
            std::unique_ptr<RenderInterface> renderInterface = createInterface();
            return renderInterface ? std::make_unique<DTInterface>(deviceThread, std::move(renderInterface)) : nullptr;
        });
    }

#if defined(__EMSCRIPTEN_PTHREADS__)
    struct EmscriptenDeviceThread : DeviceThread {
        emscripten::ProxyingQueue queue;
        pthread_t thread;

        EmscriptenDeviceThread(pthread_t thread) : thread(thread) { }

        bool isCurrent() const override {
            return pthread_equal(pthread_self(), thread);
        }

        void runSync(const std::function<void()> &work) override {
            if (isCurrent()) {
                queue.execute();
                work();
            }
            else {
                [[maybe_unused]] bool proxied = queue.proxySync(thread, work);
                assert(proxied && "The device thread is gone.");
            }
        }

        void runAsync(std::function<void()> &&work) override {
            if (isCurrent()) {
                queue.execute();
                work();
            }
            else {
                [[maybe_unused]] bool proxied = queue.proxyAsync(thread, std::move(work));
                assert(proxied && "The device thread is gone.");
            }
        }
    };

    std::shared_ptr<DeviceThread> CreateEmscriptenDeviceThread(pthread_t thread) {
        return std::make_shared<EmscriptenDeviceThread>(thread);
    }
#endif
};
