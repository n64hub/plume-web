//
// plume
//
// Copyright (c) 2024 renderbag and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file for details.
//

// WebGPU backend, for the browser through Emscripten and Emdawnwebgpu (--use-port=emdawnwebgpu).
//
// What core WebGPU lacks, and what this backend does instead:
// - Push constants: a shader has one push constant block, as in Vulkan, with each range at its offset in it. Here
//   the block is a uniform buffer at @group(3) @binding(999) (PushConstantsGroup, PushConstantsBinding: the last of
//   the 4 bind groups WebGPU guarantees), written into a per command list ring and bound with a dynamic offset. A
//   layout with 4 descriptor sets shares group 3 between its 4th set and the block, so the set can't use binding
//   999; with fewer, groups between the layout's sets and the block are bound empty. WGSL made from SPIR-V by naga
//   declares the block as var<immediate> (var<push_constant> before naga 30); PlumeWGSL.cmake rewrites that line.
// - Bind group layouts need each texture's sample type and each sampler's filtering, which plume's descriptor
//   ranges don't carry. So a descriptor set derives its layout from what is bound in it, and a pipeline is created
//   at draw time for the layouts of the sets bound then, and cached per pipeline layout.
// - A bind group can't be destroyed: the browser frees one once JavaScript has collected it. A renderer that rewrites
//   its descriptor sets every frame makes thousands a second, and Firefox (158, D3D12) collected them only after
//   ~150k had piled up, holding ~700k descriptors of its heap's million; past it, creating one fails with Out of
//   Memory and the device is lost. Most of them bind what a set held a frame or two before, so the device keeps bind
//   groups by their layout and what they bind (each handle's handleSerial, never reused), the most recently used
//   BindGroupCacheSize of them, and a set takes its group from there (WebGPUDevice::getBindGroup()).
// - Building a pipeline holds up the browser's GPU process (a large WGSL shader can take seconds), so a graphics
//   pipeline created with buildInBackground is built with createRenderPipelineAsync for the WebGPU layouts draws
//   with its plume layout have resolved to, and isReady() is false until every build has called back. Created before
//   any draw resolved that layout, it waits for the first one to. Behind plume_device_thread.h, createGraphicsPipeline()
//   waits for isReady() on the calling thread, which draws with another pipeline meanwhile. A draw with a layout the
//   pipeline wasn't built for (a new binding stage changes them all) builds it in place, as any other pipeline is.
// - Buffer mapping is asynchronous. An UPLOAD buffer keeps a CPU copy: map() returns it and unmap() writes it to the
//   GPU, and a buffer left mapped is written in full on every executeCommandLists(), since a persistently mapped
//   D3D12-style upload buffer never says what changed. A READBACK buffer that a submitted command list copied into
//   is mapped asynchronously right after the submission, and the callback copies it into the CPU copy, which map()
//   returns, and isMapReady() is false until it has run. Behind plume_device_thread.h (a build with threads), the
//   calling thread polls isMapReady() while the device thread is back in its event loop, where the callback runs.
//   Without threads, map() waits for it itself when the build can (ASYNCIFY or JSPI: emscripten_sleep() hands the
//   thread back to its event loop). Either way execute, wait, map reads what the GPU wrote, as on the other backends.
//   A build that can't wait gets whatever arrived before the call, possibly a submission old. Firefox looks for
//   finished work only after a submission or every 100 ms, so while a map waits, or a present is in flight (below), an
//   empty submission at most once a millisecond makes it look again (WebGPUDevice::pollQueue()).
// - Formatted buffer views (texel buffers) don't exist. A formatted view binds its buffer as a read-only storage
//   buffer of raw bytes, the binding padded to a multiple of 4, so the shader reads array<u32> and unpacks R8 or R16
//   elements itself: element i of an n-byte format is at byte i * n, little-endian. A writable formatted view of
//   elements under 4 bytes shares each u32 between neighbours, so a shader writing it needs atomicOr/atomicAnd.
// - A command list is recorded straight into a WGPUCommandEncoder. Render passes open on the first draw or clear
//   after setFramebuffer() and close on anything that can't happen inside one (copies, dispatches, barriers). A clear
//   of a whole target is the load op of the next pass, and a pass with only clears in it is opened when it closes. A
//   clear of rectangles draws a triangle over the target clipped to each rectangle, with a pipeline of the backend's
//   own: the color is the blend constant, the depth both ends of the viewport, the stencil value the reference.
//   Integer color targets can't blend, nor 32-bit float ones without float32-blendable, so rectangle clears of
//   those are ignored.
// - Resolving part of a texture, MSAA sample locations, timestamp queries, geometry shaders, binding arrays,
//   triangle fans, logic ops and dynamic depth bias are not supported, and the capabilities say so where plume has
//   one (resolveRegion, sampleLocations, queryPools, triangleFan, dynamicDepthBias).
// - Fences and semaphores are no-ops: WebGPU orders submissions and writeBuffer() calls on its one queue, and a
//   READBACK buffer waits for its own data in map().
// - The browser shows the canvas once the thread that drew it returns to its event loop. When the build can wait,
//   present() does that until the next animation frame, which paces presents to the display and keeps at most one
//   frame in flight. Otherwise present() returns at once and the caller has to give the thread back to the browser
//   itself (emscripten_set_main_loop or similar). Behind plume_device_thread.h the device thread does that between any
//   two of its tasks, and presents aren't paced.
// - Nor does every browser hold the page back when the GPU falls behind: Firefox (157, on an Intel UHD laptop GPU)
//   takes 60 frames a second whatever the GPU finishes, so its queue grew seconds deep and the page showed a frame or
//   two a second. So a swap chain counts the presents whose work the GPU hasn't finished (onSubmittedWorkDone), and
//   with presentWait, wait() holds the next present until fewer than the swap chain's maxFrameLatency (at least 2)
//   are, and isWaitReady() says whether it would. Behind plume_device_thread.h the calling thread waits for that itself
//   (waitOnCaller(), a futex the done callbacks wake), polling isWaitReady() only for the first present; without
//   threads, wait() waits when the build can. Not 1: Firefox calls back late, and with
//   one frame in flight a GPU that could finish ~54 frames a second gave 38, against 46 with two and 47 with three
//   (whose queue sat 50 ms behind against 33). Nor always 2: where the callback comes back later than two frames, two
//   in flight hold presents below the display's rate with the GPU idle, and RT64 reads the presents it skips as the GPU
//   falling behind and paces them to the game's frames. With each callback held 20-40 ms (an RTX 2070's Firefox gives
//   11-16 ms; a laptop's Intel graphics is suspected of more), Firefox 158 drew Banjo-Kazooie at 20 a second with 2 in
//   flight, 58 with 3 and 59 with 4 (before present() kicked Firefox to look for finished work while a present is in
//   flight, which brings a laptop's callback back in ~5 ms rather than at the next frame). The callbacks alone can't tell a late callback from a slow GPU, but only the first
//   gains from one more in flight, so while wait() holds presents a swap chain tries one more, up to 4, and keeps it if
//   presents rise (adaptPresentsInFlight()), but only from presents steady before the try: after a stall any recovery is
//   a rise. The page's query can fix the number (presents-in-flight=<1-8>), and
//   present-stats prints a line every 5 s: presents a second, how long after its present each frame's onSubmittedWorkDone
//   came back, the time wait() held presents, the time sleepUntil() spun on the clock, and the animation frames a
//   second, and each try that goes back. On a GPU
//   at hand, that says whether the GPU, the callbacks or the browser's frames hold presents back. uncapped=<rate>
//   reports the display's rate as <rate> (360) and keeps 4 in flight, to measure how many presents a second a page can
//   make with the browser's own frames uncapped (the swap chain's constructor says how).
// - With presentWait, the first present waits until the GPU process has done everything the device was asked for
//   before it (onSubmittedWorkDone). A caller's setup creates its shader modules and pipelines all at once, and the
//   browser does much of that in line, on the GPU process's thread every present queues behind (Chrome parses a
//   module with Tint there, and builds a pipeline created without createRenderPipelineAsync there): presented before
//   it was done, the first frames shown were followed by none for as long. RT64's setup held GoldenEye's launcher
//   ~0.45 s after its first frames that way in Chrome 155 (RTX 2070): two modules of ~250 KB of WGSL, 150 ms each, a
//   compute pipeline built at its first dispatch and render pipelines built at their first draw.
// - The browser doesn't say what the display's refresh rate is, so a swap chain times the animation frames with a
//   loop of its own on the device's thread, whatever its caller presents at, and reports displayTiming. The loop wakes
//   a thread in sleepUntil() at each frame, where a timed sleep on a page's thread ends late (plume_web_waits.h).
//   sleepUntil(), waitOnCaller() and isHidden() are the calls another thread may make.
// - isHidden() is document.hidden, asked on the page's thread. While the page is hidden, waitOnCaller() doesn't hold a
//   present to the cap on presents in flight: nothing is shown then, and Chrome calls a hidden page's
//   onSubmittedWorkDone back late, at times by seconds for the first 5-15 s (an empty submission's callback came back
//   0-12 times a second, against 80 shown, RTX 2070), so a caller that waited for it held its game behind it.
// - Every call must come from the thread that owns the device's JavaScript object: WebGPU objects can't be shared
//   between workers.
// - Shaders are WGSL text (RenderShaderFormat::WGSL). Specialization constants are WGSL overrides named by their id,
//   and a pipeline gets only those its shader declares: WebGPU refuses the rest, and a shader compiler drops what the
//   shader doesn't use.
// - A binding counts against the per-stage limits (16 storage buffers, 16 samplers...) in every stage its layout
//   entry is visible to, whether or not a shader there reads it. So an entry is visible only to the stages whose
//   shaders, of all created so far, declare that binding number (in any group), and a set's layout is rebuilt when a
//   new shader adds a stage to one of its bindings. A shader has to be created before the first draw that reads a
//   binding from a stage no other shader reads it from.
// - Depth textures are bound as unfilterable-float, which is how a shader written for the other backends reads them
//   (a texture of floats, not a depth texture).
// - Buffer copies are in whole 4-byte words: a copy whose offsets or size aren't is widened to the words around it,
//   so up to 3 bytes either side are copied too. That is exact for a staging buffer that mirrors its destination.
//
// The device is created by the page before the module starts (Module.preinitializedWebGPUDevice, see
// plume_webgpu_device.js), because requesting one is asynchronous. Emdawnwebgpu marks emscripten_webgpu_get_device(),
// which returns it, as deprecated; the replacement, requesting the adapter and device here and waiting with
// wgpuInstanceWaitAny(), needs ASYNCIFY or JSPI. RenderWindow is the CSS selector of the canvas, such as "#canvas".
//
// The examples build for the browser with emcmake and -DPLUME_BUILD_EXAMPLES=ON, given PLUME_DXC_EXECUTABLE (a dxc
// for the host) and naga-cli on the PATH or as PLUME_NAGA_EXECUTABLE; see examples/cmake/modules/PlumeWGSL.cmake.

#pragma once

#include "plume_render_interface.h"

#include <atomic>
#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include <webgpu/webgpu.h>

namespace plume {
    static const uint32_t PushConstantsGroup = 3;
    static const uint32_t PushConstantsBinding = 999;
    static const size_t BindGroupCacheSize = 4096;

    struct WebGPUBuffer;
    struct WebGPUCommandQueue;
    struct WebGPUDevice;
    struct WebGPUGraphicsPipeline;
    struct WebGPUInterface;
    struct WebGPURenderPipelineBuilds;
    struct WebGPUPipelineLayout;
    struct WebGPUPool;
    struct WebGPUTexture;

    struct WebGPUBufferFormattedView : RenderBufferFormattedView {
        RenderFormat format = RenderFormat::UNKNOWN;
        WebGPUBuffer *buffer = nullptr;

        WebGPUBufferFormattedView(WebGPUBuffer *buffer, RenderFormat format);
        ~WebGPUBufferFormattedView() override;
    };

    // A READBACK buffer's asynchronous map, shared with the callback that finishes it, so a buffer destroyed while a
    // map is in flight is never touched.
    struct WebGPUReadback {
        WebGPUBuffer *buffer = nullptr;
        uint64_t serial = 0;
        bool pending = false;
    };

    struct WebGPUBuffer : RenderBuffer {
        WGPUBuffer buffer = nullptr;
        uint64_t handleSerial = 0;
        WebGPUDevice *device = nullptr;
        RenderBufferDesc desc;
        std::vector<uint8_t> shadow;
        std::shared_ptr<WebGPUReadback> readback;
        bool mapped = false;

        WebGPUBuffer(WebGPUDevice *device, WebGPUPool *pool, const RenderBufferDesc &desc);
        ~WebGPUBuffer() override;
        void *map(uint32_t subresource, const RenderRange *readRange) override;
        void unmap(uint32_t subresource, const RenderRange *writtenRange) override;
        std::unique_ptr<RenderBufferFormattedView> createBufferFormattedView(RenderFormat format) override;
        void setName(const std::string &name) override;
        uint64_t getDeviceAddress() const override;
        bool isMapReady() override;
        void flush(uint64_t begin, uint64_t end);
        void requestReadback();
        void waitForReadback();
        void cancelReadback();
    };

    struct WebGPUTexture : RenderTexture {
        WGPUTexture texture = nullptr;
        WGPUTextureView defaultView = nullptr;
        WGPUTextureView storageView = nullptr;
        uint64_t handleSerial = 0;
        WebGPUDevice *device = nullptr;
        RenderTextureDesc desc;
        bool ownsTexture = true;

        WebGPUTexture() = default;
        WebGPUTexture(WebGPUDevice *device, WebGPUPool *pool, const RenderTextureDesc &desc);
        ~WebGPUTexture() override;
        std::unique_ptr<RenderTextureView> createTextureView(const RenderTextureViewDesc &desc) const override;
        void setName(const std::string &name) override;
        WGPUTextureView getDefaultView();
        WGPUTextureView getStorageView();
        void releaseHandle();
    };

    struct WebGPUTextureView : RenderTextureView {
        WGPUTextureView view = nullptr;
        uint64_t handleSerial = 0;
        const WebGPUTexture *texture = nullptr;
        RenderTextureViewDesc desc;

        WebGPUTextureView(const WebGPUTexture *texture, const RenderTextureViewDesc &desc);
        ~WebGPUTextureView() override;
    };

    struct WebGPUSampler : RenderSampler {
        WGPUSampler sampler = nullptr;
        uint64_t handleSerial = 0;
        RenderSamplerDesc desc;

        WebGPUSampler(WebGPUDevice *device, const RenderSamplerDesc &desc);
        ~WebGPUSampler() override;
    };

    struct WebGPUShader : RenderShader {
        WGPUShaderModule module = nullptr;
        std::string entryPointName;
        std::set<uint32_t> overrideIds;

        WebGPUShader(WebGPUDevice *device, const void *data, uint64_t size, const char *entryPointName, RenderShaderFormat format);
        ~WebGPUShader() override;
        void setName(const std::string &name) override;
    };

    struct WebGPUDescriptorSet : RenderDescriptorSet {
        struct Binding {
            RenderDescriptorRangeType type = RenderDescriptorRangeType::UNKNOWN;
            uint32_t binding = 0;
            const WebGPUBuffer *buffer = nullptr;
            uint64_t bufferSerial = 0;
            uint64_t bufferOffset = 0;
            uint64_t bufferSize = 0;
            const WebGPUTexture *texture = nullptr;
            const WebGPUTextureView *textureView = nullptr;
            const WebGPUSampler *sampler = nullptr;
        };

        WebGPUDevice *device = nullptr;
        std::vector<Binding> bindings;
        WGPUBindGroupLayout layout = nullptr;
        WGPUBindGroup bindGroup = nullptr;
        std::vector<WGPUBindGroupLayoutEntry> layoutEntries;
        std::vector<WGPUBindGroupEntry> groupEntries;
        std::string layoutKey;
        std::string groupKey;
        WGPUBindGroupLayout layoutWithPushConstants = nullptr;
        std::map<std::pair<uint64_t, uint32_t>, WGPUBindGroup> groupsWithPushConstants;
        uint64_t bindingStagesRevision = 0;
        bool dirty = true;

        WebGPUDescriptorSet(WebGPUDevice *device, const RenderDescriptorSetDesc &desc);
        ~WebGPUDescriptorSet() override;
        void setBuffer(uint32_t descriptorIndex, const RenderBuffer *buffer, uint64_t bufferSize, const RenderBufferStructuredView *bufferStructuredView, const RenderBufferFormattedView *bufferFormattedView) override;
        void setTexture(uint32_t descriptorIndex, const RenderTexture *texture, RenderTextureLayout textureLayout, const RenderTextureView *textureView) override;
        void setSampler(uint32_t descriptorIndex, const RenderSampler *sampler) override;
        void setAccelerationStructure(uint32_t descriptorIndex, const RenderAccelerationStructure *accelerationStructure) override;
        void update();
        WGPUBindGroupLayout getLayoutWithPushConstants();
        WGPUBindGroup getBindGroupWithPushConstants(WGPUBuffer buffer, uint64_t chunkId, uint32_t bindingSize);
        void releaseGroupsWithPushConstants();
    };

    // A pipeline is created when it is first drawn with, after the RenderShader may be gone, so it keeps its own
    // reference to the module.
    struct WebGPUShaderStage {
        WGPUShaderModule module = nullptr;
        std::string entryPointName;
        std::set<uint32_t> overrideIds;

        WebGPUShaderStage() = default;
        WebGPUShaderStage(const RenderShader *shader);
        WebGPUShaderStage(const WebGPUShaderStage &) = delete;
        WebGPUShaderStage &operator=(const WebGPUShaderStage &) = delete;
        ~WebGPUShaderStage();
    };

    struct WebGPUPipeline : RenderPipeline {
        enum class Type {
            Unknown,
            Compute,
            Graphics
        };

        Type type = Type::Unknown;
        WebGPUDevice *device = nullptr;

        WebGPUPipeline(WebGPUDevice *device, Type type);
        virtual ~WebGPUPipeline() override;
        void setName(const std::string &name) override;
        RenderPipelineProgram getProgram(const std::string &name) const override;
    };

    struct WebGPUComputePipeline : WebGPUPipeline {
        RenderComputePipelineDesc desc;
        WebGPUShaderStage computeStage;
        std::vector<RenderSpecConstant> specConstants;
        std::map<WGPUPipelineLayout, WGPUComputePipeline> pipelines;

        WebGPUComputePipeline(WebGPUDevice *device, const RenderComputePipelineDesc &desc);
        ~WebGPUComputePipeline() override;
        WGPUComputePipeline get(WGPUPipelineLayout layout);
    };

    // A graphics pipeline's builds in flight, shared with their callbacks, which can run after it is gone.
    struct WebGPURenderPipelineBuilds {
        WebGPUDevice *device = nullptr;
        WebGPUGraphicsPipeline *pipeline = nullptr;
        std::set<WGPUPipelineLayout> pending;
        bool waitingForLayout = false;
        std::string shadersKey;
    };

    // Background builds that compile the same shaders for the same layout as one in flight, started once it is done.
    struct WebGPUSharedShaderBuilds {
        std::vector<std::pair<std::weak_ptr<WebGPURenderPipelineBuilds>, WGPUPipelineLayout>> deferred;
    };

    struct WebGPUGraphicsPipeline : WebGPUPipeline {
        RenderGraphicsPipelineDesc desc;
        WebGPUShaderStage vertexStage;
        WebGPUShaderStage pixelStage;
        std::vector<RenderInputSlot> inputSlots;
        std::vector<RenderInputElement> inputElements;
        std::vector<RenderSpecConstant> specConstants;
        std::map<WGPUPipelineLayout, WGPURenderPipeline> pipelines;
        std::shared_ptr<WebGPURenderPipelineBuilds> builds;

        WebGPUGraphicsPipeline(WebGPUDevice *device, const RenderGraphicsPipelineDesc &desc);
        ~WebGPUGraphicsPipeline() override;
        bool isReady() override;
        WGPURenderPipeline get(WGPUPipelineLayout layout);
        WGPURenderPipeline build(WGPUPipelineLayout layout, bool async, bool deferIfShared = true);
    };

    struct WebGPUPipelineLayout : RenderPipelineLayout {
        WebGPUDevice *device = nullptr;
        uint32_t descriptorSetCount = 0;
        std::vector<RenderPushConstantRange> pushConstantRanges;
        uint32_t pushConstantsSize = 0;

        // The WebGPU layouts draws with it have resolved to since the last new binding stage (bindingStagesRevision).
        mutable std::vector<WGPUPipelineLayout> resolvedLayouts;
        mutable uint64_t resolvedRevision = 0;

        // Pipelines to build in the background that were created before any draw resolved it.
        mutable std::vector<std::weak_ptr<WebGPURenderPipelineBuilds>> waitingBuilds;

        WebGPUPipelineLayout(WebGPUDevice *device, const RenderPipelineLayoutDesc &desc);
        ~WebGPUPipelineLayout() override;
    };

    struct WebGPUCommandFence : RenderCommandFence {
    };

    struct WebGPUCommandSemaphore : RenderCommandSemaphore {
    };

    // The intervals between the browser's animation frames and the display's rate read from them, for getRefreshRate().
    // Its loop frees it on the first frame after the swap chain has stopped it.
    struct WebGPUFrameTimer {
        bool stopped = false;
        double lastFrameTime = 0.0;
        std::vector<double> frameIntervals;
        uint32_t frameIntervalCursor = 0;
        uint32_t refreshRate = 60;
        uint32_t frameCount = 0;

        // For sleepUntil() on other threads: the frames so far (a futex woken at each), when the last one came
        // (emscripten_get_now()) and the display's frame in ms.
        std::atomic<uint32_t> frameSerial = 0;
        std::atomic<double> frameNow = 0.0;
        std::atomic<double> frameMs = 1000.0 / 60.0;
    };

    // The presents whose work the GPU hasn't finished (a futex, woken as each is done), and for present-stats how long
    // each one's took to be reported done. Shared with their callbacks, which can outlive the swap chain. inFlight,
    // caughtUp, hidden (isHidden()'s last answer), waitedMs (the time waitOnCaller() held presents, not yet counted) and
    // spunMs (the time sleepUntil() spun on the clock, not yet counted) are read on other threads too.
    struct WebGPUPresentState {
        std::atomic<uint32_t> inFlight = 0;
        std::atomic<bool> caughtUp = false;
        std::atomic<bool> hidden = false;
        std::atomic<double> waitedMs = 0.0;
        std::atomic<double> spunMs = 0.0;
        bool timeWorkDone = false;
        std::vector<double> workDoneMs;
    };

    struct WebGPUSwapChain : RenderSwapChain {
        WebGPUCommandQueue *commandQueue = nullptr;
        WGPUSurface surface = nullptr;
        std::string selector;
        RenderSwapChainDesc desc;
        std::vector<std::unique_ptr<WebGPUTexture>> textures;
        uint32_t nextTextureIndex = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t configuredWidth = 0;
        uint32_t configuredHeight = 0;
        bool vsyncEnabled = true;
        WebGPUFrameTimer *frameTimer = nullptr;
        uint32_t uncappedRate = 0;

        std::shared_ptr<WebGPUPresentState> presentState = std::make_shared<WebGPUPresentState>();
        std::atomic<uint32_t> maxPresentsInFlight = 2;
        double waitStartTime = 0.0;
        bool caughtUpAsked = false;
        int kickTimer = 0;

        // The number in flight tried one higher while wait() holds presents, and kept only if presents rise with it.
        bool presentsInFlightFixed = false;
        double capWindowStart = 0.0;
        uint32_t capWindowPresents = 0;
        double capWaitMs = 0.0;
        double capLastPresent = 0.0;
        double capLongestGapMs = 0.0;
        double capTryRate = 0.0;
        bool capTrying = false;
        double capNextTry = 0.0;
        double capRetryMs = 10000.0;

        // present-stats: what happened since statsStartTime.
        bool presentStats = false;
        double statsStartTime = 0.0;
        uint32_t statsPresents = 0;
        uint32_t statsFrameCount = 0;
        double statsWaitMs = 0.0;
        double statsSpunMs = 0.0;

        WebGPUSwapChain(WebGPUCommandQueue *commandQueue, const RenderSwapChainDesc &desc);
        ~WebGPUSwapChain() override;
        bool present(uint32_t textureIndex, RenderCommandSemaphore **waitSemaphores, uint32_t waitSemaphoreCount) override;
        void wait() override;
        bool isWaitReady() override;
        bool sleepUntil(std::chrono::steady_clock::time_point time) override;
        bool waitOnCaller(const std::atomic<uint32_t> &queued) override;
        bool isHidden() const override;
        bool resize() override;
        bool needsResize() const override;
        void setVsyncEnabled(bool vsyncEnabled) override;
        bool isVsyncEnabled() const override;
        uint32_t getWidth() const override;
        uint32_t getHeight() const override;
        RenderTexture *getTexture(uint32_t textureIndex) override;
        uint32_t getTextureCount() const override;
        bool acquireTexture(RenderCommandSemaphore *signalSemaphore, uint32_t *textureIndex) override;
        RenderWindow getWindow() const override;
        bool isEmpty() const override;
        uint32_t getRefreshRate() const override;
        void getCanvasSize(uint32_t &width, uint32_t &height) const;
        void reportPresentStats();
        void adaptPresentsInFlight();
    };

    struct WebGPUFramebuffer : RenderFramebuffer {
        std::vector<const WebGPUTexture *> colorTextures;
        std::vector<const WebGPUTextureView *> colorViews;
        const WebGPUTexture *depthTexture = nullptr;
        const WebGPUTextureView *depthView = nullptr;
        bool depthReadOnly = false;
        uint32_t width = 0;
        uint32_t height = 0;

        WebGPUFramebuffer(const RenderFramebufferDesc &desc);
        ~WebGPUFramebuffer() override;
        uint32_t getWidth() const override;
        uint32_t getHeight() const override;
    };

    struct WebGPUCommandList : RenderCommandList {
        static const uint32_t MaxDescriptorSets = 8;
        static const uint32_t MaxVertexBuffers = 8;
        static const uint32_t PushConstantsChunkSize = 64 * 1024;

        struct PushConstantsChunk {
            WGPUBuffer buffer = nullptr;
            uint64_t id = 0;
            std::map<uint32_t, WGPUBindGroup> bindGroups;
            std::vector<uint8_t> data;
            uint32_t used = 0;
        };

        // The pipeline layout the last draw or dispatch resolved its bindings to, and what from: while none of it changes,
        // the next one's is the same. Each set's layout too, since another command list may have updated a set since.
        struct ResolvedBindings {
            const WebGPUPipelineLayout *layout = nullptr;
            WebGPUDescriptorSet *descriptorSets[MaxDescriptorSets] = {};
            WGPUBindGroupLayout setLayouts[MaxDescriptorSets] = {};
            uint64_t bindingStagesRevision = 0;
            WGPUPipelineLayout pipelineLayout = nullptr;
        };

        struct BindState {
            const WebGPUPipelineLayout *layout = nullptr;
            WebGPUDescriptorSet *descriptorSets[MaxDescriptorSets] = {};
            std::vector<uint8_t> pushConstants;
            bool pushConstantsDirty = false;
            ResolvedBindings resolved;
        };

        // What the open render pass has been given, so that a draw passes on only what changed: on a page each call is a
        // trip through JavaScript into the browser's GPU client on the page's thread, which a slow CPU runs out of first.
        // The pipeline and the groups are held (a reference each), so that one released meanwhile can't come back as
        // another object at the same address. A new pass starts with nothing set.
        struct PassState {
            WGPURenderPipeline pipeline = nullptr;
            WGPUBindGroup bindGroups[MaxDescriptorSets] = {};
            // Where in the ring the push constants bound to PushConstantsGroup are, and how many bytes of them.
            bool pushConstantsBound = false;
            uint32_t pushConstantsChunkIndex = 0;
            uint32_t pushConstantsOffset = 0;
            uint32_t pushConstantsSize = 0;
            uint32_t stencilReference = 0;
            bool stencilReferenceSet = false;
            RenderVertexBufferView vertexBuffers[MaxVertexBuffers] = {};
            RenderIndexBufferView indexBuffer;
            bool indexBufferSet = false;
            RenderViewport viewport;
            bool viewportSet = false;
            RenderRect scissor;
            bool scissorSet = false;
            bool blendConstantSet = false;
        };

        WebGPUDevice *device = nullptr;
        WGPUCommandEncoder encoder = nullptr;
        WGPUCommandBuffer commandBuffer = nullptr;
        WGPURenderPassEncoder renderPass = nullptr;
        const WebGPUFramebuffer *framebuffer = nullptr;
        std::vector<std::pair<bool, RenderColor>> pendingColorClears;
        bool pendingDepthClear = false;
        bool pendingStencilClear = false;
        float pendingDepthValue = 1.0f;
        uint32_t pendingStencilValue = 0;
        const WebGPUPipeline *pipeline = nullptr;
        BindState graphics;
        BindState compute;
        RenderVertexBufferView vertexBuffers[MaxVertexBuffers] = {};
        uint32_t vertexBufferCount = 0;
        RenderIndexBufferView indexBuffer;
        RenderViewport viewport;
        RenderRect scissor;
        bool viewportSet = false;
        bool scissorSet = false;
        bool renderStateDirty = true;
        PassState pass;
        std::vector<PushConstantsChunk> pushConstantsChunks;
        uint32_t pushConstantsChunkIndex = 0;

        WebGPUCommandList(WebGPUDevice *device);
        ~WebGPUCommandList() override;
        void begin() override;
        void end() override;
        void barriers(RenderBarrierStages stages, const RenderBufferBarrier *bufferBarriers, uint32_t bufferBarriersCount, const RenderTextureBarrier *textureBarriers, uint32_t textureBarriersCount) override;
        void dispatch(uint32_t threadGroupCountX, uint32_t threadGroupCountY, uint32_t threadGroupCountZ) override;
        void traceRays(uint32_t width, uint32_t height, uint32_t depth, RenderBufferReference shaderBindingTable, const RenderShaderBindingGroupsInfo &shaderBindingGroupsInfo) override;
        void drawInstanced(uint32_t vertexCountPerInstance, uint32_t instanceCount, uint32_t startVertexLocation, uint32_t startInstanceLocation) override;
        void drawIndexedInstanced(uint32_t indexCountPerInstance, uint32_t instanceCount, uint32_t startIndexLocation, int32_t baseVertexLocation, uint32_t startInstanceLocation) override;
        void setPipeline(const RenderPipeline *pipeline) override;
        void setComputePipelineLayout(const RenderPipelineLayout *pipelineLayout) override;
        void setComputePushConstants(uint32_t rangeIndex, const void *data, uint32_t offset, uint32_t size) override;
        void setComputeDescriptorSet(RenderDescriptorSet *descriptorSet, uint32_t setIndex) override;
        void setGraphicsPipelineLayout(const RenderPipelineLayout *pipelineLayout) override;
        void setGraphicsPushConstants(uint32_t rangeIndex, const void *data, uint32_t offset, uint32_t size) override;
        void setGraphicsDescriptorSet(RenderDescriptorSet *descriptorSet, uint32_t setIndex) override;
        void setGraphicsRootDescriptor(RenderBufferReference bufferReference, uint32_t rootDescriptorIndex) override;
        void setRaytracingPipelineLayout(const RenderPipelineLayout *pipelineLayout) override;
        void setRaytracingPushConstants(uint32_t rangeIndex, const void *data, uint32_t offset, uint32_t size) override;
        void setRaytracingDescriptorSet(RenderDescriptorSet *descriptorSet, uint32_t setIndex) override;
        void setIndexBuffer(const RenderIndexBufferView *view) override;
        void setVertexBuffers(uint32_t startSlot, const RenderVertexBufferView *views, uint32_t viewCount, const RenderInputSlot *inputSlots) override;
        void setViewports(const RenderViewport *viewports, uint32_t count) override;
        void setScissors(const RenderRect *scissorRects, uint32_t count) override;
        void setFramebuffer(const RenderFramebuffer *framebuffer) override;
        void setDepthBias(float depthBias, float depthBiasClamp, float slopeScaledDepthBias) override;
        void clearColor(uint32_t attachmentIndex, RenderColor colorValue, const RenderRect *clearRects, uint32_t clearRectsCount) override;
        void clearDepthStencil(bool clearDepth, bool clearStencil, float depthValue, uint32_t stencilValue, const RenderRect *clearRects, uint32_t clearRectsCount) override;
        void copyBufferRegion(RenderBufferReference dstBuffer, RenderBufferReference srcBuffer, uint64_t size) override;
        void copyTextureRegion(const RenderTextureCopyLocation &dstLocation, const RenderTextureCopyLocation &srcLocation, uint32_t dstX, uint32_t dstY, uint32_t dstZ, const RenderBox *srcBox) override;
        void copyBuffer(const RenderBuffer *dstBuffer, const RenderBuffer *srcBuffer) override;
        void copyTexture(const RenderTexture *dstTexture, const RenderTexture *srcTexture) override;
        void resolveTexture(const RenderTexture *dstTexture, const RenderTexture *srcTexture) override;
        void resolveTextureRegion(const RenderTexture *dstTexture, uint32_t dstX, uint32_t dstY, const RenderTexture *srcTexture, const RenderRect *srcRect, RenderResolveMode resolveMode) override;
        void buildBottomLevelAS(const RenderAccelerationStructure *dstAccelerationStructure, RenderBufferReference scratchBuffer, const RenderBottomLevelASBuildInfo &buildInfo) override;
        void buildTopLevelAS(const RenderAccelerationStructure *dstAccelerationStructure, RenderBufferReference scratchBuffer, RenderBufferReference instancesBuffer, const RenderTopLevelASBuildInfo &buildInfo) override;
        void discardTexture(const RenderTexture *texture) override;
        void resetQueryPool(const RenderQueryPool *queryPool, uint32_t queryFirstIndex, uint32_t queryCount) override;
        void writeTimestamp(const RenderQueryPool *queryPool, uint32_t queryIndex) override;

        std::vector<WebGPUBuffer *> readbackDestinations;

        void endRenderPass();
        bool beginRenderPass();
        void resetPassState();
        void setPassPipeline(WGPURenderPipeline renderPipeline);
        void clearRects(int32_t colorAttachment, const RenderColor &color, bool clearDepth, bool clearStencil, float depthValue, uint32_t stencilValue, const RenderRect *rects, uint32_t rectCount);
        void copiedInto(const RenderBuffer *buffer);
        bool prepareDraw(bool drawsNothing);
        void setPushConstants(BindState &state, uint32_t rangeIndex, const void *data, uint32_t offset, uint32_t size);
        WGPUPipelineLayout resolveBindings(BindState &state, bool drawWith = true);
        void bindGroups(BindState &state, WGPURenderPassEncoder renderPass, WGPUComputePassEncoder computePass);
    };

    struct WebGPUCommandQueue : RenderCommandQueue {
        WebGPUDevice *device = nullptr;

        WebGPUCommandQueue(WebGPUDevice *device);
        ~WebGPUCommandQueue() override;
        std::unique_ptr<RenderCommandList> createCommandList() override;
        std::unique_ptr<RenderSwapChain> createSwapChain(const RenderSwapChainDesc &desc) override;
        void executeCommandLists(const RenderCommandList **commandLists, uint32_t commandListCount, RenderCommandSemaphore **waitSemaphores, uint32_t waitSemaphoreCount, RenderCommandSemaphore **signalSemaphores, uint32_t signalSemaphoreCount, RenderCommandFence *signalFence) override;
        void waitForCommandFence(RenderCommandFence *fence) override;
    };

    struct WebGPUPool : RenderPool {
        WebGPUDevice *device = nullptr;
        RenderPoolDesc desc;

        WebGPUPool(WebGPUDevice *device, const RenderPoolDesc &desc);
        ~WebGPUPool() override;
        std::unique_ptr<RenderBuffer> createBuffer(const RenderBufferDesc &desc) override;
        std::unique_ptr<RenderTexture> createTexture(const RenderTextureDesc &desc) override;
    };

    struct WebGPUQueryPool : RenderQueryPool {
        std::vector<uint64_t> results;

        WebGPUQueryPool(uint32_t queryCount);
        ~WebGPUQueryPool() override;
        void queryResults() override;
        const uint64_t *getResults() const override;
        uint32_t getCount() const override;
    };

    struct WebGPUDevice : RenderDevice {
        WGPUDevice device = nullptr;
        WGPUQueue queue = nullptr;
        WebGPUInterface *renderInterface = nullptr;
        RenderDeviceCapabilities capabilities;
        RenderDeviceDescription description;
        bool float32Filterable = false;
        bool float32Blendable = false;
        uint64_t pushConstantsChunkCount = 0;
        bool depthClipControl = false;
        bool depth32FloatStencil8 = false;
        WGPUBindGroupLayout emptyLayout = nullptr;
        WGPUBindGroup emptyBindGroup = nullptr;
        // The handleSerial of every buffer not destroyed yet, so a descriptor set can tell a buffer it still binds
        // was destroyed since (WebGPUDescriptorSet::update()), and how many times one was. Buffers are made, destroyed
        // and bound on the device's one thread (plume_device_thread.h), so neither takes a lock.
        std::unordered_set<uint64_t> liveBufferSerials;
        uint32_t destroyedBufferBindings = 0;
        WGPUBindGroupLayoutEntry pushConstantsEntry = {};
        WGPUBindGroupLayout pushConstantsLayout = nullptr;
        std::unordered_map<std::string, WGPUBindGroupLayout> bindGroupLayouts;
        std::unordered_map<std::string, std::pair<WGPUBindGroup, std::list<std::string>::iterator>> bindGroups;
        std::list<std::string> bindGroupUses;
        std::unordered_map<uint32_t, WGPUShaderStage> bindingStages;
        uint64_t bindingStagesRevision = 0;
        std::map<std::vector<WGPUBindGroupLayout>, WGPUPipelineLayout> pipelineLayouts;
        std::vector<WebGPUBuffer *> mappedBuffers;
        WGPUShaderModule clearShader = nullptr;
        std::unordered_map<std::string, WGPURenderPipeline> clearPipelines;
        std::map<std::pair<std::string, WGPUPipelineLayout>, WebGPUSharedShaderBuilds> sharedShaderBuilds;
        double lastPollTime = 0.0;
        bool workDoneAfterSubmit = false;

        WebGPUDevice(WebGPUInterface *renderInterface);
        ~WebGPUDevice() override;
        std::unique_ptr<RenderDescriptorSet> createDescriptorSet(const RenderDescriptorSetDesc &desc) override;
        std::unique_ptr<RenderShader> createShader(const void *data, uint64_t size, const char *entryPointName, RenderShaderFormat format) override;
        std::unique_ptr<RenderSampler> createSampler(const RenderSamplerDesc &desc) override;
        std::unique_ptr<RenderPipeline> createComputePipeline(const RenderComputePipelineDesc &desc) override;
        std::unique_ptr<RenderPipeline> createGraphicsPipeline(const RenderGraphicsPipelineDesc &desc) override;
        std::unique_ptr<RenderPipeline> createRaytracingPipeline(const RenderRaytracingPipelineDesc &desc, const RenderPipeline *previousPipeline) override;
        std::unique_ptr<RenderCommandQueue> createCommandQueue(RenderCommandListType type) override;
        std::unique_ptr<RenderBuffer> createBuffer(const RenderBufferDesc &desc) override;
        std::unique_ptr<RenderTexture> createTexture(const RenderTextureDesc &desc) override;
        std::unique_ptr<RenderAccelerationStructure> createAccelerationStructure(const RenderAccelerationStructureDesc &desc) override;
        std::unique_ptr<RenderPool> createPool(const RenderPoolDesc &desc) override;
        std::unique_ptr<RenderPipelineLayout> createPipelineLayout(const RenderPipelineLayoutDesc &desc) override;
        std::unique_ptr<RenderCommandFence> createCommandFence() override;
        std::unique_ptr<RenderCommandSemaphore> createCommandSemaphore() override;
        std::unique_ptr<RenderFramebuffer> createFramebuffer(const RenderFramebufferDesc &desc) override;
        std::unique_ptr<RenderQueryPool> createQueryPool(uint32_t queryCount) override;
        void setBottomLevelASBuildInfo(RenderBottomLevelASBuildInfo &buildInfo, const RenderBottomLevelASMesh *meshes, uint32_t meshCount, bool preferFastBuild, bool preferFastTrace) override;
        void setTopLevelASBuildInfo(RenderTopLevelASBuildInfo &buildInfo, const RenderTopLevelASInstance *instances, uint32_t instanceCount, bool preferFastBuild, bool preferFastTrace) override;
        void setShaderBindingTableInfo(RenderShaderBindingTableInfo &tableInfo, const RenderShaderBindingGroups &groups, const RenderPipeline *pipeline, RenderDescriptorSet **descriptorSets, uint32_t descriptorSetCount) override;
        const RenderDeviceCapabilities &getCapabilities() const override;
        const RenderDeviceDescription &getDescription() const override;
        RenderSampleCounts getSampleCountsSupported(RenderFormat format) const override;
        bool beginCapture() override;
        bool endCapture() override;
        void setReadbackCallback(void (*callback)()) override;
        WGPUBindGroupLayout getBindGroupLayout(const std::string &key, const std::vector<WGPUBindGroupLayoutEntry> &entries);
        WGPUBindGroup getBindGroup(const std::string &key, WGPUBindGroupLayout layout, const std::vector<WGPUBindGroupEntry> &entries);
        WGPUPipelineLayout getPipelineLayout(const std::vector<WGPUBindGroupLayout> &layouts);
        void pollQueue();
        void flushMappedBuffers();
        WGPURenderPipeline getClearPipeline(const WebGPUFramebuffer *framebuffer, int32_t colorAttachment, bool clearDepth, bool clearStencil);
    };

    struct WebGPUInterface : RenderInterface {
        RenderInterfaceCapabilities capabilities;
        std::vector<std::string> deviceNames;

        WebGPUInterface();
        ~WebGPUInterface() override;
        std::unique_ptr<RenderDevice> createDevice(const std::string &preferredDeviceName) override;
        const RenderInterfaceCapabilities &getCapabilities() const override;
        const std::vector<std::string> &getDeviceNames() const override;
        bool isValid() const;
    };

    extern std::unique_ptr<RenderInterface> CreateWebGPUInterface();
};
