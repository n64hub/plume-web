//
// plume
//
// Copyright (c) 2024 renderbag and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file for details.
//

// The WebGPU backend. plume_webgpu.h says what it does where core WebGPU has no equivalent.

#include "plume_webgpu.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <emscripten/emscripten.h>
#include <emscripten/eventloop.h>
#include <emscripten/html5.h>
#include <emscripten/threading.h>

#include "plume_web_waits.h"

// Whether the browser builds a pipeline made with createRenderPipelineAsync without holding up the device. Chrome builds
// it on threads of its own. Firefox (157, Windows) builds it on its GPU process's WebGPU thread, in line with everything
// else on the device: a specialized RasterPS of RT64's took 2-8 s there, with no frame presented meanwhile. (On an Intel
// adapter Chrome builds them in the background as elsewhere: what was slow there was the frames drawn meanwhile.)
EM_JS(int, plume_webgpu_background_pipeline_builds, (), {
    if (!navigator.userAgent.includes('Firefox')) {
        return 1;
    }
    console.log('plume: pipelines are built in line with the device here (Firefox)');
    return 0;
});

// Whether the browser calls onSubmittedWorkDone and mapAsync back only when it next looks for finished work, after a
// submission or on a timer (WebGPUDevice::pollQueue()). Chrome calls back when the work is done.
EM_JS(int, plume_webgpu_work_done_after_submit, (), {
    return navigator.userAgent.includes('Firefox') ? 1 : 0;
});

// Whether the browser's preferred canvas format is bgra8unorm (rgba8unorm otherwise): the one it presents a canvas in
// without a copy. Firefox (158, Windows) hands a canvas to its compositor as a shared texture only in bgra8unorm, and
// reads any other format's frames back to the CPU and uploads them again: Rush 2049's page at 3840x2000 took its GPU
// process 1.2 cores instead of 0.5 and the GPU 80% busy instead of 43% (RTX 2070). Chrome cost the same either way.
EM_JS(int, plume_webgpu_bgra_swap_chain, (), {
    return navigator.gpu.getPreferredCanvasFormat() === 'bgra8unorm' ? 1 : 0;
});

// Whether the page is hidden: in a tab behind another, or a window minimized.
EM_JS(int, plume_webgpu_document_hidden, (), {
    return ((typeof document !== 'undefined') && document.hidden) ? 1 : 0;
});

// A number from the page's query: -1 without the name, 0 for one with no number (present-stats).
EM_JS(int, plume_webgpu_query_number, (const char *name), {
    const value = new URLSearchParams(globalThis.location ? location.search : '').get(UTF8ToString(name));
    if (value === null) {
        return -1;
    }
    const number = parseInt(value, 10);
    return isNaN(number) ? 0 : number;
});

// A canvas's drawing buffer is replaced with a black one whenever its width or height is set or its context configured,
// and the browser shows that until a frame is presented in it. SDL sets the size in its resize handler, and a swap chain
// is resized from a call that presents nothing, so each size a window went through was black for a frame or more, and
// the frames presented meanwhile were copied into a canvas of the other size (a black edge, or a copy WebGPU refused):
// going fullscreen, a few sizes in a row, flickered for a second or two. So the canvas keeps its buffer's size: its
// width and height read and write the size asked for, and the buffer takes that size only in plume_webgpu_size_canvas(),
// which acquireTexture() calls in the same task as the frame it then presents. Until then the browser stretches the
// last frame to the canvas's box.
EM_JS(void, plume_webgpu_hold_canvas_size, (const char *selector), {
    const canvas = document.querySelector(UTF8ToString(selector));
    if (!canvas || canvas.plumeSize) {
        return;
    }
    const size = canvas.plumeSize = { width: canvas.width, height: canvas.height };
    for (const name of ['width', 'height']) {
        Object.defineProperty(canvas, name, { configurable: true, get: () => size[name], set: (value) => { size[name] = value >>> 0; } });
    }
});

EM_JS(void, plume_webgpu_size_canvas, (const char *selector, uint32_t width, uint32_t height), {
    const canvas = document.querySelector(UTF8ToString(selector));
    if (canvas) {
        Object.getOwnPropertyDescriptor(HTMLCanvasElement.prototype, 'width').set.call(canvas, width);
        Object.getOwnPropertyDescriptor(HTMLCanvasElement.prototype, 'height').set.call(canvas, height);
    }
});

namespace plume {
    static const uint32_t PushConstantsAlignment = 256;

    // Unique to each WebGPU handle made, never reused as a handle's address can be: what the device's bind group cache
    // knows a resource by.
    static uint64_t newHandleSerial() {
        static std::atomic<uint64_t> serial = 0;
        return ++serial;
    }

    static WGPUStringView toStringView(const std::string &str) {
        return { str.c_str(), str.size() };
    }

    static void warnOnce(bool &warned, const char *message) {
        if (!warned) {
            fprintf(stderr, "plume WebGPU: %s\n", message);
            warned = true;
        }
    }

#define PLUME_WEBGPU_UNSUPPORTED(message) do { static bool warned = false; warnOnce(warned, message); } while (0)

    // Whether a call can hand the thread back to the browser and resume (ASYNCIFY or JSPI), which waiting for a
    // READBACK buffer or an animation frame needs. Not in a build with threads: there plume runs behind
    // plume_device_thread.h, in tasks of the device thread's event loop that can't be suspended, and the thread that
    // called waits instead (isMapReady()).
    static bool canWait() {
#if defined(__EMSCRIPTEN_PTHREADS__)
        return false;
#else
        static const bool result = emscripten_has_asyncify() != 0;
        return result;
#endif
    }

    static uint64_t alignUp(uint64_t value, uint64_t alignment) {
        return (value + alignment - 1) / alignment * alignment;
    }

    // Conversion functions.

    static WGPUTextureFormat toWGPU(RenderFormat format) {
        switch (format) {
        case RenderFormat::UNKNOWN: return WGPUTextureFormat_Undefined;
        case RenderFormat::R32G32B32A32_TYPELESS:
        case RenderFormat::R32G32B32A32_FLOAT: return WGPUTextureFormat_RGBA32Float;
        case RenderFormat::R32G32B32A32_UINT: return WGPUTextureFormat_RGBA32Uint;
        case RenderFormat::R32G32B32A32_SINT: return WGPUTextureFormat_RGBA32Sint;
        case RenderFormat::R16G16B16A16_TYPELESS:
        case RenderFormat::R16G16B16A16_FLOAT: return WGPUTextureFormat_RGBA16Float;
        case RenderFormat::R16G16B16A16_UNORM: return WGPUTextureFormat_RGBA16Unorm;
        case RenderFormat::R16G16B16A16_UINT: return WGPUTextureFormat_RGBA16Uint;
        case RenderFormat::R16G16B16A16_SNORM: return WGPUTextureFormat_RGBA16Snorm;
        case RenderFormat::R16G16B16A16_SINT: return WGPUTextureFormat_RGBA16Sint;
        case RenderFormat::R32G32_TYPELESS:
        case RenderFormat::R32G32_FLOAT: return WGPUTextureFormat_RG32Float;
        case RenderFormat::R32G32_UINT: return WGPUTextureFormat_RG32Uint;
        case RenderFormat::R32G32_SINT: return WGPUTextureFormat_RG32Sint;
        case RenderFormat::R8G8B8A8_TYPELESS:
        case RenderFormat::R8G8B8A8_UNORM: return WGPUTextureFormat_RGBA8Unorm;
        case RenderFormat::R8G8B8A8_UINT: return WGPUTextureFormat_RGBA8Uint;
        case RenderFormat::R8G8B8A8_SNORM: return WGPUTextureFormat_RGBA8Snorm;
        case RenderFormat::R8G8B8A8_SINT: return WGPUTextureFormat_RGBA8Sint;
        case RenderFormat::B8G8R8A8_UNORM: return WGPUTextureFormat_BGRA8Unorm;
        case RenderFormat::R16G16_TYPELESS:
        case RenderFormat::R16G16_FLOAT: return WGPUTextureFormat_RG16Float;
        case RenderFormat::R16G16_UNORM: return WGPUTextureFormat_RG16Unorm;
        case RenderFormat::R16G16_UINT: return WGPUTextureFormat_RG16Uint;
        case RenderFormat::R16G16_SNORM: return WGPUTextureFormat_RG16Snorm;
        case RenderFormat::R16G16_SINT: return WGPUTextureFormat_RG16Sint;
        case RenderFormat::R32_TYPELESS:
        case RenderFormat::R32_FLOAT: return WGPUTextureFormat_R32Float;
        case RenderFormat::D32_FLOAT: return WGPUTextureFormat_Depth32Float;
        case RenderFormat::D32_FLOAT_S8_UINT: return WGPUTextureFormat_Depth32FloatStencil8;
        case RenderFormat::R32_UINT: return WGPUTextureFormat_R32Uint;
        case RenderFormat::R32_SINT: return WGPUTextureFormat_R32Sint;
        case RenderFormat::R8G8_TYPELESS:
        case RenderFormat::R8G8_UNORM: return WGPUTextureFormat_RG8Unorm;
        case RenderFormat::R8G8_UINT: return WGPUTextureFormat_RG8Uint;
        case RenderFormat::R8G8_SNORM: return WGPUTextureFormat_RG8Snorm;
        case RenderFormat::R8G8_SINT: return WGPUTextureFormat_RG8Sint;
        case RenderFormat::R16_TYPELESS:
        case RenderFormat::R16_FLOAT: return WGPUTextureFormat_R16Float;
        case RenderFormat::D16_UNORM: return WGPUTextureFormat_Depth16Unorm;
        case RenderFormat::R16_UNORM: return WGPUTextureFormat_R16Unorm;
        case RenderFormat::R16_UINT: return WGPUTextureFormat_R16Uint;
        case RenderFormat::R16_SNORM: return WGPUTextureFormat_R16Snorm;
        case RenderFormat::R16_SINT: return WGPUTextureFormat_R16Sint;
        case RenderFormat::R8_TYPELESS:
        case RenderFormat::R8_UNORM: return WGPUTextureFormat_R8Unorm;
        case RenderFormat::R8_UINT: return WGPUTextureFormat_R8Uint;
        case RenderFormat::R8_SNORM: return WGPUTextureFormat_R8Snorm;
        case RenderFormat::R8_SINT: return WGPUTextureFormat_R8Sint;
        case RenderFormat::BC1_TYPELESS:
        case RenderFormat::BC1_UNORM: return WGPUTextureFormat_BC1RGBAUnorm;
        case RenderFormat::BC1_UNORM_SRGB: return WGPUTextureFormat_BC1RGBAUnormSrgb;
        case RenderFormat::BC2_TYPELESS:
        case RenderFormat::BC2_UNORM: return WGPUTextureFormat_BC2RGBAUnorm;
        case RenderFormat::BC2_UNORM_SRGB: return WGPUTextureFormat_BC2RGBAUnormSrgb;
        case RenderFormat::BC3_TYPELESS:
        case RenderFormat::BC3_UNORM: return WGPUTextureFormat_BC3RGBAUnorm;
        case RenderFormat::BC3_UNORM_SRGB: return WGPUTextureFormat_BC3RGBAUnormSrgb;
        case RenderFormat::BC4_TYPELESS:
        case RenderFormat::BC4_UNORM: return WGPUTextureFormat_BC4RUnorm;
        case RenderFormat::BC4_SNORM: return WGPUTextureFormat_BC4RSnorm;
        case RenderFormat::BC5_TYPELESS:
        case RenderFormat::BC5_UNORM: return WGPUTextureFormat_BC5RGUnorm;
        case RenderFormat::BC5_SNORM: return WGPUTextureFormat_BC5RGSnorm;
        case RenderFormat::BC6H_TYPELESS:
        case RenderFormat::BC6H_UF16: return WGPUTextureFormat_BC6HRGBUfloat;
        case RenderFormat::BC6H_SF16: return WGPUTextureFormat_BC6HRGBFloat;
        case RenderFormat::BC7_TYPELESS:
        case RenderFormat::BC7_UNORM: return WGPUTextureFormat_BC7RGBAUnorm;
        case RenderFormat::BC7_UNORM_SRGB: return WGPUTextureFormat_BC7RGBAUnormSrgb;
        default:
            // R32G32B32 is a vertex format only.
            PLUME_WEBGPU_UNSUPPORTED("a texture format WebGPU doesn't have was asked for");
            return WGPUTextureFormat_Undefined;
        }
    }

    static WGPUVertexFormat toVertexFormat(RenderFormat format) {
        switch (format) {
        case RenderFormat::R32G32B32A32_FLOAT: return WGPUVertexFormat_Float32x4;
        case RenderFormat::R32G32B32A32_UINT: return WGPUVertexFormat_Uint32x4;
        case RenderFormat::R32G32B32A32_SINT: return WGPUVertexFormat_Sint32x4;
        case RenderFormat::R32G32B32_FLOAT: return WGPUVertexFormat_Float32x3;
        case RenderFormat::R32G32B32_UINT: return WGPUVertexFormat_Uint32x3;
        case RenderFormat::R32G32B32_SINT: return WGPUVertexFormat_Sint32x3;
        case RenderFormat::R32G32_FLOAT: return WGPUVertexFormat_Float32x2;
        case RenderFormat::R32G32_UINT: return WGPUVertexFormat_Uint32x2;
        case RenderFormat::R32G32_SINT: return WGPUVertexFormat_Sint32x2;
        case RenderFormat::R32_FLOAT: return WGPUVertexFormat_Float32;
        case RenderFormat::R32_UINT: return WGPUVertexFormat_Uint32;
        case RenderFormat::R32_SINT: return WGPUVertexFormat_Sint32;
        case RenderFormat::R16G16B16A16_FLOAT: return WGPUVertexFormat_Float16x4;
        case RenderFormat::R16G16B16A16_UNORM: return WGPUVertexFormat_Unorm16x4;
        case RenderFormat::R16G16B16A16_SNORM: return WGPUVertexFormat_Snorm16x4;
        case RenderFormat::R16G16B16A16_UINT: return WGPUVertexFormat_Uint16x4;
        case RenderFormat::R16G16B16A16_SINT: return WGPUVertexFormat_Sint16x4;
        case RenderFormat::R16G16_FLOAT: return WGPUVertexFormat_Float16x2;
        case RenderFormat::R16G16_UNORM: return WGPUVertexFormat_Unorm16x2;
        case RenderFormat::R16G16_SNORM: return WGPUVertexFormat_Snorm16x2;
        case RenderFormat::R16G16_UINT: return WGPUVertexFormat_Uint16x2;
        case RenderFormat::R16G16_SINT: return WGPUVertexFormat_Sint16x2;
        case RenderFormat::R8G8B8A8_UNORM: return WGPUVertexFormat_Unorm8x4;
        case RenderFormat::R8G8B8A8_SNORM: return WGPUVertexFormat_Snorm8x4;
        case RenderFormat::R8G8B8A8_UINT: return WGPUVertexFormat_Uint8x4;
        case RenderFormat::R8G8B8A8_SINT: return WGPUVertexFormat_Sint8x4;
        case RenderFormat::B8G8R8A8_UNORM: return WGPUVertexFormat_Unorm8x4BGRA;
        case RenderFormat::R8G8_UNORM: return WGPUVertexFormat_Unorm8x2;
        case RenderFormat::R8G8_SNORM: return WGPUVertexFormat_Snorm8x2;
        case RenderFormat::R8G8_UINT: return WGPUVertexFormat_Uint8x2;
        case RenderFormat::R8G8_SINT: return WGPUVertexFormat_Sint8x2;
        case RenderFormat::R16_FLOAT: return WGPUVertexFormat_Float16;
        case RenderFormat::R16_UNORM: return WGPUVertexFormat_Unorm16;
        case RenderFormat::R16_UINT: return WGPUVertexFormat_Uint16;
        case RenderFormat::R16_SINT: return WGPUVertexFormat_Sint16;
        case RenderFormat::R8_UNORM: return WGPUVertexFormat_Unorm8;
        case RenderFormat::R8_UINT: return WGPUVertexFormat_Uint8;
        case RenderFormat::R8_SINT: return WGPUVertexFormat_Sint8;
        default:
            PLUME_WEBGPU_UNSUPPORTED("a vertex format WebGPU doesn't have was asked for");
            return WGPUVertexFormat_Float32;
        }
    }

    static WGPUIndexFormat toIndexFormat(RenderFormat format) {
        switch (format) {
        case RenderFormat::R16_UINT: return WGPUIndexFormat_Uint16;
        case RenderFormat::R32_UINT: return WGPUIndexFormat_Uint32;
        default:
            assert(false && "Unknown index format.");
            return WGPUIndexFormat_Undefined;
        }
    }

    static WGPUTextureDimension toWGPU(RenderTextureDimension dimension) {
        switch (dimension) {
        case RenderTextureDimension::TEXTURE_1D: return WGPUTextureDimension_1D;
        case RenderTextureDimension::TEXTURE_3D: return WGPUTextureDimension_3D;
        default: return WGPUTextureDimension_2D;
        }
    }

    static WGPUTextureViewDimension viewDimension(const RenderTextureDesc &textureDesc, const RenderTextureViewDesc *viewDesc) {
        RenderTextureViewDimension dimension = (viewDesc != nullptr) ? viewDesc->dimension : RenderTextureDimensionToView(textureDesc.dimension);
        uint32_t arraySize = (viewDesc != nullptr && viewDesc->arraySize != UINT32_MAX) ? viewDesc->arraySize : (textureDesc.arraySize - ((viewDesc != nullptr) ? viewDesc->arrayIndex : 0));
        switch (dimension) {
        case RenderTextureViewDimension::TEXTURE_1D: return WGPUTextureViewDimension_1D;
        case RenderTextureViewDimension::TEXTURE_3D: return WGPUTextureViewDimension_3D;
        case RenderTextureViewDimension::TEXTURE_CUBE: return (arraySize > 6) ? WGPUTextureViewDimension_CubeArray : WGPUTextureViewDimension_Cube;
        default: return (arraySize > 1) ? WGPUTextureViewDimension_2DArray : WGPUTextureViewDimension_2D;
        }
    }

    static WGPUCompareFunction toWGPU(RenderComparisonFunction function) {
        switch (function) {
        case RenderComparisonFunction::NEVER: return WGPUCompareFunction_Never;
        case RenderComparisonFunction::LESS: return WGPUCompareFunction_Less;
        case RenderComparisonFunction::EQUAL: return WGPUCompareFunction_Equal;
        case RenderComparisonFunction::LESS_EQUAL: return WGPUCompareFunction_LessEqual;
        case RenderComparisonFunction::GREATER: return WGPUCompareFunction_Greater;
        case RenderComparisonFunction::NOT_EQUAL: return WGPUCompareFunction_NotEqual;
        case RenderComparisonFunction::GREATER_EQUAL: return WGPUCompareFunction_GreaterEqual;
        case RenderComparisonFunction::ALWAYS: return WGPUCompareFunction_Always;
        default: return WGPUCompareFunction_Undefined;
        }
    }

    static WGPUStencilOperation toWGPU(RenderStencilOp op) {
        switch (op) {
        case RenderStencilOp::ZERO: return WGPUStencilOperation_Zero;
        case RenderStencilOp::REPLACE: return WGPUStencilOperation_Replace;
        case RenderStencilOp::INCREMENT_AND_CLAMP: return WGPUStencilOperation_IncrementClamp;
        case RenderStencilOp::DECREMENT_AND_CLAMP: return WGPUStencilOperation_DecrementClamp;
        case RenderStencilOp::INVERT: return WGPUStencilOperation_Invert;
        case RenderStencilOp::INCREMENT_AND_WRAP: return WGPUStencilOperation_IncrementWrap;
        case RenderStencilOp::DECREMENT_AND_WRAP: return WGPUStencilOperation_DecrementWrap;
        default: return WGPUStencilOperation_Keep;
        }
    }

    static WGPUBlendFactor toWGPU(RenderBlend blend) {
        switch (blend) {
        case RenderBlend::ZERO: return WGPUBlendFactor_Zero;
        case RenderBlend::ONE: return WGPUBlendFactor_One;
        case RenderBlend::SRC_COLOR: return WGPUBlendFactor_Src;
        case RenderBlend::INV_SRC_COLOR: return WGPUBlendFactor_OneMinusSrc;
        case RenderBlend::SRC_ALPHA: return WGPUBlendFactor_SrcAlpha;
        case RenderBlend::INV_SRC_ALPHA: return WGPUBlendFactor_OneMinusSrcAlpha;
        case RenderBlend::DEST_ALPHA: return WGPUBlendFactor_DstAlpha;
        case RenderBlend::INV_DEST_ALPHA: return WGPUBlendFactor_OneMinusDstAlpha;
        case RenderBlend::DEST_COLOR: return WGPUBlendFactor_Dst;
        case RenderBlend::INV_DEST_COLOR: return WGPUBlendFactor_OneMinusDst;
        case RenderBlend::SRC_ALPHA_SAT: return WGPUBlendFactor_SrcAlphaSaturated;
        case RenderBlend::BLEND_FACTOR: return WGPUBlendFactor_Constant;
        case RenderBlend::INV_BLEND_FACTOR: return WGPUBlendFactor_OneMinusConstant;
        case RenderBlend::SRC1_COLOR: return WGPUBlendFactor_Src1;
        case RenderBlend::INV_SRC1_COLOR: return WGPUBlendFactor_OneMinusSrc1;
        case RenderBlend::SRC1_ALPHA: return WGPUBlendFactor_Src1Alpha;
        case RenderBlend::INV_SRC1_ALPHA: return WGPUBlendFactor_OneMinusSrc1Alpha;
        default: return WGPUBlendFactor_One;
        }
    }

    static WGPUBlendOperation toWGPU(RenderBlendOperation op) {
        switch (op) {
        case RenderBlendOperation::SUBTRACT: return WGPUBlendOperation_Subtract;
        case RenderBlendOperation::REV_SUBTRACT: return WGPUBlendOperation_ReverseSubtract;
        case RenderBlendOperation::MIN: return WGPUBlendOperation_Min;
        case RenderBlendOperation::MAX: return WGPUBlendOperation_Max;
        default: return WGPUBlendOperation_Add;
        }
    }

    static WGPUPrimitiveTopology toWGPU(RenderPrimitiveTopology topology) {
        switch (topology) {
        case RenderPrimitiveTopology::POINT_LIST: return WGPUPrimitiveTopology_PointList;
        case RenderPrimitiveTopology::LINE_LIST: return WGPUPrimitiveTopology_LineList;
        case RenderPrimitiveTopology::LINE_STRIP: return WGPUPrimitiveTopology_LineStrip;
        case RenderPrimitiveTopology::TRIANGLE_STRIP: return WGPUPrimitiveTopology_TriangleStrip;
        case RenderPrimitiveTopology::TRIANGLE_FAN:
            PLUME_WEBGPU_UNSUPPORTED("triangle fans are drawn as triangle lists");
            return WGPUPrimitiveTopology_TriangleList;
        default: return WGPUPrimitiveTopology_TriangleList;
        }
    }

    static WGPUAddressMode toWGPU(RenderTextureAddressMode mode) {
        switch (mode) {
        case RenderTextureAddressMode::WRAP: return WGPUAddressMode_Repeat;
        case RenderTextureAddressMode::MIRROR: return WGPUAddressMode_MirrorRepeat;
        case RenderTextureAddressMode::BORDER:
            PLUME_WEBGPU_UNSUPPORTED("border addressing is clamped to the edge");
            return WGPUAddressMode_ClampToEdge;
        case RenderTextureAddressMode::MIRROR_ONCE:
            PLUME_WEBGPU_UNSUPPORTED("mirror-once addressing is mirrored");
            return WGPUAddressMode_MirrorRepeat;
        default: return WGPUAddressMode_ClampToEdge;
        }
    }

    static bool isIntegerFormat(RenderFormat format, bool &isSigned) {
        switch (format) {
        case RenderFormat::R32G32B32A32_UINT: case RenderFormat::R32G32B32_UINT: case RenderFormat::R16G16B16A16_UINT:
        case RenderFormat::R32G32_UINT: case RenderFormat::R8G8B8A8_UINT: case RenderFormat::R16G16_UINT:
        case RenderFormat::R32_UINT: case RenderFormat::R8G8_UINT: case RenderFormat::R16_UINT: case RenderFormat::R8_UINT:
            isSigned = false;
            return true;
        case RenderFormat::R32G32B32A32_SINT: case RenderFormat::R32G32B32_SINT: case RenderFormat::R16G16B16A16_SINT:
        case RenderFormat::R32G32_SINT: case RenderFormat::R8G8B8A8_SINT: case RenderFormat::R16G16_SINT:
        case RenderFormat::R32_SINT: case RenderFormat::R8G8_SINT: case RenderFormat::R16_SINT: case RenderFormat::R8_SINT:
            isSigned = true;
            return true;
        default:
            return false;
        }
    }

    static bool isFloat32Format(RenderFormat format) {
        switch (format) {
        case RenderFormat::R32G32B32A32_TYPELESS: case RenderFormat::R32G32B32A32_FLOAT:
        case RenderFormat::R32G32_TYPELESS: case RenderFormat::R32G32_FLOAT:
        case RenderFormat::R32_TYPELESS: case RenderFormat::R32_FLOAT:
            return true;
        default:
            return false;
        }
    }

    // Bytes per row of blocks, and rows of blocks, for a copy of the given texel size.
    static void copyFootprint(RenderFormat format, uint32_t width, uint32_t height, uint32_t &bytesPerRow, uint32_t &rows) {
        const uint32_t blockWidth = RenderFormatBlockWidth(format);
        bytesPerRow = ((width + blockWidth - 1) / blockWidth) * RenderFormatSize(format);
        rows = (height + blockWidth - 1) / blockWidth;
    }

    // WebGPUBuffer

    WebGPUBuffer::WebGPUBuffer(WebGPUDevice *device, WebGPUPool *pool, const RenderBufferDesc &desc) {
        assert(device != nullptr);

        this->device = device;
        this->desc = desc;

        WGPUBufferDescriptor bufferDesc = WGPU_BUFFER_DESCRIPTOR_INIT;

        // A uniform binding covers its struct rounded up to 16 bytes, which can be more than the C++ struct the
        // buffer is made for, so a constant buffer is made in whole 16-byte rows and bound that way.
        const uint64_t sizeAlignment = (desc.flags & RenderBufferFlag::CONSTANT) ? 16 : 4;
        bufferDesc.size = alignUp(std::max<uint64_t>(desc.size, 4), sizeAlignment);
        if (desc.heapType == RenderHeapType::READBACK) {
            bufferDesc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
        }
        else {
            bufferDesc.usage = WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
            if (desc.flags & RenderBufferFlag::VERTEX) bufferDesc.usage |= WGPUBufferUsage_Vertex;
            if (desc.flags & RenderBufferFlag::INDEX) bufferDesc.usage |= WGPUBufferUsage_Index;
            if (desc.flags & RenderBufferFlag::CONSTANT) bufferDesc.usage |= WGPUBufferUsage_Uniform;
            if (desc.flags & (RenderBufferFlag::STORAGE | RenderBufferFlag::UNORDERED_ACCESS | RenderBufferFlag::FORMATTED)) bufferDesc.usage |= WGPUBufferUsage_Storage;
        }

        buffer = wgpuDeviceCreateBuffer(device->device, &bufferDesc);
        handleSerial = newHandleSerial();
        if ((desc.heapType == RenderHeapType::UPLOAD) || (desc.heapType == RenderHeapType::GPU_UPLOAD) || (desc.heapType == RenderHeapType::READBACK)) {
            shadow.resize(bufferDesc.size, 0);
        }

        if (desc.heapType == RenderHeapType::READBACK) {
            readback = std::make_shared<WebGPUReadback>();
            readback->buffer = this;
        }
    }

    WebGPUBuffer::~WebGPUBuffer() {
        if (mapped) {
            auto &mappedBuffers = device->mappedBuffers;
            mappedBuffers.erase(std::remove(mappedBuffers.begin(), mappedBuffers.end(), this), mappedBuffers.end());
        }

        if (readback != nullptr) {
            readback->buffer = nullptr;
        }

        if (buffer != nullptr) {
            wgpuBufferDestroy(buffer);
            wgpuBufferRelease(buffer);
        }
    }

    void *WebGPUBuffer::map(uint32_t subresource, const RenderRange *readRange) {
        if (shadow.empty()) {
            assert(false && "Only UPLOAD and READBACK buffers can be mapped.");
            return nullptr;
        }

        if (desc.heapType == RenderHeapType::READBACK) {
            waitForReadback();
        }
        else if (!mapped) {
            device->mappedBuffers.emplace_back(this);
        }

        mapped = true;
        return shadow.data();
    }

    void WebGPUBuffer::unmap(uint32_t subresource, const RenderRange *writtenRange) {
        if (!mapped) {
            return;
        }

        mapped = false;
        if (desc.heapType == RenderHeapType::READBACK) {
            return;
        }

        auto &mappedBuffers = device->mappedBuffers;
        mappedBuffers.erase(std::remove(mappedBuffers.begin(), mappedBuffers.end(), this), mappedBuffers.end());
        if (writtenRange != nullptr) {
            flush(writtenRange->begin, writtenRange->end);
        }
        else {
            flush(0, shadow.size());
        }
    }

    void WebGPUBuffer::flush(uint64_t begin, uint64_t end) {
        // writeBuffer wants a multiple of 4 bytes at a multiple of 4.
        begin = begin & ~uint64_t(3);
        end = std::min<uint64_t>(alignUp(end, 4), shadow.size());
        if (end > begin) {
            wgpuQueueWriteBuffer(device->queue, buffer, begin, shadow.data() + begin, end - begin);
        }
    }

    // Called once a READBACK buffer's data has arrived (setReadbackCallback). One for every device: the callbacks of a
    // device's maps can come after it is gone.
    static void (*readbackCallback)() = nullptr;

    static void readbackMapped(WGPUMapAsyncStatus status, WGPUStringView message, void *userdata1, void *userdata2) {
        std::shared_ptr<WebGPUReadback> *pointer = static_cast<std::shared_ptr<WebGPUReadback> *>(userdata1);
        std::shared_ptr<WebGPUReadback> readback = std::move(*pointer);
        delete pointer;

        // A map that was cancelled has been replaced by a newer one.
        if (uint32_t(readback->serial) != uint32_t(uintptr_t(userdata2))) {
            return;
        }

        readback->pending = false;
        WebGPUBuffer *buffer = readback->buffer;
        if ((status == WGPUMapAsyncStatus_Success) && (buffer != nullptr)) {
            const void *data = wgpuBufferGetConstMappedRange(buffer->buffer, 0, buffer->shadow.size());
            if (data != nullptr) {
                memcpy(buffer->shadow.data(), data, buffer->shadow.size());
            }

            wgpuBufferUnmap(buffer->buffer);
        }

        if (readbackCallback != nullptr) {
            readbackCallback();
        }
    }

    // Called after a submission that copied into this buffer.
    void WebGPUBuffer::requestReadback() {
        assert((readback != nullptr) && !readback->pending);

        readback->serial++;
        readback->pending = true;
        WGPUBufferMapCallbackInfo callbackInfo = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
        callbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
        callbackInfo.callback = readbackMapped;
        callbackInfo.userdata1 = new std::shared_ptr<WebGPUReadback>(readback);
        callbackInfo.userdata2 = reinterpret_cast<void *>(uintptr_t(readback->serial));
        wgpuBufferMapAsync(buffer, WGPUMapMode_Read, 0, shadow.size(), callbackInfo);
    }

    // Waits for the map in flight, if the build can wait. Called before reading the buffer and before a submission
    // copies into it again, since a buffer with a map pending can't be used in one: a map that can't be waited for is
    // cancelled then, and its data is lost.
    void WebGPUBuffer::waitForReadback() {
        if (!readback->pending) {
            return;
        }

        if (canWait()) {
            while (readback->pending) {
                device->pollQueue();
                emscripten_sleep(0);
            }
        }
        else {
            PLUME_WEBGPU_UNSUPPORTED("a READBACK buffer was mapped before its data arrived and this build can't wait for it (no ASYNCIFY or JSPI, or threads without plume_device_thread.h): it reads what arrived before the call");
        }
    }

    bool WebGPUBuffer::isMapReady() {
        if ((readback == nullptr) || !readback->pending) {
            return true;
        }

        device->pollQueue();
        return false;
    }

    void WebGPUBuffer::cancelReadback() {
        if (readback->pending) {
            wgpuBufferUnmap(buffer);
            readback->pending = false;
            readback->serial++;
        }
    }

    std::unique_ptr<RenderBufferFormattedView> WebGPUBuffer::createBufferFormattedView(RenderFormat format) {
        return std::make_unique<WebGPUBufferFormattedView>(this, format);
    }

    void WebGPUBuffer::setName(const std::string &name) {
        wgpuBufferSetLabel(buffer, toStringView(name));
    }

    uint64_t WebGPUBuffer::getDeviceAddress() const {
        assert(false && "WebGPU has no buffer device addresses.");
        return 0;
    }

    // WebGPUBufferFormattedView

    WebGPUBufferFormattedView::WebGPUBufferFormattedView(WebGPUBuffer *buffer, RenderFormat format) {
        // WebGPU has no texel buffers. The view is kept so descriptor sets can bind the buffer as a storage buffer.
        this->buffer = buffer;
        this->format = format;
    }

    WebGPUBufferFormattedView::~WebGPUBufferFormattedView() { }

    // WebGPUTexture

    WebGPUTexture::WebGPUTexture(WebGPUDevice *device, WebGPUPool *pool, const RenderTextureDesc &desc) {
        assert(device != nullptr);

        this->device = device;
        this->desc = desc;

        const bool multisampled = desc.multisampling.sampleCount > 1;
        WGPUTextureDescriptor textureDesc = WGPU_TEXTURE_DESCRIPTOR_INIT;
        textureDesc.dimension = toWGPU(desc.dimension);
        textureDesc.size.width = std::max(desc.width, 1U);
        textureDesc.size.height = std::max(desc.height, 1U);
        textureDesc.size.depthOrArrayLayers = (desc.dimension == RenderTextureDimension::TEXTURE_3D) ? std::max(desc.depth, 1U) : std::max(desc.arraySize, 1U);
        textureDesc.format = toWGPU(desc.format);
        textureDesc.mipLevelCount = std::max(desc.mipLevels, 1U);
        textureDesc.sampleCount = std::max(desc.multisampling.sampleCount, 1U);
        textureDesc.usage = WGPUTextureUsage_TextureBinding;
        if (!multisampled) {
            textureDesc.usage |= WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst;
        }

        if (desc.flags & (RenderTextureFlag::RENDER_TARGET | RenderTextureFlag::DEPTH_TARGET)) {
            textureDesc.usage |= WGPUTextureUsage_RenderAttachment;
        }

        if (desc.flags & (RenderTextureFlag::STORAGE | RenderTextureFlag::UNORDERED_ACCESS)) {
            textureDesc.usage |= WGPUTextureUsage_StorageBinding;
        }

        texture = wgpuDeviceCreateTexture(device->device, &textureDesc);
        handleSerial = newHandleSerial();
    }

    WebGPUTexture::~WebGPUTexture() {
        releaseHandle();
    }

    void WebGPUTexture::releaseHandle() {
        if (storageView != nullptr) {
            wgpuTextureViewRelease(storageView);
            storageView = nullptr;
        }

        if (defaultView != nullptr) {
            wgpuTextureViewRelease(defaultView);
            defaultView = nullptr;
        }

        if (texture != nullptr) {
            if (ownsTexture) {
                wgpuTextureDestroy(texture);
            }

            wgpuTextureRelease(texture);
            texture = nullptr;
        }
    }

    WGPUTextureView WebGPUTexture::getDefaultView() {
        if ((defaultView == nullptr) && (texture != nullptr)) {
            WGPUTextureViewDescriptor viewDesc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
            viewDesc.dimension = viewDimension(desc, nullptr);
            if (desc.format == RenderFormat::D32_FLOAT_S8_UINT) {
                viewDesc.aspect = WGPUTextureAspect_All;
            }

            defaultView = wgpuTextureCreateView(texture, &viewDesc);
        }

        return defaultView;
    }

    // The view a storage slot binds for a texture set without one. WebGPU binds one mip level to a storage texture, so
    // this is the first, as D3D12's null UAV desc is: the default view of a texture with mipmaps would fail the bind
    // group, its pass and the whole submit.
    WGPUTextureView WebGPUTexture::getStorageView() {
        if (desc.mipLevels <= 1) {
            return getDefaultView();
        }

        if ((storageView == nullptr) && (texture != nullptr)) {
            WGPUTextureViewDescriptor viewDesc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
            viewDesc.dimension = viewDimension(desc, nullptr);
            viewDesc.mipLevelCount = 1;
            storageView = wgpuTextureCreateView(texture, &viewDesc);
        }

        return storageView;
    }

    std::unique_ptr<RenderTextureView> WebGPUTexture::createTextureView(const RenderTextureViewDesc &desc) const {
        return std::make_unique<WebGPUTextureView>(this, desc);
    }

    void WebGPUTexture::setName(const std::string &name) {
        if (texture != nullptr) {
            wgpuTextureSetLabel(texture, toStringView(name));
        }
    }

    // WebGPUTextureView

    WebGPUTextureView::WebGPUTextureView(const WebGPUTexture *texture, const RenderTextureViewDesc &desc) {
        assert(texture != nullptr);

        this->texture = texture;
        this->desc = desc;

        if ((desc.componentMapping.r != RenderSwizzle::IDENTITY) || (desc.componentMapping.g != RenderSwizzle::IDENTITY) ||
            (desc.componentMapping.b != RenderSwizzle::IDENTITY) || (desc.componentMapping.a != RenderSwizzle::IDENTITY))
        {
            PLUME_WEBGPU_UNSUPPORTED("texture view swizzles are ignored");
        }

        WGPUTextureViewDescriptor viewDesc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
        viewDesc.format = toWGPU((desc.format != RenderFormat::UNKNOWN) ? desc.format : texture->desc.format);
        viewDesc.dimension = viewDimension(texture->desc, &desc);
        viewDesc.baseMipLevel = desc.mipSlice;
        viewDesc.mipLevelCount = (desc.mipLevels == UINT32_MAX) ? WGPU_MIP_LEVEL_COUNT_UNDEFINED : desc.mipLevels;
        viewDesc.baseArrayLayer = desc.arrayIndex;
        viewDesc.arrayLayerCount = (desc.arraySize == UINT32_MAX) ? WGPU_ARRAY_LAYER_COUNT_UNDEFINED : desc.arraySize;
        if (RenderFormatIsDepth(texture->desc.format) && RenderFormatIsStencil(texture->desc.format)) {
            viewDesc.format = WGPUTextureFormat_Undefined;
            viewDesc.aspect = WGPUTextureAspect_DepthOnly;
        }

        view = wgpuTextureCreateView(texture->texture, &viewDesc);
        handleSerial = newHandleSerial();
    }

    WebGPUTextureView::~WebGPUTextureView() {
        if (view != nullptr) {
            wgpuTextureViewRelease(view);
        }
    }

    // WebGPUSampler

    WebGPUSampler::WebGPUSampler(WebGPUDevice *device, const RenderSamplerDesc &desc) {
        assert(device != nullptr);

        this->desc = desc;

        if (desc.mipLODBias != 0.0f) {
            PLUME_WEBGPU_UNSUPPORTED("sampler LOD bias is ignored");
        }

        WGPUSamplerDescriptor samplerDesc = WGPU_SAMPLER_DESCRIPTOR_INIT;
        samplerDesc.addressModeU = toWGPU(desc.addressU);
        samplerDesc.addressModeV = toWGPU(desc.addressV);
        samplerDesc.addressModeW = toWGPU(desc.addressW);
        samplerDesc.magFilter = (desc.magFilter == RenderFilter::LINEAR) ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
        samplerDesc.minFilter = (desc.minFilter == RenderFilter::LINEAR) ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
        samplerDesc.mipmapFilter = (desc.mipmapMode == RenderMipmapMode::LINEAR) ? WGPUMipmapFilterMode_Linear : WGPUMipmapFilterMode_Nearest;
        samplerDesc.lodMinClamp = std::max(desc.minLOD, 0.0f);
        samplerDesc.lodMaxClamp = std::min(desc.maxLOD, 32.0f);
        samplerDesc.compare = desc.comparisonEnabled ? toWGPU(desc.comparisonFunc) : WGPUCompareFunction_Undefined;

        // Anisotropy needs every filter to be linear.
        const bool allLinear = (samplerDesc.magFilter == WGPUFilterMode_Linear) && (samplerDesc.minFilter == WGPUFilterMode_Linear) && (samplerDesc.mipmapFilter == WGPUMipmapFilterMode_Linear);
        samplerDesc.maxAnisotropy = (desc.anisotropyEnabled && allLinear) ? uint16_t(std::clamp(desc.maxAnisotropy, 1U, 16U)) : 1;
        sampler = wgpuDeviceCreateSampler(device->device, &samplerDesc);
        handleSerial = newHandleSerial();
    }

    WebGPUSampler::~WebGPUSampler() {
        if (sampler != nullptr) {
            wgpuSamplerRelease(sampler);
        }
    }

    // WebGPUShader

    WebGPUShader::WebGPUShader(WebGPUDevice *device, const void *data, uint64_t size, const char *entryPointName, RenderShaderFormat format) {
        assert(device != nullptr);
        assert(format == RenderShaderFormat::WGSL);

        this->entryPointName = (entryPointName != nullptr) ? entryPointName : "main";

        // The blob may carry a terminating zero.
        const char *code = reinterpret_cast<const char *>(data);
        size_t length = size_t(size);
        while ((length > 0) && (code[length - 1] == '\0')) {
            length--;
        }

        WGPUShaderSourceWGSL wgslSource = WGPU_SHADER_SOURCE_WGSL_INIT;
        wgslSource.code = { code, length };

        WGPUShaderModuleDescriptor moduleDesc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
        moduleDesc.nextInChain = &wgslSource.chain;
        module = wgpuDeviceCreateShaderModule(device->device, &moduleDesc);

        // What the header says pipelines and layouts need from the source: its overrides, and the stage its bindings
        // are read from.
        const std::string source(code, length);
        auto numbersAfter = [&](const char *prefix, auto &&callback) {
            const size_t prefixLength = strlen(prefix);
            for (size_t i = source.find(prefix); i != std::string::npos; i = source.find(prefix, i + prefixLength)) {
                callback(uint32_t(strtoul(source.c_str() + i + prefixLength, nullptr, 10)));
            }
        };

        numbersAfter("@id(", [&](uint32_t id) { overrideIds.insert(id); });

        WGPUShaderStage stage = WGPUShaderStage_None;
        if (source.find("@vertex") != std::string::npos) {
            stage |= WGPUShaderStage_Vertex;
        }

        if (source.find("@fragment") != std::string::npos) {
            stage |= WGPUShaderStage_Fragment;
        }

        if (source.find("@compute") != std::string::npos) {
            stage |= WGPUShaderStage_Compute;
        }

        numbersAfter("@binding(", [&](uint32_t binding) {
            WGPUShaderStage &stages = device->bindingStages[binding];
            if ((stages & stage) != stage) {
                stages |= stage;
                device->bindingStagesRevision++;
            }
        });
    }

    WebGPUShader::~WebGPUShader() {
        if (module != nullptr) {
            wgpuShaderModuleRelease(module);
        }
    }

    void WebGPUShader::setName(const std::string &name) {
        wgpuShaderModuleSetLabel(module, toStringView(name));
    }

    // WebGPUShaderStage

    WebGPUShaderStage::WebGPUShaderStage(const RenderShader *shader) {
        const WebGPUShader *webgpuShader = static_cast<const WebGPUShader *>(shader);
        if (webgpuShader != nullptr) {
            module = webgpuShader->module;
            wgpuShaderModuleAddRef(module);
            entryPointName = webgpuShader->entryPointName;
            overrideIds = webgpuShader->overrideIds;
        }
    }

    WebGPUShaderStage::~WebGPUShaderStage() {
        if (module != nullptr) {
            wgpuShaderModuleRelease(module);
        }
    }

    // WebGPUDescriptorSet

    WebGPUDescriptorSet::WebGPUDescriptorSet(WebGPUDevice *device, const RenderDescriptorSetDesc &desc) {
        assert(device != nullptr);

        this->device = device;

        // Descriptors are numbered across the ranges in order. Element i of a range with more than one descriptor is
        // bound at the range's binding + i, since core WebGPU has no binding arrays.
        for (uint32_t i = 0; i < desc.descriptorRangesCount; i++) {
            const RenderDescriptorRange &range = desc.descriptorRanges[i];
            uint32_t count = range.count;
            if (desc.lastRangeIsBoundless && (i == desc.descriptorRangesCount - 1)) {
                PLUME_WEBGPU_UNSUPPORTED("boundless descriptor ranges are bound as separate bindings");
                count = desc.boundlessRangeSize;
            }

            for (uint32_t j = 0; j < count; j++) {
                Binding binding;
                binding.type = range.type;
                binding.binding = range.binding + j;
                if (range.immutableSampler != nullptr) {
                    binding.sampler = static_cast<const WebGPUSampler *>(range.immutableSampler[j]);
                }

                bindings.emplace_back(binding);
            }
        }
    }

    WebGPUDescriptorSet::~WebGPUDescriptorSet() {
        releaseGroupsWithPushConstants();
        if (bindGroup != nullptr) {
            wgpuBindGroupRelease(bindGroup);
        }
    }

    void WebGPUDescriptorSet::setBuffer(uint32_t descriptorIndex, const RenderBuffer *buffer, uint64_t bufferSize, const RenderBufferStructuredView *bufferStructuredView, const RenderBufferFormattedView *bufferFormattedView) {
        assert(descriptorIndex < bindings.size());

        Binding &binding = bindings[descriptorIndex];
        binding.buffer = static_cast<const WebGPUBuffer *>(buffer);
        binding.bufferOffset = 0;
        if (bufferStructuredView != nullptr) {
            binding.bufferOffset = uint64_t(bufferStructuredView->firstElement) * bufferStructuredView->structureByteStride;
        }

        binding.bufferSize = (bufferSize > 0) ? bufferSize : WGPU_WHOLE_SIZE;
        dirty = true;
    }

    void WebGPUDescriptorSet::setTexture(uint32_t descriptorIndex, const RenderTexture *texture, RenderTextureLayout textureLayout, const RenderTextureView *textureView) {
        assert(descriptorIndex < bindings.size());

        Binding &binding = bindings[descriptorIndex];
        binding.texture = static_cast<const WebGPUTexture *>(texture);
        binding.textureView = static_cast<const WebGPUTextureView *>(textureView);
        dirty = true;
    }

    void WebGPUDescriptorSet::setSampler(uint32_t descriptorIndex, const RenderSampler *sampler) {
        assert(descriptorIndex < bindings.size());

        bindings[descriptorIndex].sampler = static_cast<const WebGPUSampler *>(sampler);
        dirty = true;
    }

    void WebGPUDescriptorSet::setAccelerationStructure(uint32_t descriptorIndex, const RenderAccelerationStructure *accelerationStructure) {
        assert(false && "WebGPU has no acceleration structures.");
    }

    // Appends a kind and values to a cache key as their bytes: a set's keys are rebuilt every time a binding changes
    // (thousands of times a second in RT64), and formatting them as text with snprintf took ~3% of the page's thread.
    template<typename... T>
    static void appendKey(std::string &key, char kind, T... values) {
        key += kind;
        (key.append(reinterpret_cast<const char *>(&values), sizeof(values)), ...);
    }

    // Builds the bind group layout from what is bound, and the bind group for it. Unset descriptors are left out of
    // both, so a shader that reads one fails pipeline validation instead of reading garbage.
    void WebGPUDescriptorSet::update() {
        if (!dirty && (bindingStagesRevision == device->bindingStagesRevision)) {
            return;
        }

        bindingStagesRevision = device->bindingStagesRevision;

        layoutEntries.clear();
        groupEntries.clear();
        std::string key;
        std::string entriesKey;
        for (const Binding &binding : bindings) {
            WGPUBindGroupLayoutEntry layoutEntry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
            WGPUBindGroupEntry groupEntry = WGPU_BIND_GROUP_ENTRY_INIT;
            layoutEntry.binding = binding.binding;
            auto stagesIt = device->bindingStages.find(binding.binding);
            layoutEntry.visibility = (stagesIt != device->bindingStages.end()) ? stagesIt->second : WGPUShaderStage_None;
            groupEntry.binding = binding.binding;

            switch (binding.type) {
            case RenderDescriptorRangeType::CONSTANT_BUFFER:
            case RenderDescriptorRangeType::FORMATTED_BUFFER:
            case RenderDescriptorRangeType::STRUCTURED_BUFFER:
            case RenderDescriptorRangeType::BYTE_ADDRESS_BUFFER:
            case RenderDescriptorRangeType::READ_WRITE_FORMATTED_BUFFER:
            case RenderDescriptorRangeType::READ_WRITE_STRUCTURED_BUFFER:
            case RenderDescriptorRangeType::READ_WRITE_BYTE_ADDRESS_BUFFER: {
                if (binding.buffer == nullptr) {
                    continue;
                }

                if (binding.type == RenderDescriptorRangeType::CONSTANT_BUFFER) {
                    layoutEntry.buffer.type = WGPUBufferBindingType_Uniform;
                }
                else if ((binding.type == RenderDescriptorRangeType::FORMATTED_BUFFER) || (binding.type == RenderDescriptorRangeType::STRUCTURED_BUFFER) || (binding.type == RenderDescriptorRangeType::BYTE_ADDRESS_BUFFER)) {
                    layoutEntry.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
                }
                else {
                    // Writable storage isn't allowed in vertex shaders.
                    layoutEntry.buffer.type = WGPUBufferBindingType_Storage;
                    layoutEntry.visibility &= ~WGPUShaderStage_Vertex;
                }

                groupEntry.buffer = binding.buffer->buffer;
                groupEntry.offset = binding.bufferOffset;
                groupEntry.size = binding.bufferSize;
                if (groupEntry.size != WGPU_WHOLE_SIZE) {
                    // A storage binding is a multiple of 4 bytes long, and a uniform one of 16. Buffers are made that
                    // way, so the padding is there.
                    const uint64_t sizeAlignment = (layoutEntry.buffer.type == WGPUBufferBindingType_Uniform) ? 16 : 4;
                    groupEntry.size = std::min<uint64_t>(alignUp(groupEntry.size, sizeAlignment), wgpuBufferGetSize(groupEntry.buffer) - groupEntry.offset);
                }

                appendKey(key, 'b', uint32_t(binding.binding), int32_t(layoutEntry.buffer.type), int32_t(0), int32_t(0));
                break;
            }
            case RenderDescriptorRangeType::TEXTURE:
            case RenderDescriptorRangeType::READ_WRITE_TEXTURE: {
                if (binding.texture == nullptr) {
                    continue;
                }

                const RenderTextureDesc &textureDesc = binding.texture->desc;
                const RenderTextureViewDesc *viewDesc = (binding.textureView != nullptr) ? &binding.textureView->desc : nullptr;
                const RenderFormat format = ((viewDesc != nullptr) && (viewDesc->format != RenderFormat::UNKNOWN)) ? viewDesc->format : textureDesc.format;
                const WGPUTextureViewDimension dimension = viewDimension(textureDesc, viewDesc);
                if (binding.type == RenderDescriptorRangeType::READ_WRITE_TEXTURE) {
                    // Core WebGPU reads and writes one storage texture only in r32float, r32uint and r32sint, so
                    // any other format is write-only, as PlumeWGSL.cmake declares it in the shader too.
                    const bool readWrite = (format == RenderFormat::R32_FLOAT) || (format == RenderFormat::R32_UINT) || (format == RenderFormat::R32_SINT);
                    layoutEntry.storageTexture.access = readWrite ? WGPUStorageTextureAccess_ReadWrite : WGPUStorageTextureAccess_WriteOnly;
                    layoutEntry.storageTexture.format = toWGPU(format);
                    layoutEntry.storageTexture.viewDimension = dimension;
                    layoutEntry.visibility &= ~WGPUShaderStage_Vertex;
                    appendKey(key, 's', uint32_t(binding.binding), int32_t(layoutEntry.storageTexture.format), int32_t(dimension), int32_t(0));
                }
                else {
                    bool isSigned = false;
                    const bool multisampled = textureDesc.multisampling.sampleCount > 1;
                    if (RenderFormatIsDepth(format)) {
                        layoutEntry.texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
                    }
                    else if (isIntegerFormat(format, isSigned)) {
                        layoutEntry.texture.sampleType = isSigned ? WGPUTextureSampleType_Sint : WGPUTextureSampleType_Uint;
                    }
                    else if (multisampled || (isFloat32Format(format) && !device->float32Filterable)) {
                        layoutEntry.texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
                    }
                    else {
                        layoutEntry.texture.sampleType = WGPUTextureSampleType_Float;
                    }

                    layoutEntry.texture.viewDimension = dimension;
                    layoutEntry.texture.multisampled = multisampled;
                    appendKey(key, 't', uint32_t(binding.binding), int32_t(layoutEntry.texture.sampleType), int32_t(dimension), int32_t(multisampled));
                }

                WebGPUTexture *texture = const_cast<WebGPUTexture *>(binding.texture);
                if (binding.textureView != nullptr) {
                    groupEntry.textureView = binding.textureView->view;
                }
                else {
                    groupEntry.textureView = (binding.type == RenderDescriptorRangeType::READ_WRITE_TEXTURE) ? texture->getStorageView() : texture->getDefaultView();
                }
                break;
            }
            case RenderDescriptorRangeType::SAMPLER: {
                if (binding.sampler == nullptr) {
                    continue;
                }

                const RenderSamplerDesc &samplerDesc = binding.sampler->desc;
                if (samplerDesc.comparisonEnabled) {
                    layoutEntry.sampler.type = WGPUSamplerBindingType_Comparison;
                }
                else if ((samplerDesc.minFilter == RenderFilter::LINEAR) || (samplerDesc.magFilter == RenderFilter::LINEAR) || (samplerDesc.mipmapMode == RenderMipmapMode::LINEAR)) {
                    layoutEntry.sampler.type = WGPUSamplerBindingType_Filtering;
                }
                else {
                    layoutEntry.sampler.type = WGPUSamplerBindingType_NonFiltering;
                }

                groupEntry.sampler = binding.sampler->sampler;
                appendKey(key, 'p', uint32_t(binding.binding), int32_t(layoutEntry.sampler.type), int32_t(0), int32_t(0));
                break;
            }
            default:
                assert(false && "Unsupported descriptor type.");
                continue;
            }

            appendKey(key, 'v', uint64_t(layoutEntry.visibility));

            // What the entry binds, for the device's bind group cache: a texture without a view binds its default one, or
            // on a storage slot its storage view (the layout's key tells the two apart).
            uint64_t serial = 0;
            if (groupEntry.buffer != nullptr) serial = binding.buffer->handleSerial;
            else if (groupEntry.sampler != nullptr) serial = binding.sampler->handleSerial;
            else if (binding.textureView != nullptr) serial = binding.textureView->handleSerial;
            else serial = binding.texture->handleSerial;
            appendKey(entriesKey, 'e', uint32_t(binding.binding), uint64_t(serial), uint64_t(groupEntry.offset), uint64_t(groupEntry.size));
            layoutEntries.emplace_back(layoutEntry);
            groupEntries.emplace_back(groupEntry);
        }

        layout = device->getBindGroupLayout(key, layoutEntries);
        layoutKey = key;
        layoutWithPushConstants = nullptr;
        releaseGroupsWithPushConstants();

        if (bindGroup != nullptr) {
            wgpuBindGroupRelease(bindGroup);
        }

        // A layout lives as long as the device, so its address names it.
        groupKey.clear();
        appendKey(groupKey, 'l', reinterpret_cast<uintptr_t>(layout));
        groupKey += entriesKey;
        bindGroup = device->getBindGroup(groupKey, layout, groupEntries);
        dirty = false;
    }

    // A set in group 3 of a layout with push constants shares the group with them: its layout and bind groups get the
    // push constant block's binding too, a bind group for each ring chunk and block size it is drawn with.
    WGPUBindGroupLayout WebGPUDescriptorSet::getLayoutWithPushConstants() {
        if (layoutWithPushConstants == nullptr) {
            std::vector<WGPUBindGroupLayoutEntry> entries = layoutEntries;
            entries.emplace_back(device->pushConstantsEntry);
            layoutWithPushConstants = device->getBindGroupLayout(layoutKey + "pushConstants;", entries);
        }

        return layoutWithPushConstants;
    }

    // Keyed by the chunk's id and not its buffer: a command list's chunks die with it, and a later buffer at the same
    // address must not find a group of the destroyed one.
    WGPUBindGroup WebGPUDescriptorSet::getBindGroupWithPushConstants(WGPUBuffer buffer, uint64_t chunkId, uint32_t bindingSize) {
        WGPUBindGroup &group = groupsWithPushConstants[{ chunkId, bindingSize }];
        if (group == nullptr) {
            std::vector<WGPUBindGroupEntry> entries = groupEntries;
            WGPUBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_INIT;
            entry.binding = PushConstantsBinding;
            entry.buffer = buffer;
            entry.size = bindingSize;
            entries.emplace_back(entry);
            const std::string key = groupKey + "pushConstants:" + std::to_string(chunkId) + ":" + std::to_string(bindingSize);
            group = device->getBindGroup(key, getLayoutWithPushConstants(), entries);
        }

        return group;
    }

    void WebGPUDescriptorSet::releaseGroupsWithPushConstants() {
        for (auto &it : groupsWithPushConstants) {
            wgpuBindGroupRelease(it.second);
        }

        groupsWithPushConstants.clear();
    }

    // WebGPUPipeline

    WebGPUPipeline::WebGPUPipeline(WebGPUDevice *device, Type type) {
        assert(device != nullptr);
        assert(type != Type::Unknown);

        this->device = device;
        this->type = type;
    }

    WebGPUPipeline::~WebGPUPipeline() { }

    void WebGPUPipeline::setName(const std::string &name) { }

    RenderPipelineProgram WebGPUPipeline::getProgram(const std::string &name) const {
        assert(false && "Pipeline programs are only for raytracing.");
        return RenderPipelineProgram();
    }

    static std::vector<WGPUConstantEntry> toConstantEntries(const std::vector<RenderSpecConstant> &specConstants, const WebGPUShaderStage &stage, std::vector<std::string> &keys) {
        // naga turns SPIR-V specialization constants into overrides with the constant's id, and only for the ones
        // the shader uses.
        std::vector<const RenderSpecConstant *> declared;
        for (const RenderSpecConstant &constant : specConstants) {
            if (stage.overrideIds.count(constant.index) > 0) {
                declared.emplace_back(&constant);
            }
        }

        std::vector<WGPUConstantEntry> entries;
        keys.clear();
        keys.reserve(declared.size());
        for (const RenderSpecConstant *constant : declared) {
            keys.emplace_back(std::to_string(constant->index));
        }

        for (size_t i = 0; i < declared.size(); i++) {
            WGPUConstantEntry entry = WGPU_CONSTANT_ENTRY_INIT;
            entry.key = toStringView(keys[i]);
            entry.value = double(declared[i]->value);
            entries.emplace_back(entry);
        }

        return entries;
    }

    // WebGPUComputePipeline

    WebGPUComputePipeline::WebGPUComputePipeline(WebGPUDevice *device, const RenderComputePipelineDesc &desc) : WebGPUPipeline(device, Type::Compute), computeStage(desc.computeShader) {
        this->desc = desc;
        this->desc.computeShader = nullptr;
        specConstants.assign(desc.specConstants, desc.specConstants + desc.specConstantsCount);
        this->desc.specConstants = nullptr;
    }

    WebGPUComputePipeline::~WebGPUComputePipeline() {
        for (auto &it : pipelines) {
            wgpuComputePipelineRelease(it.second);
        }
    }

    WGPUComputePipeline WebGPUComputePipeline::get(WGPUPipelineLayout layout) {
        auto it = pipelines.find(layout);
        if (it != pipelines.end()) {
            return it->second;
        }

        std::vector<std::string> keys;
        std::vector<WGPUConstantEntry> constants = toConstantEntries(specConstants, computeStage, keys);
        WGPUComputePipelineDescriptor pipelineDesc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
        pipelineDesc.layout = layout;
        pipelineDesc.compute.module = computeStage.module;
        pipelineDesc.compute.entryPoint = toStringView(computeStage.entryPointName);
        pipelineDesc.compute.constantCount = constants.size();
        pipelineDesc.compute.constants = constants.data();
        WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(device->device, &pipelineDesc);
        pipelines[layout] = pipeline;
        return pipeline;
    }

    // WebGPUGraphicsPipeline

    WebGPUGraphicsPipeline::WebGPUGraphicsPipeline(WebGPUDevice *device, const RenderGraphicsPipelineDesc &desc) : WebGPUPipeline(device, Type::Graphics), vertexStage(desc.vertexShader), pixelStage(desc.pixelShader) {
        if (desc.geometryShader != nullptr) {
            PLUME_WEBGPU_UNSUPPORTED("geometry shaders are ignored");
        }

        if (desc.logicOpEnabled) {
            PLUME_WEBGPU_UNSUPPORTED("logic ops are ignored");
        }

        this->desc = desc;
        inputSlots.assign(desc.inputSlots, desc.inputSlots + desc.inputSlotsCount);
        inputElements.assign(desc.inputElements, desc.inputElements + desc.inputElementsCount);
        specConstants.assign(desc.specConstants, desc.specConstants + desc.specConstantsCount);
        this->desc.inputSlots = nullptr;
        this->desc.inputElements = nullptr;
        this->desc.specConstants = nullptr;
        this->desc.vertexShader = nullptr;
        this->desc.geometryShader = nullptr;
        this->desc.pixelShader = nullptr;

        // Start building it in the background for the layouts draws with its pipeline layout have resolved to, or
        // once the first one has. Where the device builds in the background (Chrome), a pipeline created without
        // buildInBackground is too, though it is ready at once: a draw that wants it before it is built builds it in
        // line, and on Chrome's GPU main thread that held a page's presents for as long (RT64's ubershader, one
        // pipeline per depth and coverage state, made at setup: ~1 s the first time a game drew with a third state,
        // Ocarina of Time's title 25 s after its boot). Firefox builds those in line too, so it builds each when drawn.
        builds = std::make_shared<WebGPURenderPipelineBuilds>();
        builds->device = device;
        builds->pipeline = this;

        // What the browser compiles for each stage: the modules and the override constants.
        auto appendKey = [&](const void *data, size_t size) { builds->shadersKey.append(static_cast<const char *>(data), size); };
        appendKey(&vertexStage.module, sizeof(vertexStage.module));
        appendKey(&pixelStage.module, sizeof(pixelStage.module));
        for (const RenderSpecConstant &specConstant : specConstants) {
            appendKey(&specConstant, sizeof(specConstant));
        }

        const bool background = desc.buildInBackground || device->capabilities.backgroundPipelineBuilds;
        if (background && (desc.pipelineLayout != nullptr)) {
            const WebGPUPipelineLayout *pipelineLayout = static_cast<const WebGPUPipelineLayout *>(desc.pipelineLayout);
            if (pipelineLayout->resolvedLayouts.empty()) {
                builds->waitingForLayout = desc.buildInBackground;
                pipelineLayout->waitingBuilds.emplace_back(builds);
            }

            for (WGPUPipelineLayout layout : pipelineLayout->resolvedLayouts) {
                build(layout, true);
            }
        }
    }

    WebGPUGraphicsPipeline::~WebGPUGraphicsPipeline() {
        builds->pipeline = nullptr;
        for (auto &it : pipelines) {
            wgpuRenderPipelineRelease(it.second);
        }
    }

    static void renderPipelineBuilt(WGPUCreatePipelineAsyncStatus status, WGPURenderPipeline result, WGPUStringView message, void *userdata1, void *userdata2) {
        std::shared_ptr<WebGPURenderPipelineBuilds> *pointer = static_cast<std::shared_ptr<WebGPURenderPipelineBuilds> *>(userdata1);
        std::shared_ptr<WebGPURenderPipelineBuilds> builds = std::move(*pointer);
        delete pointer;

        WGPUPipelineLayout layout = static_cast<WGPUPipelineLayout>(userdata2);
        builds->pending.erase(layout);
        if ((status != WGPUCreatePipelineAsyncStatus_Success) && (status != WGPUCreatePipelineAsyncStatus_CallbackCancelled)) {
            fprintf(stderr, "plume WebGPU: a pipeline built in the background failed: %.*s\n", int(message.length), message.data);
        }

        // The builds that waited for this one's shaders: the browser has them compiled now.
        if (status != WGPUCreatePipelineAsyncStatus_CallbackCancelled) {
            auto &sharedShaderBuilds = builds->device->sharedShaderBuilds;
            auto sharedIt = sharedShaderBuilds.find({ builds->shadersKey, layout });
            if (sharedIt != sharedShaderBuilds.end()) {
                auto deferred = std::move(sharedIt->second.deferred);
                sharedShaderBuilds.erase(sharedIt);
                for (const auto &waiting : deferred) {
                    std::shared_ptr<WebGPURenderPipelineBuilds> waitingBuilds = waiting.first.lock();
                    if (waitingBuilds == nullptr) {
                        continue;
                    }

                    WebGPUGraphicsPipeline *waitingPipeline = waitingBuilds->pipeline;
                    waitingBuilds->pending.erase(waiting.second);
                    if ((waitingPipeline != nullptr) && (waitingPipeline->pipelines.find(waiting.second) == waitingPipeline->pipelines.end())) {
                        waitingPipeline->build(waiting.second, true, false);
                    }
                }
            }
        }

        if (result == nullptr) {
            return;
        }

        // Gone, or built in place by a draw that couldn't wait.
        WebGPUGraphicsPipeline *pipeline = builds->pipeline;
        if ((pipeline == nullptr) || (pipeline->pipelines.find(layout) != pipeline->pipelines.end())) {
            wgpuRenderPipelineRelease(result);
            return;
        }

        pipeline->pipelines[layout] = result;
    }

    bool WebGPUGraphicsPipeline::isReady() {
        return !desc.buildInBackground || (!builds->waitingForLayout && builds->pending.empty());
    }

    WGPURenderPipeline WebGPUGraphicsPipeline::get(WGPUPipelineLayout layout) {
        auto it = pipelines.find(layout);
        if (it != pipelines.end()) {
            return it->second;
        }

        return build(layout, false);
    }

    WGPURenderPipeline WebGPUGraphicsPipeline::build(WGPUPipelineLayout layout, bool async, bool deferIfShared) {
        // A build in the background whose shaders, constants and layout match one in flight waits for it, and starts once
        // that one is built, when the browser has them compiled: Chrome compiles each pipeline's shaders on its own, so
        // RT64's ubershader (one 240 KB pixel shader, a pipeline per depth and coverage state) built 8 times over at once.
        if (async && deferIfShared) {
            auto sharedIt = device->sharedShaderBuilds.find({ builds->shadersKey, layout });
            if (sharedIt != device->sharedShaderBuilds.end()) {
                builds->pending.insert(layout);
                sharedIt->second.deferred.emplace_back(builds, layout);
                return nullptr;
            }

            device->sharedShaderBuilds[{ builds->shadersKey, layout }];
        }

        std::vector<std::string> keys;
        std::vector<WGPUConstantEntry> constants = toConstantEntries(specConstants, vertexStage, keys);
        std::vector<std::string> pixelKeys;
        std::vector<WGPUConstantEntry> pixelConstants = toConstantEntries(specConstants, pixelStage, pixelKeys);

        // Vertex input.
        std::vector<std::vector<WGPUVertexAttribute>> attributes(inputSlots.size());
        std::vector<WGPUVertexBufferLayout> bufferLayouts;
        uint32_t maxSlotIndex = 0;
        for (const RenderInputSlot &slot : inputSlots) {
            maxSlotIndex = std::max(maxSlotIndex, slot.index + 1);
        }

        bufferLayouts.resize(maxSlotIndex);
        attributes.resize(maxSlotIndex);
        for (WGPUVertexBufferLayout &bufferLayout : bufferLayouts) {
            bufferLayout = WGPU_VERTEX_BUFFER_LAYOUT_INIT;
            // A slot no input uses stays Undefined with no attributes, which leaves it out.
            bufferLayout.stepMode = WGPUVertexStepMode_Undefined;
        }

        for (const RenderInputSlot &slot : inputSlots) {
            WGPUVertexBufferLayout &bufferLayout = bufferLayouts[slot.index];
            bufferLayout.stepMode = (slot.classification == RenderInputSlotClassification::PER_INSTANCE_DATA) ? WGPUVertexStepMode_Instance : WGPUVertexStepMode_Vertex;
            bufferLayout.arrayStride = slot.stride;
        }

        for (const RenderInputElement &element : inputElements) {
            assert(element.slotIndex < maxSlotIndex);
            WGPUVertexAttribute attribute = WGPU_VERTEX_ATTRIBUTE_INIT;
            attribute.format = toVertexFormat(element.format);
            attribute.offset = element.alignedByteOffset;
            attribute.shaderLocation = element.location;
            attributes[element.slotIndex].emplace_back(attribute);
        }

        for (uint32_t i = 0; i < maxSlotIndex; i++) {
            bufferLayouts[i].attributeCount = attributes[i].size();
            bufferLayouts[i].attributes = attributes[i].data();
        }

        WGPURenderPipelineDescriptor pipelineDesc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
        pipelineDesc.layout = layout;
        pipelineDesc.vertex.module = vertexStage.module;
        pipelineDesc.vertex.entryPoint = toStringView(vertexStage.entryPointName);
        pipelineDesc.vertex.constantCount = constants.size();
        pipelineDesc.vertex.constants = constants.data();
        pipelineDesc.vertex.bufferCount = bufferLayouts.size();
        pipelineDesc.vertex.buffers = bufferLayouts.data();
        pipelineDesc.primitive.topology = toWGPU(desc.primitiveTopology);
        pipelineDesc.primitive.frontFace = (desc.frontFace == RenderFrontFace::COUNTER_CLOCKWISE) ? WGPUFrontFace_CCW : WGPUFrontFace_CW;
        pipelineDesc.primitive.cullMode = (desc.cullMode == RenderCullMode::FRONT) ? WGPUCullMode_Front : ((desc.cullMode == RenderCullMode::BACK) ? WGPUCullMode_Back : WGPUCullMode_None);
        pipelineDesc.primitive.unclippedDepth = !desc.depthClipEnabled && device->depthClipControl;
        pipelineDesc.multisample.count = std::max(desc.multisampling.sampleCount, 1U);
        pipelineDesc.multisample.alphaToCoverageEnabled = desc.alphaToCoverageEnabled;

        WGPUDepthStencilState depthStencil = WGPU_DEPTH_STENCIL_STATE_INIT;
        if (desc.depthTargetFormat != RenderFormat::UNKNOWN) {
            depthStencil.format = toWGPU(desc.depthTargetFormat);
            depthStencil.depthWriteEnabled = (desc.depthEnabled && desc.depthWriteEnabled) ? WGPUOptionalBool_True : WGPUOptionalBool_False;
            depthStencil.depthCompare = desc.depthEnabled ? toWGPU(desc.depthFunction) : WGPUCompareFunction_Always;
            depthStencil.depthBias = desc.depthBias;
            depthStencil.depthBiasSlopeScale = desc.slopeScaledDepthBias;
            depthStencil.depthBiasClamp = desc.depthBiasClamp;
            if (desc.stencilEnabled) {
                depthStencil.stencilFront = { toWGPU(desc.stencilFrontFace.compareFunction), toWGPU(desc.stencilFrontFace.failOp), toWGPU(desc.stencilFrontFace.depthFailOp), toWGPU(desc.stencilFrontFace.passOp) };
                depthStencil.stencilBack = { toWGPU(desc.stencilBackFace.compareFunction), toWGPU(desc.stencilBackFace.failOp), toWGPU(desc.stencilBackFace.depthFailOp), toWGPU(desc.stencilBackFace.passOp) };
                depthStencil.stencilReadMask = desc.stencilReadMask;
                depthStencil.stencilWriteMask = desc.stencilWriteMask;
            }
            else {
                depthStencil.stencilReadMask = 0;
                depthStencil.stencilWriteMask = 0;
            }

            pipelineDesc.depthStencil = &depthStencil;
        }

        WGPUColorTargetState targets[RenderGraphicsPipelineDesc::MaxRenderTargets];
        WGPUBlendState blends[RenderGraphicsPipelineDesc::MaxRenderTargets];
        for (uint32_t i = 0; i < desc.renderTargetCount; i++) {
            const RenderBlendDesc &blendDesc = desc.renderTargetBlend[i];
            targets[i] = WGPU_COLOR_TARGET_STATE_INIT;
            targets[i].format = toWGPU(desc.renderTargetFormat[i]);
            targets[i].writeMask = blendDesc.renderTargetWriteMask & WGPUColorWriteMask_All;
            if (blendDesc.blendEnabled) {
                blends[i] = WGPU_BLEND_STATE_INIT;
                blends[i].color = { toWGPU(blendDesc.blendOp), toWGPU(blendDesc.srcBlend), toWGPU(blendDesc.dstBlend) };
                blends[i].alpha = { toWGPU(blendDesc.blendOpAlpha), toWGPU(blendDesc.srcBlendAlpha), toWGPU(blendDesc.dstBlendAlpha) };
                targets[i].blend = &blends[i];
            }
        }

        WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
        if (pixelStage.module != nullptr) {
            fragment.module = pixelStage.module;
            fragment.entryPoint = toStringView(pixelStage.entryPointName);
            fragment.constantCount = pixelConstants.size();
            fragment.constants = pixelConstants.data();
            fragment.targetCount = desc.renderTargetCount;
            fragment.targets = targets;
            pipelineDesc.fragment = &fragment;
        }

        if (async) {
            WGPUCreateRenderPipelineAsyncCallbackInfo callbackInfo = WGPU_CREATE_RENDER_PIPELINE_ASYNC_CALLBACK_INFO_INIT;
            callbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
            callbackInfo.callback = renderPipelineBuilt;
            callbackInfo.userdata1 = new std::shared_ptr<WebGPURenderPipelineBuilds>(builds);
            callbackInfo.userdata2 = layout;
            builds->pending.insert(layout);
            wgpuDeviceCreateRenderPipelineAsync(device->device, &pipelineDesc, callbackInfo);
            return nullptr;
        }

        WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(device->device, &pipelineDesc);
        pipelines[layout] = pipeline;
        return pipeline;
    }

    // WebGPUPipelineLayout

    WebGPUPipelineLayout::WebGPUPipelineLayout(WebGPUDevice *device, const RenderPipelineLayoutDesc &desc) {
        assert(device != nullptr);

        this->device = device;
        descriptorSetCount = desc.descriptorSetDescsCount;
        pushConstantRanges.assign(desc.pushConstantRanges, desc.pushConstantRanges + desc.pushConstantRangesCount);

        if (desc.rootDescriptorDescsCount > 0) {
            PLUME_WEBGPU_UNSUPPORTED("root descriptors are ignored");
        }

        for (const RenderPushConstantRange &range : pushConstantRanges) {
            pushConstantsSize = std::max(pushConstantsSize, range.offset + range.size);
        }

        assert(((pushConstantsSize == 0) || (descriptorSetCount <= PushConstantsGroup + 1)) && "Push constants share bind group 3, so a layout with them can have 4 descriptor sets at most.");
    }

    WebGPUPipelineLayout::~WebGPUPipelineLayout() { }

    // WebGPUSwapChain

    // The display's rate from the last intervals between animation frames. Frames come a whole number of refreshes apart,
    // but not all of them one: a page the GPU or its own thread holds back gets some two or three refreshes apart, which a
    // median read as a slower display (20-39 on a 60 Hz one), and Firefox can deliver extra callbacks between them, 0.1-5
    // ms after a frame, a quarter of them and more, which the shortest quarter read as hundreds of Hz. So each interval is
    // tried as the display's period: the callbacks are merged into frames until a frame spans 0.9 of it, a quarter of
    // those frames must be within 10% of it, and the period that puts the most frames 1-4 periods apart wins, the shorter
    // on a tie. An extra callback merges into its frame there, and frames held back still sit on the period's grid. No
    // period found keeps the rate read before.
    static uint32_t displayRate(const WebGPUFrameTimer &timer) {
        const std::vector<double> &intervals = timer.frameIntervals;
        uint32_t bestScore = 0;
        double bestPeriod = 0.0;
        double bestRate = 0.0;
        for (double period : intervals) {
            uint32_t frames = 0;
            uint32_t onTime = 0;
            uint32_t score = 0;
            double onTimeSum = 0.0;
            double span = 0.0;
            for (size_t i = 0; i < intervals.size(); i++) {
                span += intervals[(timer.frameIntervalCursor + i) % intervals.size()];
                if (span < 0.9 * period) {
                    continue;
                }

                frames++;
                if (std::abs(span - period) <= 0.1 * period) {
                    onTime++;
                    onTimeSum += span;
                }

                const double refreshes = std::round(span / period);
                if ((refreshes <= 4.0) && (std::abs(span - refreshes * period) <= 0.1 * period)) {
                    score++;
                }

                span = 0.0;
            }

            if ((4 * onTime >= frames) && ((score > bestScore) || ((score == bestScore) && (period < bestPeriod)))) {
                bestScore = score;
                bestPeriod = period;
                bestRate = 1000.0 * onTime / onTimeSum;
            }
        }

        return (bestScore > 0) ? uint32_t(std::lround(bestRate)) : timer.refreshRate;
    }

    // Keeps the last 60 intervals between animation frames, about a second of them, and the display's rate read from them
    // for getRefreshRate(). A gap of more than 100 ms is the page being hidden or its thread being busy, not the display.
    static bool timeAnimationFrame(double time, void *userData) {
        WebGPUFrameTimer *timer = static_cast<WebGPUFrameTimer *>(userData);
        if (timer->stopped) {
            delete timer;
            return false;
        }

        timer->frameCount++;
        const double interval = time - timer->lastFrameTime;
        if ((timer->lastFrameTime > 0.0) && (interval > 0.0) && (interval < 100.0)) {
            const uint32_t IntervalCount = 60;
            if (timer->frameIntervals.size() < IntervalCount) {
                timer->frameIntervals.emplace_back(interval);
            }
            else {
                timer->frameIntervals[timer->frameIntervalCursor] = interval;
                timer->frameIntervalCursor = (timer->frameIntervalCursor + 1) % IntervalCount;
            }

            // Until a few frames have been timed, 60 is what they run at on most displays.
            if (timer->frameIntervals.size() >= 5) {
                timer->refreshRate = displayRate(*timer);
                timer->frameMs = 1000.0 / std::max(timer->refreshRate, 1U);
            }
        }

        timer->lastFrameTime = time;
        timer->frameNow = emscripten_get_now();
        timer->frameSerial++;
        emscripten_futex_wake(&timer->frameSerial, INT_MAX);
        return true;
    }

    // The most presents in flight adaptPresentsInFlight() tries, and what uncapped keeps.
    static const uint32_t kMaxPresentsInFlight = 4;

    WebGPUSwapChain::WebGPUSwapChain(WebGPUCommandQueue *commandQueue, const RenderSwapChainDesc &desc) {
        assert(commandQueue != nullptr);
        assert(desc.renderWindow != nullptr);

        this->commandQueue = commandQueue;
        this->desc = desc;
        selector = desc.renderWindow;
        this->desc.renderWindow = selector.c_str();

        WGPUEmscriptenSurfaceSourceCanvasHTMLSelector canvasSource = WGPU_EMSCRIPTEN_SURFACE_SOURCE_CANVAS_HTML_SELECTOR_INIT;
        canvasSource.selector = toStringView(selector);
        WGPUSurfaceDescriptor surfaceDesc = WGPU_SURFACE_DESCRIPTOR_INIT;
        surfaceDesc.nextInChain = &canvasSource.chain;
        WGPUInstance instance = wgpuCreateInstance(nullptr);
        surface = wgpuInstanceCreateSurface(instance, &surfaceDesc);
        wgpuInstanceRelease(instance);
        plume_webgpu_hold_canvas_size(selector.c_str());

        // The canvas hands out a new texture every frame, so the swap chain's textures are proxies that acquireTexture()
        // points at it.
        const uint32_t textureCount = std::max(desc.textureCount, 1U);
        for (uint32_t i = 0; i < textureCount; i++) {
            auto texture = std::make_unique<WebGPUTexture>();
            texture->device = commandQueue->device;
            texture->ownsTexture = false;
            texture->desc = RenderTextureDesc::ColorTarget(1, 1, desc.format);
            textures.emplace_back(std::move(texture));
        }

        frameTimer = new WebGPUFrameTimer();
        emscripten_request_animation_frame_loop(timeAnimationFrame, frameTimer);

        maxPresentsInFlight = std::max(desc.maxFrameLatency, 2U);
        const int presentsInFlightQuery = plume_webgpu_query_number("presents-in-flight");
        if ((presentsInFlightQuery >= 1) && (presentsInFlightQuery <= 8)) {
            maxPresentsInFlight = uint32_t(presentsInFlightQuery);
            presentsInFlightFixed = true;
        }

        // uncapped[=<rate>] measures how fast a page can present, not how it plays: the display's rate is reported as
        // <rate> (360 without a number, 60-1000), so a caller that interpolates to it (RT64) draws as many frames as the
        // GPU and the browser take, and 4 stay in flight unless presents-in-flight says otherwise, so that a callback
        // that comes back late doesn't hold them. present() still waits for an animation frame, so the browser's frames
        // must be uncapped too: Chrome with --disable-frame-rate-limit, Firefox with layout.frame_rate 0 (a test page's
        // animation frames, a WebGPU clear each, came 3,115 and 814 a second that way, 61 without, in Chrome 155 and
        // Firefox 158).
        const int uncappedQuery = plume_webgpu_query_number("uncapped");
        if (uncappedQuery >= 0) {
            uncappedRate = (uncappedQuery == 0) ? 360U : uint32_t(std::clamp(uncappedQuery, 60, 1000));
            if (!presentsInFlightFixed) {
                maxPresentsInFlight = kMaxPresentsInFlight;
                presentsInFlightFixed = true;
            }

            printf("plume: uncapped, the display reported as %u Hz, at most %u presents in flight\n", uncappedRate, maxPresentsInFlight.load());
        }

        presentStats = (plume_webgpu_query_number("present-stats") >= 0);
        presentState->timeWorkDone = presentStats;
        if (presentStats || (presentsInFlightQuery >= 0)) {
            printf("plume: at most %u presents in flight\n", maxPresentsInFlight.load());
        }
    }

    WebGPUSwapChain::~WebGPUSwapChain() {
        if (kickTimer != 0) {
            emscripten_clear_timeout(kickTimer);
        }

        frameTimer->stopped = true;
        textures.clear();
        if (surface != nullptr) {
            wgpuSurfaceUnconfigure(surface);
            wgpuSurfaceRelease(surface);
        }
    }

    static bool animationFrame(double time, void *userData) {
        *static_cast<double *>(userData) = time;
        return false;
    }

    struct WebGPUPresentWork {
        std::shared_ptr<WebGPUPresentState> state;
        double presentTime;
    };

    static void presentWorkDone(WGPUQueueWorkDoneStatus status, WGPUStringView message, void *userdata1, void *userdata2) {
        WebGPUPresentWork *work = static_cast<WebGPUPresentWork *>(userdata1);
        work->state->inFlight--;
        emscripten_futex_wake(&work->state->inFlight, INT_MAX);
        if (work->state->timeWorkDone) {
            work->state->workDoneMs.emplace_back(emscripten_get_now() - work->presentTime);
        }

        delete work;
    }

    static void setupWorkDone(WGPUQueueWorkDoneStatus status, WGPUStringView message, void *userdata1, void *userdata2) {
        std::shared_ptr<WebGPUPresentState> *state = static_cast<std::shared_ptr<WebGPUPresentState> *>(userdata1);
        (*state)->caughtUp = true;
        delete state;
    }

    // Where the browser calls back only when it next looks for finished work (Firefox), a present's onSubmittedWorkDone
    // came back at the next frame's submission, about a frame late, and nothing polled while presents stayed under the
    // cap. That read as GPU time in present-stats' done delay and in adaptPresentsInFlight(), which raised the cap for
    // it. So while any present is in flight, an empty submission 1 ms after it and then on a timer makes the browser look
    // again (pollQueue(), at most once a millisecond): a test page's callback on a laptop's Intel UHD came back in 49.9
    // ms alone, 2 ms with one empty submission 1 ms after its pass, and a frame's in 5 ms at 60 frames a second, against
    // 18.5; Rush 2049's page's done delay fell from 16 to 4 ms. Firefox nests the timer down to 4 ms, so it costs ~150
    // empty submissions a second, and none once presents are done.
    static void kickWorkDone(void *userData) {
        WebGPUSwapChain *swapChain = static_cast<WebGPUSwapChain *>(userData);
        swapChain->kickTimer = 0;
        if (swapChain->presentState->inFlight > 0) {
            swapChain->commandQueue->device->pollQueue();
            swapChain->kickTimer = emscripten_set_timeout(kickWorkDone, 2.0, swapChain);
        }
    }

    // The browser shows the canvas when this thread returns to its event loop, so when the build can wait, this does
    // that until the next animation frame (or just once without vsync).
    bool WebGPUSwapChain::present(uint32_t textureIndex, RenderCommandSemaphore **waitSemaphores, uint32_t waitSemaphoreCount) {
        // In flight until the GPU has done everything submitted so far, which ends with this frame.
        presentState->inFlight++;
        WGPUQueueWorkDoneCallbackInfo callbackInfo = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
        callbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
        callbackInfo.callback = presentWorkDone;
        callbackInfo.userdata1 = new WebGPUPresentWork{ presentState, emscripten_get_now() };
        wgpuQueueOnSubmittedWorkDone(commandQueue->device->queue, callbackInfo);
        if (commandQueue->device->workDoneAfterSubmit && (kickTimer == 0)) {
            kickTimer = emscripten_set_timeout(kickWorkDone, 1.0, this);
        }

        const double waitedMs = presentState->waitedMs.exchange(0.0);
        statsWaitMs += waitedMs;
        statsSpunMs += presentState->spunMs.exchange(0.0);
        capWaitMs += waitedMs;
        if (presentStats) {
            statsPresents++;
            reportPresentStats();
        }

        if (!presentsInFlightFixed) {
            capWindowPresents++;
            adaptPresentsInFlight();
        }

        if (!canWait()) {
            return true;
        }

        if (!vsyncEnabled) {
            emscripten_sleep(0);
            return true;
        }

        double frameTime = 0.0;
        emscripten_request_animation_frame(animationFrame, &frameTime);
        while (frameTime == 0.0) {
            emscripten_sleep(0);
        }

        return true;
    }

    // Where a timed wait on a page's thread ends late (plume_web_waits.h: Firefox on Windows, on the system timer's 15.6
    // ms tick), RT64's sleeps of 100 us until each present's time took a tick each, and a present was late by up to a
    // tick. A wake through the futex comes at once, so there this waits for the animation frames, the display's own
    // timing, and on a timer only until the time itself, for a page whose frames have stopped (hidden): in Chrome on a
    // laptop's Intel graphics, presents on the animation frames made the GPU busier and the gaps between frames longer
    // than RT64's own sleeps, so only there.
    // Where they are judged on time, this sleeps until the time itself: woken by each animation frame that comes before
    // it (a wake, which ends a wait at once) and spinning on the clock from the last of them, no more than a frame and
    // 0.5 ms. Not a timed wait to just before it either: in a game's page in Chrome on Windows a tenth of the timed
    // waits of 1.5 ms ended 4-16 ms late, the system timer's tick, whether or not the page kept a 1 ms timer of its own
    // pending. Not RT64's own sleeps: one of its 100 us sleeps in Chrome (on an RTX 5080's 240 Hz display) now and then
    // took up to 16 ms, a tick, though the 100 us waits judged beside them ended on time, and woke its present 5-30 ms
    // past its slot, holding a frame on screen and then two at once; RT64's pacing took those drops for a GPU falling
    // behind and held Perfect Dark's page at 60 presents a second for half a play, at 120 for the rest, where the
    // display gives 240. A spin in their place woke within 0.3 ms of every slot. Woken by the frames and spun from the
    // last, the page presented 212 and 213 a second against 112 and 117 (Defection at 210-228 against 108-131), no
    // present woke 4 ms late, and its frames' p99 gap was 17-18 ms against 20-21.
    // The spin is a core's time whenever a page sleeps to its slots (the display faster than the game): from a frame 2 ms
    // before the slot, Mario Kart 64's page spun 635-646 ms a second at 174 animation frames a second and 493-526 at 57,
    // Perfect Dark's 405-412 at 172; from one 0.5 ms before, 462-483 and 304-310 at the same presents a second, and no
    // sleep woke later for it (Chrome 155, RTX 5080).
    // While timed waits are judged on time, the waits for frames that end on their time are judged (judgeTimedWait()):
    // the present thread's sleeps are what a page's presents hang on, and once they end late, sleeps here are on the
    // frames for the rest of the play. Pokemon Snap's page in Firefox 158 (uncapped, RTX 5080), held to RT64's sleeps,
    // woke 140 ms past its slots in the launcher and 0.1 ms past them once the game ran, presenting 308-330 a second,
    // but in 1 play of 4 they stayed late all through and it presented 94, and in later plays its waits turned late at
    // its menus or its level and it presented 172-176; on the animation frames it presented 206-232, and 206-218 turned
    // to them after 10 late seconds.
    bool WebGPUSwapChain::sleepUntil(std::chrono::steady_clock::time_point time) {
        if (emscripten_is_main_browser_thread()) {
            return false;
        }

        const double endMs = emscripten_get_now() + std::chrono::duration<double, std::milli>(time - std::chrono::steady_clock::now()).count();
        if (!timedWaitsEndLate()) {
            const double SpinLeadMs = 0.5;
            for (;;) {
                const uint32_t serial = frameTimer->frameSerial;
                const double now = emscripten_get_now();
                const double nextFrameMs = std::max(frameTimer->frameNow + frameTimer->frameMs, now);
                if (((nextFrameMs + SpinLeadMs) >= endMs) || timedWaitsEndLate()) {
                    break;
                }

                judgeTimedWait(&frameTimer->frameSerial, serial, endMs - SpinLeadMs - now);
            }

            const double spinStart = emscripten_get_now();
            double spinNow = spinStart;
            while (spinNow < endMs) {
                spinNow = emscripten_get_now();
            }

            double counted = presentState->spunMs;
            while (!presentState->spunMs.compare_exchange_weak(counted, counted + (spinNow - spinStart))) {
            }

            return true;
        }

        const double fromMs = endMs - frameTimer->frameMs / 2.0;
        for (;;) {
            const uint32_t serial = frameTimer->frameSerial;
            const double now = emscripten_get_now();
            if ((now >= endMs) || (frameTimer->frameNow >= fromMs)) {
                return true;
            }

            emscripten_futex_wait(&frameTimer->frameSerial, serial, endMs - now);
        }
    }

    // On a thread other than the page's, once the first present is caught up (isWaitReady() asks the device's thread for
    // that): presents leave flight in their done callbacks, which wake this, so it needs nothing of the device's thread.
    // A present the caller has queued for that thread is counted in `queued` until present() counts it in inFlight, so
    // it is never in neither. The wait ends after 100 ms all the same, for what changes without a wake (a queued present
    // that never reaches present()).
    bool WebGPUSwapChain::waitOnCaller(const std::atomic<uint32_t> &queued) {
        if (emscripten_is_main_browser_thread() || !presentState->caughtUp) {
            return false;
        }

        double startMs = 0.0;
        for (;;) {
            const uint32_t queuedNow = queued;
            const uint32_t inFlight = presentState->inFlight;
            if (((queuedNow + inFlight) < maxPresentsInFlight) || presentState->hidden) {
                break;
            }

            if (startMs == 0.0) {
                startMs = emscripten_get_now();
            }

            emscripten_futex_wait(&presentState->inFlight, inFlight, 100.0);
        }

        if (startMs > 0.0) {
            const double waitedMs = emscripten_get_now() - startMs;
            double counted = presentState->waitedMs;
            while (!presentState->waitedMs.compare_exchange_weak(counted, counted + waitedMs)) {
            }
        }

        return true;
    }

    void WebGPUSwapChain::wait() {
        if (!canWait()) {
            return;
        }

        while (!isWaitReady()) {
            emscripten_sleep(1);
        }
    }

    bool WebGPUSwapChain::isWaitReady() {
        // The first present waits until the GPU process has done everything the device was asked for before it
        // (plume_webgpu.h), and isn't counted as a wait of the presents in flight.
        if (!presentState->caughtUp) {
            if (!caughtUpAsked) {
                caughtUpAsked = true;
                WGPUQueueWorkDoneCallbackInfo callbackInfo = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
                callbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
                callbackInfo.callback = setupWorkDone;
                callbackInfo.userdata1 = new std::shared_ptr<WebGPUPresentState>(presentState);
                wgpuQueueOnSubmittedWorkDone(commandQueue->device->queue, callbackInfo);
            }

            commandQueue->device->pollQueue();
            return false;
        }

        if (presentState->inFlight < maxPresentsInFlight) {
            if (waitStartTime > 0.0) {
                const double waitedMs = emscripten_get_now() - waitStartTime;
                statsWaitMs += waitedMs;
                capWaitMs += waitedMs;
                waitStartTime = 0.0;
            }

            return true;
        }

        if (waitStartTime == 0.0) {
            waitStartTime = emscripten_get_now();
        }

        commandQueue->device->pollQueue();
        return false;
    }

    // Every 2 s: while wait() held presents for more than 10% of the time and they fell short of the display's rate, one
    // more in flight is tried, and kept if presents rose by 8% or reached the display's rate. Only from a steady window,
    // with no present more than 100 ms after the one before: one with a stall in it (a load, a wait for shaders, Chrome's
    // GPU process building a pipeline) is held by the stall, not the cap, and whatever follows it reads as a rise.
    // GoldenEye's page in Chrome went to 3 from a window at 0.8 presents a second in its launcher, and from one at 48
    // while its shaders built behind the game, each with a callback hundreds of ms late. Nor for less than 10%: that
    // build's stalls held GoldenEye's presents 20-60 ms a second, where late callbacks held Banjo's 160 and Rush 2049's 800. A GPU that is the limit
    // gains nothing from it, only a deeper queue, so it goes back, and the next try comes after 10 s, twice as long
    // each time (2 min at most). A done signal that comes back late gains from it. A number kept stays: a GPU that
    // later becomes the limit is left with at most 2 more frames queued than before.

    void WebGPUSwapChain::adaptPresentsInFlight() {
        const double now = emscripten_get_now();
        if (capLastPresent > 0.0) {
            capLongestGapMs = std::max(capLongestGapMs, now - capLastPresent);
        }

        capLastPresent = now;
        if (capWindowStart == 0.0) {
            capWindowStart = now;
            capWindowPresents = 0;
            capWaitMs = 0.0;
            capLongestGapMs = 0.0;
            return;
        }

        const double seconds = (now - capWindowStart) / 1000.0;
        if (seconds < 2.0) {
            return;
        }

        const double rate = capWindowPresents / seconds;
        const double displayRate = frameTimer->refreshRate;
        const bool held = (capLongestGapMs <= 100.0) && (capWaitMs > (100.0 * seconds)) && (rate < (displayRate * 0.95));
        if (capTrying) {
            capTrying = false;
            if ((rate >= (capTryRate * 1.08)) || (rate >= (displayRate * 0.95))) {
                capRetryMs = 10000.0;
                printf("plume: %u presents in flight (%.1f a second, from %.1f)\n", maxPresentsInFlight.load(), rate, capTryRate);
            }
            else {
                if (presentStats) {
                    printf("plume: back to %u presents in flight (%.1f a second with one more, %.1f without)\n",
                        maxPresentsInFlight - 1, rate, capTryRate);
                }

                maxPresentsInFlight--;
                capNextTry = now + capRetryMs;
                capRetryMs = std::min(capRetryMs * 2.0, 120000.0);
            }
        }
        else if (held && (maxPresentsInFlight < kMaxPresentsInFlight) && (now >= capNextTry)) {
            if (presentStats) {
                printf("plume: trying %u presents in flight (%.1f a second, %.0f ms a second waiting, "
                    "at most %.0f ms between two)\n",
                    maxPresentsInFlight + 1, rate, capWaitMs / seconds, capLongestGapMs);
            }

            capTryRate = rate;
            capTrying = true;
            maxPresentsInFlight++;
            emscripten_futex_wake(&presentState->inFlight, INT_MAX);
        }

        capWindowStart = now;
        capWindowPresents = 0;
        capWaitMs = 0.0;
        capLongestGapMs = 0.0;
    }

    // present-stats' line, every 5 s.
    void WebGPUSwapChain::reportPresentStats() {
        const double now = emscripten_get_now();
        if (statsStartTime == 0.0) {
            statsStartTime = now;
            statsPresents = 0;
            statsFrameCount = frameTimer->frameCount;
            statsWaitMs = 0.0;
            statsSpunMs = 0.0;
            presentState->workDoneMs.clear();
            return;
        }

        const double seconds = (now - statsStartTime) / 1000.0;
        if (seconds < 5.0) {
            return;
        }

        std::vector<double> &workDoneMs = presentState->workDoneMs;
        std::sort(workDoneMs.begin(), workDoneMs.end());
        auto quantile = [&](double q) {
            return workDoneMs.empty() ? 0.0 : workDoneMs[std::min(size_t(q * workDoneMs.size()), workDoneMs.size() - 1)];
        };

        printf("plume: %.1f presents a second, each done on the GPU %.0f ms after it (median; %.0f at p90, %.0f at most), "
            "%.0f ms a second waiting for fewer than %u in flight, %.0f ms a second spun on the clock to a present's time, "
            "%.1f animation frames a second (display read as %u Hz)\n",
            statsPresents / seconds, quantile(0.5), quantile(0.9), quantile(1.0), statsWaitMs / seconds, maxPresentsInFlight.load(),
            statsSpunMs / seconds, (frameTimer->frameCount - statsFrameCount) / seconds, frameTimer->refreshRate);

        statsStartTime = now;
        statsPresents = 0;
        statsFrameCount = frameTimer->frameCount;
        statsWaitMs = 0.0;
        statsSpunMs = 0.0;
        workDoneMs.clear();
    }

    void WebGPUSwapChain::getCanvasSize(uint32_t &width, uint32_t &height) const {
        int canvasWidth = 0, canvasHeight = 0;
        emscripten_get_canvas_element_size(selector.c_str(), &canvasWidth, &canvasHeight);
        width = uint32_t(std::max(canvasWidth, 0));
        height = uint32_t(std::max(canvasHeight, 0));
    }

    // The surface takes the new size at the next acquireTexture(), with the canvas's buffer.
    bool WebGPUSwapChain::resize() {
        getCanvasSize(width, height);
        if (isEmpty()) {
            return false;
        }

        for (auto &texture : textures) {
            texture->releaseHandle();
            texture->desc.width = width;
            texture->desc.height = height;
        }

        return true;
    }

    bool WebGPUSwapChain::needsResize() const {
        uint32_t canvasWidth, canvasHeight;
        getCanvasSize(canvasWidth, canvasHeight);
        return (canvasWidth != width) || (canvasHeight != height);
    }

    void WebGPUSwapChain::setVsyncEnabled(bool vsyncEnabled) {
        // The browser always paces the canvas to the display.
        this->vsyncEnabled = vsyncEnabled;
    }

    bool WebGPUSwapChain::isVsyncEnabled() const {
        return vsyncEnabled;
    }

    uint32_t WebGPUSwapChain::getWidth() const {
        return width;
    }

    uint32_t WebGPUSwapChain::getHeight() const {
        return height;
    }

    RenderTexture *WebGPUSwapChain::getTexture(uint32_t textureIndex) {
        assert(textureIndex < textures.size());
        return textures[textureIndex].get();
    }

    uint32_t WebGPUSwapChain::getTextureCount() const {
        return uint32_t(textures.size());
    }

    bool WebGPUSwapChain::acquireTexture(RenderCommandSemaphore *signalSemaphore, uint32_t *textureIndex) {
        assert(textureIndex != nullptr);

        if (isEmpty()) {
            return false;
        }

        if ((configuredWidth != width) || (configuredHeight != height)) {
            // The surface's configuration sets the canvas's width and height to its own (emdawnwebgpu), which are the
            // size asked for (plume_webgpu_hold_canvas_size()), so a size SDL asked for since resize() read it was lost:
            // a phone turned to landscape right after going fullscreen upright kept drawing upright, stretched across
            // the landscape canvas, since the canvas's box didn't change again. So the size asked for is put back after
            // it, and the next present's needsResize() sees it.
            int askedWidth = 0, askedHeight = 0;
            emscripten_get_canvas_element_size(selector.c_str(), &askedWidth, &askedHeight);
            plume_webgpu_size_canvas(selector.c_str(), width, height);
            WGPUSurfaceConfiguration config = WGPU_SURFACE_CONFIGURATION_INIT;
            config.device = commandQueue->device->device;
            config.format = toWGPU(desc.format);
            config.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopyDst;
            config.width = width;
            config.height = height;
            config.alphaMode = WGPUCompositeAlphaMode_Opaque;
            config.presentMode = WGPUPresentMode_Fifo;
            wgpuSurfaceConfigure(surface, &config);
            emscripten_set_canvas_element_size(selector.c_str(), askedWidth, askedHeight);
            configuredWidth = width;
            configuredHeight = height;
        }

        WGPUSurfaceTexture surfaceTexture = WGPU_SURFACE_TEXTURE_INIT;
        wgpuSurfaceGetCurrentTexture(surface, &surfaceTexture);
        if ((surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal) && (surfaceTexture.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal)) {
            return false;
        }

        WebGPUTexture *texture = textures[nextTextureIndex].get();
        texture->releaseHandle();
        texture->texture = surfaceTexture.texture;
        texture->handleSerial = newHandleSerial();
        *textureIndex = nextTextureIndex;
        nextTextureIndex = (nextTextureIndex + 1) % textures.size();
        return true;
    }

    RenderWindow WebGPUSwapChain::getWindow() const {
        return desc.renderWindow;
    }

    bool WebGPUSwapChain::isEmpty() const {
        return (width == 0) || (height == 0);
    }

    uint32_t WebGPUSwapChain::getRefreshRate() const {
        return (uncappedRate > 0) ? uncappedRate : frameTimer->refreshRate;
    }

    // On the page's thread it asks the document, and keeps the answer for other threads and waitOnCaller(), which it
    // wakes when the page is hidden.
    bool WebGPUSwapChain::isHidden() const {
        if (emscripten_is_main_browser_thread()) {
            const bool hidden = (plume_webgpu_document_hidden() != 0);
            if (hidden != presentState->hidden.exchange(hidden)) {
                emscripten_futex_wake(&presentState->inFlight, INT_MAX);
            }
        }

        return presentState->hidden;
    }

    // WebGPUFramebuffer

    WebGPUFramebuffer::WebGPUFramebuffer(const RenderFramebufferDesc &desc) {
        for (uint32_t i = 0; i < desc.colorAttachmentsCount; i++) {
            const WebGPUTexture *texture = static_cast<const WebGPUTexture *>(desc.colorAttachments[i]);
            colorTextures.emplace_back(texture);
            colorViews.emplace_back((desc.colorAttachmentViews != nullptr) ? static_cast<const WebGPUTextureView *>(desc.colorAttachmentViews[i]) : nullptr);
            if (i == 0) {
                width = texture->desc.width;
                height = texture->desc.height;
            }
        }

        depthTexture = static_cast<const WebGPUTexture *>(desc.depthAttachment);
        depthView = static_cast<const WebGPUTextureView *>(desc.depthAttachmentView);
        depthReadOnly = desc.depthAttachmentReadOnly;
        if ((depthTexture != nullptr) && colorTextures.empty()) {
            width = depthTexture->desc.width;
            height = depthTexture->desc.height;
        }
    }

    WebGPUFramebuffer::~WebGPUFramebuffer() { }

    uint32_t WebGPUFramebuffer::getWidth() const {
        return width;
    }

    uint32_t WebGPUFramebuffer::getHeight() const {
        return height;
    }

    // WebGPUCommandList

    WebGPUCommandList::WebGPUCommandList(WebGPUDevice *device) {
        assert(device != nullptr);

        this->device = device;
    }

    WebGPUCommandList::~WebGPUCommandList() {
        endRenderPass();
        if (encoder != nullptr) {
            wgpuCommandEncoderRelease(encoder);
        }

        if (commandBuffer != nullptr) {
            wgpuCommandBufferRelease(commandBuffer);
        }

        for (PushConstantsChunk &chunk : pushConstantsChunks) {
            for (auto &it : chunk.bindGroups) {
                wgpuBindGroupRelease(it.second);
            }

            wgpuBufferDestroy(chunk.buffer);
            wgpuBufferRelease(chunk.buffer);
        }
    }

    void WebGPUCommandList::begin() {
        assert(encoder == nullptr);

        if (commandBuffer != nullptr) {
            wgpuCommandBufferRelease(commandBuffer);
            commandBuffer = nullptr;
        }

        encoder = wgpuDeviceCreateCommandEncoder(device->device, nullptr);
        framebuffer = nullptr;
        pipeline = nullptr;
        graphics = BindState();
        compute = BindState();
        vertexBufferCount = 0;
        indexBuffer = RenderIndexBufferView();
        viewportSet = false;
        scissorSet = false;
        renderStateDirty = true;
        pushConstantsChunkIndex = 0;
        for (PushConstantsChunk &chunk : pushConstantsChunks) {
            chunk.used = 0;
        }

        readbackDestinations.clear();
    }

    void WebGPUCommandList::end() {
        assert(encoder != nullptr);

        endRenderPass();

        // The push constants recorded into this list reach the GPU before it is submitted.
        for (uint32_t i = 0; i <= pushConstantsChunkIndex && i < pushConstantsChunks.size(); i++) {
            PushConstantsChunk &chunk = pushConstantsChunks[i];
            if (chunk.used > 0) {
                wgpuQueueWriteBuffer(device->queue, chunk.buffer, 0, chunk.data.data(), chunk.used);
            }
        }

        commandBuffer = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuCommandEncoderRelease(encoder);
        encoder = nullptr;
    }

    void WebGPUCommandList::barriers(RenderBarrierStages stages, const RenderBufferBarrier *bufferBarriers, uint32_t bufferBarriersCount, const RenderTextureBarrier *textureBarriers, uint32_t textureBarriersCount) {
        // WebGPU tracks usage itself. A barrier usually means a pass's output is about to be read, which can't
        // happen inside the pass, so it ends the render pass.
        endRenderPass();
    }

    void WebGPUCommandList::dispatch(uint32_t threadGroupCountX, uint32_t threadGroupCountY, uint32_t threadGroupCountZ) {
        assert((pipeline != nullptr) && (pipeline->type == WebGPUPipeline::Type::Compute));

        endRenderPass();
        WGPUPipelineLayout layout = resolveBindings(compute);
        WGPUComputePassEncoder computePass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
        wgpuComputePassEncoderSetPipeline(computePass, const_cast<WebGPUComputePipeline *>(static_cast<const WebGPUComputePipeline *>(pipeline))->get(layout));
        bindGroups(compute, nullptr, computePass);
        wgpuComputePassEncoderDispatchWorkgroups(computePass, threadGroupCountX, threadGroupCountY, threadGroupCountZ);
        wgpuComputePassEncoderEnd(computePass);
        wgpuComputePassEncoderRelease(computePass);
    }

    void WebGPUCommandList::traceRays(uint32_t width, uint32_t height, uint32_t depth, RenderBufferReference shaderBindingTable, const RenderShaderBindingGroupsInfo &shaderBindingGroupsInfo) {
        assert(false && "WebGPU has no raytracing.");
    }

    // The bind group layouts of the bound descriptor sets, and the push constants in group 3, make the pipeline layout.
    // Without drawWith nothing is drawn with the bound pipeline now (a draw of nothing), so it is built in the background too.
    WGPUPipelineLayout WebGPUCommandList::resolveBindings(BindState &state, bool drawWith) {
        assert(state.layout != nullptr);

        // The same layout and sets as the last time, none of them changed since.
        ResolvedBindings &resolved = state.resolved;
        bool sameAsResolved = (resolved.pipelineLayout != nullptr) && (resolved.layout == state.layout) && (resolved.bindingStagesRevision == device->bindingStagesRevision);
        for (uint32_t i = 0; sameAsResolved && (i < state.layout->descriptorSetCount); i++) {
            const WebGPUDescriptorSet *descriptorSet = state.descriptorSets[i];
            sameAsResolved = (descriptorSet == resolved.descriptorSets[i]) && ((descriptorSet == nullptr) || (!descriptorSet->dirty && (descriptorSet->layout == resolved.setLayouts[i])));
        }

        if (sameAsResolved) {
            return resolved.pipelineLayout;
        }

        const bool pushConstants = state.layout->pushConstantsSize > 0;
        std::vector<WGPUBindGroupLayout> layouts;
        for (uint32_t i = 0; i < state.layout->descriptorSetCount; i++) {
            WebGPUDescriptorSet *descriptorSet = state.descriptorSets[i];
            const bool sharesPushConstants = pushConstants && (i == PushConstantsGroup);
            if (descriptorSet != nullptr) {
                descriptorSet->update();
                layouts.emplace_back(sharesPushConstants ? descriptorSet->getLayoutWithPushConstants() : descriptorSet->layout);
            }
            else {
                layouts.emplace_back(sharesPushConstants ? device->pushConstantsLayout : device->emptyLayout);
            }
        }

        if (pushConstants && (state.layout->descriptorSetCount <= PushConstantsGroup)) {
            while (layouts.size() < PushConstantsGroup) {
                layouts.emplace_back(device->emptyLayout);
            }

            layouts.emplace_back(device->pushConstantsLayout);
        }

        // Remember what the layout resolved to, for the pipelines created with it later (WebGPUGraphicsPipeline), and start
        // the ones waiting for it. A new binding stage rebuilds every set's layout, so what it resolved to before is no
        // longer drawn with.
        WGPUPipelineLayout pipelineLayout = device->getPipelineLayout(layouts);
        std::vector<WGPUPipelineLayout> &resolvedLayouts = state.layout->resolvedLayouts;
        if (state.layout->resolvedRevision != device->bindingStagesRevision) {
            state.layout->resolvedRevision = device->bindingStagesRevision;
            resolvedLayouts.clear();
        }

        if (std::find(resolvedLayouts.begin(), resolvedLayouts.end(), pipelineLayout) == resolvedLayouts.end()) {
            resolvedLayouts.emplace_back(pipelineLayout);
            for (const std::weak_ptr<WebGPURenderPipelineBuilds> &waiting : state.layout->waitingBuilds) {
                std::shared_ptr<WebGPURenderPipelineBuilds> builds = waiting.lock();
                if ((builds != nullptr) && (builds->pipeline != nullptr)) {
                    builds->waitingForLayout = false;

                    // A pipeline that is ready at once and drawn with now is built in place by this draw (get()).
                    const bool drawnNow = drawWith && (builds->pipeline == pipeline) && !builds->pipeline->desc.buildInBackground;
                    if (!drawnNow) {
                        builds->pipeline->build(pipelineLayout, true);
                    }
                }
            }

            state.layout->waitingBuilds.clear();
        }

        resolved.layout = state.layout;
        for (uint32_t i = 0; i < MaxDescriptorSets; i++) {
            resolved.descriptorSets[i] = state.descriptorSets[i];
            resolved.setLayouts[i] = (state.descriptorSets[i] != nullptr) ? state.descriptorSets[i]->layout : nullptr;
        }

        resolved.bindingStagesRevision = device->bindingStagesRevision;
        resolved.pipelineLayout = pipelineLayout;
        return pipelineLayout;
    }

    void WebGPUCommandList::bindGroups(BindState &state, WGPURenderPassEncoder renderPass, WGPUComputePassEncoder computePass) {
        auto setBindGroup = [&](uint32_t groupIndex, WGPUBindGroup bindGroup, uint32_t offsetCount, const uint32_t *offsets) {
            if (renderPass != nullptr) {
                WGPUBindGroup &passGroup = pass.bindGroups[groupIndex];
                if ((offsetCount == 0) && (passGroup == bindGroup)) {
                    return;
                }

                wgpuRenderPassEncoderSetBindGroup(renderPass, groupIndex, bindGroup, offsetCount, offsets);
                if (passGroup != nullptr) {
                    wgpuBindGroupRelease(passGroup);
                }

                // A group bound at an offset is held too, but is never the same as one bound without (its layout's
                // binding is dynamic), and the push constants' slot is tracked beside it.
                passGroup = bindGroup;
                wgpuBindGroupAddRef(passGroup);
                if (groupIndex == PushConstantsGroup) {
                    pass.pushConstantsBound = false;
                }
            }
            else {
                wgpuComputePassEncoderSetBindGroup(computePass, groupIndex, bindGroup, offsetCount, offsets);
            }
        };

        // An unbound set still needs a group in its slot. Group 3 is bound with the push constants, if there are any.
        const uint32_t pushConstantsSize = state.layout->pushConstantsSize;
        for (uint32_t i = 0; i < state.layout->descriptorSetCount; i++) {
            if ((pushConstantsSize > 0) && (i == PushConstantsGroup)) {
                continue;
            }

            WebGPUDescriptorSet *descriptorSet = state.descriptorSets[i];
            setBindGroup(i, (descriptorSet != nullptr) ? descriptorSet->bindGroup : device->emptyBindGroup, 0, nullptr);
        }

        if (pushConstantsSize == 0) {
            return;
        }

        for (uint32_t i = state.layout->descriptorSetCount; i < PushConstantsGroup; i++) {
            setBindGroup(i, device->emptyBindGroup, 0, nullptr);
        }

        // The binding's size is part of the bind group, so a chunk keeps one group per block size.
        const uint32_t bindingSize = uint32_t(alignUp(pushConstantsSize, 16));
        WebGPUDescriptorSet *sharedSet = (state.layout->descriptorSetCount > PushConstantsGroup) ? state.descriptorSets[PushConstantsGroup] : nullptr;
        auto chunkBindGroup = [&](PushConstantsChunk &chunk) {
            WGPUBindGroup &chunkGroup = chunk.bindGroups[bindingSize];
            if ((sharedSet == nullptr) && (chunkGroup == nullptr)) {
                WGPUBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_INIT;
                entry.binding = PushConstantsBinding;
                entry.buffer = chunk.buffer;
                entry.offset = 0;
                entry.size = bindingSize;
                WGPUBindGroupDescriptor groupDesc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
                groupDesc.layout = device->pushConstantsLayout;
                groupDesc.entryCount = 1;
                groupDesc.entries = &entry;
                chunkGroup = wgpuDeviceCreateBindGroup(device->device, &groupDesc);
            }

            return (sharedSet != nullptr) ? sharedSet->getBindGroupWithPushConstants(chunk.buffer, chunk.id, bindingSize) : chunkGroup;
        };

        // A render pass that has the same bytes bound already, through the same group, binds nothing again: a renderer
        // that sets its push constants before every draw (RT64's raster draws) mostly sets the same block, and on a page
        // each slot is bytes for writeBuffer and each bind a setBindGroup on the page's thread.
        state.pushConstants.resize(pushConstantsSize, 0);
        if ((renderPass != nullptr) && pass.pushConstantsBound && (pass.pushConstantsSize == pushConstantsSize)) {
            PushConstantsChunk &passChunk = pushConstantsChunks[pass.pushConstantsChunkIndex];
            if ((memcmp(passChunk.data.data() + pass.pushConstantsOffset, state.pushConstants.data(), pushConstantsSize) == 0) &&
                (chunkBindGroup(passChunk) == pass.bindGroups[PushConstantsGroup]))
            {
                state.pushConstantsDirty = false;
                return;
            }
        }

        // Every other draw or dispatch gets its own copy of the block in the ring.
        const uint32_t slotSize = uint32_t(alignUp(pushConstantsSize, PushConstantsAlignment));
        assert(slotSize <= PushConstantsChunkSize);
        if ((pushConstantsChunkIndex < pushConstantsChunks.size()) && (pushConstantsChunks[pushConstantsChunkIndex].used + slotSize > PushConstantsChunkSize)) {
            pushConstantsChunkIndex++;
        }

        if (pushConstantsChunkIndex >= pushConstantsChunks.size()) {
            PushConstantsChunk chunk;
            WGPUBufferDescriptor bufferDesc = WGPU_BUFFER_DESCRIPTOR_INIT;
            bufferDesc.size = PushConstantsChunkSize;
            bufferDesc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
            chunk.buffer = wgpuDeviceCreateBuffer(device->device, &bufferDesc);
            chunk.id = ++device->pushConstantsChunkCount;
            chunk.data.resize(PushConstantsChunkSize, 0);
            pushConstantsChunks.emplace_back(std::move(chunk));
        }

        PushConstantsChunk &chunk = pushConstantsChunks[pushConstantsChunkIndex];
        WGPUBindGroup bindGroup = chunkBindGroup(chunk);

        const uint32_t offset = chunk.used;
        memcpy(chunk.data.data() + offset, state.pushConstants.data(), pushConstantsSize);
        chunk.used += slotSize;
        setBindGroup(PushConstantsGroup, bindGroup, 1, &offset);
        if (renderPass != nullptr) {
            pass.pushConstantsBound = true;
            pass.pushConstantsChunkIndex = pushConstantsChunkIndex;
            pass.pushConstantsOffset = offset;
            pass.pushConstantsSize = pushConstantsSize;
        }

        state.pushConstantsDirty = false;
    }

    void WebGPUCommandList::endRenderPass() {
        // Clears nothing was drawn after still need their pass.
        const bool clearsPending = std::any_of(pendingColorClears.begin(), pendingColorClears.end(), [](const auto &clear) { return clear.first; }) || pendingDepthClear || pendingStencilClear;
        if ((renderPass == nullptr) && clearsPending && (framebuffer != nullptr) && (encoder != nullptr)) {
            beginRenderPass();
        }

        if (renderPass != nullptr) {
            wgpuRenderPassEncoderEnd(renderPass);
            wgpuRenderPassEncoderRelease(renderPass);
            renderPass = nullptr;
        }

        resetPassState();
    }

    void WebGPUCommandList::resetPassState() {
        for (WGPUBindGroup &group : pass.bindGroups) {
            if (group != nullptr) {
                wgpuBindGroupRelease(group);
            }
        }

        if (pass.pipeline != nullptr) {
            wgpuRenderPipelineRelease(pass.pipeline);
        }

        pass = PassState();
    }

    bool WebGPUCommandList::beginRenderPass() {
        if (renderPass != nullptr) {
            return true;
        }

        if (framebuffer == nullptr) {
            assert(false && "No framebuffer set.");
            return false;
        }

        std::vector<WGPURenderPassColorAttachment> colorAttachments;
        for (size_t i = 0; i < framebuffer->colorTextures.size(); i++) {
            WGPURenderPassColorAttachment attachment = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
            const WebGPUTextureView *view = framebuffer->colorViews[i];
            attachment.view = (view != nullptr) ? view->view : const_cast<WebGPUTexture *>(framebuffer->colorTextures[i])->getDefaultView();
            attachment.storeOp = WGPUStoreOp_Store;
            if ((i < pendingColorClears.size()) && pendingColorClears[i].first) {
                const RenderColor &color = pendingColorClears[i].second;
                attachment.loadOp = WGPULoadOp_Clear;
                attachment.clearValue = { color.r, color.g, color.b, color.a };
            }
            else {
                attachment.loadOp = WGPULoadOp_Load;
            }

            colorAttachments.emplace_back(attachment);
        }

        WGPURenderPassDepthStencilAttachment depthAttachment = WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
        WGPURenderPassDescriptor passDesc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
        if (framebuffer->depthTexture != nullptr) {
            const RenderFormat depthFormat = framebuffer->depthTexture->desc.format;
            depthAttachment.view = (framebuffer->depthView != nullptr) ? framebuffer->depthView->view : const_cast<WebGPUTexture *>(framebuffer->depthTexture)->getDefaultView();

            // The initializer leaves it NaN ("undefined"), which the browser refuses even when the value isn't used.
            depthAttachment.depthClearValue = 0.0f;
            if (framebuffer->depthReadOnly) {
                depthAttachment.depthReadOnly = true;
            }
            else {
                depthAttachment.depthLoadOp = pendingDepthClear ? WGPULoadOp_Clear : WGPULoadOp_Load;
                depthAttachment.depthStoreOp = WGPUStoreOp_Store;
                depthAttachment.depthClearValue = pendingDepthValue;
            }

            if (RenderFormatIsStencil(depthFormat)) {
                if (framebuffer->depthReadOnly) {
                    depthAttachment.stencilReadOnly = true;
                }
                else {
                    depthAttachment.stencilLoadOp = pendingStencilClear ? WGPULoadOp_Clear : WGPULoadOp_Load;
                    depthAttachment.stencilStoreOp = WGPUStoreOp_Store;
                    depthAttachment.stencilClearValue = pendingStencilValue;
                }
            }

            passDesc.depthStencilAttachment = &depthAttachment;
        }

        passDesc.colorAttachmentCount = colorAttachments.size();
        passDesc.colorAttachments = colorAttachments.data();
        renderPass = wgpuCommandEncoderBeginRenderPass(encoder, &passDesc);
        resetPassState();
        pendingColorClears.clear();
        pendingDepthClear = false;
        pendingStencilClear = false;
        renderStateDirty = true;
        return true;
    }

    bool WebGPUCommandList::prepareDraw(bool drawsNothing) {
        if (!beginRenderPass()) {
            return false;
        }

        // A draw of nothing records nothing, but resolves the bindings as a draw would, which starts the pipelines waiting
        // for that layout (WebGPUGraphicsPipeline) in the background, the bound one included: a renderer draws nothing with
        // the sets it draws with to have its pipelines built before its first draw needs them.
        assert((pipeline != nullptr) && (pipeline->type == WebGPUPipeline::Type::Graphics));
        WGPUPipelineLayout layout = resolveBindings(graphics, !drawsNothing);
        if (drawsNothing) {
            return false;
        }

        const WebGPUGraphicsPipeline *graphicsPipeline = static_cast<const WebGPUGraphicsPipeline *>(pipeline);
        setPassPipeline(const_cast<WebGPUGraphicsPipeline *>(graphicsPipeline)->get(layout));
        if (graphicsPipeline->desc.stencilEnabled && (!pass.stencilReferenceSet || (pass.stencilReference != graphicsPipeline->desc.stencilReference))) {
            wgpuRenderPassEncoderSetStencilReference(renderPass, graphicsPipeline->desc.stencilReference);
            pass.stencilReference = graphicsPipeline->desc.stencilReference;
            pass.stencilReferenceSet = true;
        }

        bindGroups(graphics, renderPass, nullptr);

        if (renderStateDirty) {
            for (uint32_t i = 0; i < vertexBufferCount; i++) {
                const RenderVertexBufferView &view = vertexBuffers[i];
                RenderVertexBufferView &passView = pass.vertexBuffers[i];
                if ((view.buffer.ref != nullptr) && ((view.buffer.ref != passView.buffer.ref) || (view.buffer.offset != passView.buffer.offset) || (view.size != passView.size))) {
                    const WebGPUBuffer *buffer = static_cast<const WebGPUBuffer *>(view.buffer.ref);
                    wgpuRenderPassEncoderSetVertexBuffer(renderPass, i, buffer->buffer, view.buffer.offset, view.size);
                    passView = view;
                }
            }

            const RenderIndexBufferView &passIndex = pass.indexBuffer;
            if ((indexBuffer.buffer.ref != nullptr) && (!pass.indexBufferSet || (indexBuffer.buffer.ref != passIndex.buffer.ref) || (indexBuffer.buffer.offset != passIndex.buffer.offset) ||
                (indexBuffer.size != passIndex.size) || (indexBuffer.format != passIndex.format)))
            {
                const WebGPUBuffer *buffer = static_cast<const WebGPUBuffer *>(indexBuffer.buffer.ref);
                wgpuRenderPassEncoderSetIndexBuffer(renderPass, buffer->buffer, toIndexFormat(indexBuffer.format), indexBuffer.buffer.offset, indexBuffer.size);
                pass.indexBuffer = indexBuffer;
                pass.indexBufferSet = true;
            }

            const RenderViewport drawViewport = viewportSet ? viewport : RenderViewport(0.0f, 0.0f, float(framebuffer->width), float(framebuffer->height), 0.0f, 1.0f);
            const RenderViewport &passViewport = pass.viewport;
            if (!pass.viewportSet || (drawViewport.x != passViewport.x) || (drawViewport.y != passViewport.y) || (drawViewport.width != passViewport.width) ||
                (drawViewport.height != passViewport.height) || (drawViewport.minDepth != passViewport.minDepth) || (drawViewport.maxDepth != passViewport.maxDepth))
            {
                wgpuRenderPassEncoderSetViewport(renderPass, drawViewport.x, drawViewport.y, drawViewport.width, drawViewport.height, drawViewport.minDepth, drawViewport.maxDepth);
                pass.viewport = drawViewport;
                pass.viewportSet = true;
            }

            if (!pass.blendConstantSet) {
                const WGPUColor blendConstant = { 0.0, 0.0, 0.0, 0.0 };
                wgpuRenderPassEncoderSetBlendConstant(renderPass, &blendConstant);
                pass.blendConstantSet = true;
            }

            // The scissor has to lie inside the attachments.
            RenderRect rect = scissorSet ? scissor : RenderRect(0, 0, framebuffer->width, framebuffer->height);
            rect.left = std::clamp<int32_t>(rect.left, 0, framebuffer->width);
            rect.top = std::clamp<int32_t>(rect.top, 0, framebuffer->height);
            rect.right = std::clamp<int32_t>(rect.right, rect.left, framebuffer->width);
            rect.bottom = std::clamp<int32_t>(rect.bottom, rect.top, framebuffer->height);
            const RenderRect &passScissor = pass.scissor;
            if (!pass.scissorSet || (rect.left != passScissor.left) || (rect.top != passScissor.top) || (rect.right != passScissor.right) || (rect.bottom != passScissor.bottom)) {
                wgpuRenderPassEncoderSetScissorRect(renderPass, rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top);
                pass.scissor = rect;
                pass.scissorSet = true;
            }

            renderStateDirty = false;
        }

        return true;
    }

    void WebGPUCommandList::setPassPipeline(WGPURenderPipeline renderPipeline) {
        if (renderPipeline == pass.pipeline) {
            return;
        }

        wgpuRenderPassEncoderSetPipeline(renderPass, renderPipeline);
        if (pass.pipeline != nullptr) {
            wgpuRenderPipelineRelease(pass.pipeline);
        }

        pass.pipeline = renderPipeline;
        wgpuRenderPipelineAddRef(pass.pipeline);
    }

    void WebGPUCommandList::drawInstanced(uint32_t vertexCountPerInstance, uint32_t instanceCount, uint32_t startVertexLocation, uint32_t startInstanceLocation) {
        if (prepareDraw((vertexCountPerInstance == 0) || (instanceCount == 0))) {
            wgpuRenderPassEncoderDraw(renderPass, vertexCountPerInstance, instanceCount, startVertexLocation, startInstanceLocation);
        }
    }

    void WebGPUCommandList::drawIndexedInstanced(uint32_t indexCountPerInstance, uint32_t instanceCount, uint32_t startIndexLocation, int32_t baseVertexLocation, uint32_t startInstanceLocation) {
        if (prepareDraw((indexCountPerInstance == 0) || (instanceCount == 0))) {
            wgpuRenderPassEncoderDrawIndexed(renderPass, indexCountPerInstance, instanceCount, startIndexLocation, baseVertexLocation, startInstanceLocation);
        }
    }

    void WebGPUCommandList::setPipeline(const RenderPipeline *pipeline) {
        this->pipeline = static_cast<const WebGPUPipeline *>(pipeline);
    }

    void WebGPUCommandList::setComputePipelineLayout(const RenderPipelineLayout *pipelineLayout) {
        compute.layout = static_cast<const WebGPUPipelineLayout *>(pipelineLayout);
    }

    void WebGPUCommandList::setPushConstants(BindState &state, uint32_t rangeIndex, const void *data, uint32_t offset, uint32_t size) {
        assert(state.layout != nullptr);
        assert(rangeIndex < state.layout->pushConstantRanges.size());

        const RenderPushConstantRange &range = state.layout->pushConstantRanges[rangeIndex];
        const uint32_t copySize = (size == 0) ? range.size : size;
        assert(offset + copySize <= range.size);
        state.pushConstants.resize(state.layout->pushConstantsSize, 0);
        memcpy(state.pushConstants.data() + range.offset + offset, data, copySize);
        state.pushConstantsDirty = true;
    }

    void WebGPUCommandList::setComputePushConstants(uint32_t rangeIndex, const void *data, uint32_t offset, uint32_t size) {
        setPushConstants(compute, rangeIndex, data, offset, size);
    }

    void WebGPUCommandList::setComputeDescriptorSet(RenderDescriptorSet *descriptorSet, uint32_t setIndex) {
        assert(setIndex < MaxDescriptorSets);
        compute.descriptorSets[setIndex] = static_cast<WebGPUDescriptorSet *>(descriptorSet);
    }

    void WebGPUCommandList::setGraphicsPipelineLayout(const RenderPipelineLayout *pipelineLayout) {
        graphics.layout = static_cast<const WebGPUPipelineLayout *>(pipelineLayout);
    }

    void WebGPUCommandList::setGraphicsPushConstants(uint32_t rangeIndex, const void *data, uint32_t offset, uint32_t size) {
        setPushConstants(graphics, rangeIndex, data, offset, size);
    }

    void WebGPUCommandList::setGraphicsDescriptorSet(RenderDescriptorSet *descriptorSet, uint32_t setIndex) {
        assert(setIndex < MaxDescriptorSets);
        graphics.descriptorSets[setIndex] = static_cast<WebGPUDescriptorSet *>(descriptorSet);
    }

    void WebGPUCommandList::setGraphicsRootDescriptor(RenderBufferReference bufferReference, uint32_t rootDescriptorIndex) {
        assert(false && "Root descriptors are D3D12 only.");
    }

    void WebGPUCommandList::setRaytracingPipelineLayout(const RenderPipelineLayout *pipelineLayout) {
        assert(false && "WebGPU has no raytracing.");
    }

    void WebGPUCommandList::setRaytracingPushConstants(uint32_t rangeIndex, const void *data, uint32_t offset, uint32_t size) {
        assert(false && "WebGPU has no raytracing.");
    }

    void WebGPUCommandList::setRaytracingDescriptorSet(RenderDescriptorSet *descriptorSet, uint32_t setIndex) {
        assert(false && "WebGPU has no raytracing.");
    }

    void WebGPUCommandList::setIndexBuffer(const RenderIndexBufferView *view) {
        indexBuffer = (view != nullptr) ? *view : RenderIndexBufferView();
        renderStateDirty = true;
    }

    void WebGPUCommandList::setVertexBuffers(uint32_t startSlot, const RenderVertexBufferView *views, uint32_t viewCount, const RenderInputSlot *inputSlots) {
        assert(startSlot + viewCount <= MaxVertexBuffers);

        for (uint32_t i = 0; i < viewCount; i++) {
            vertexBuffers[startSlot + i] = views[i];
        }

        vertexBufferCount = std::max(vertexBufferCount, startSlot + viewCount);
        renderStateDirty = true;
    }

    void WebGPUCommandList::setViewports(const RenderViewport *viewports, uint32_t count) {
        if (count > 1) {
            PLUME_WEBGPU_UNSUPPORTED("only the first of several viewports is used");
        }

        if (count > 0) {
            viewport = viewports[0];
            viewportSet = true;
            renderStateDirty = true;
        }
    }

    void WebGPUCommandList::setScissors(const RenderRect *scissorRects, uint32_t count) {
        if (count > 1) {
            PLUME_WEBGPU_UNSUPPORTED("only the first of several scissors is used");
        }

        if (count > 0) {
            scissor = scissorRects[0];
            scissorSet = true;
            renderStateDirty = true;
        }
    }

    void WebGPUCommandList::setFramebuffer(const RenderFramebuffer *framebuffer) {
        endRenderPass();
        this->framebuffer = static_cast<const WebGPUFramebuffer *>(framebuffer);
        pendingColorClears.clear();
        pendingDepthClear = false;
        pendingStencilClear = false;
    }

    void WebGPUCommandList::setDepthBias(float depthBias, float depthBiasClamp, float slopeScaledDepthBias) {
        PLUME_WEBGPU_UNSUPPORTED("dynamic depth bias is ignored");
    }

    static bool coversFramebuffer(const WebGPUFramebuffer *framebuffer, const RenderRect *clearRects, uint32_t clearRectsCount) {
        if (clearRectsCount == 0) {
            return true;
        }

        for (uint32_t i = 0; i < clearRectsCount; i++) {
            const RenderRect &rect = clearRects[i];
            if ((rect.left <= 0) && (rect.top <= 0) && (rect.right >= int32_t(framebuffer->width)) && (rect.bottom >= int32_t(framebuffer->height))) {
                return true;
            }
        }

        return false;
    }

    // A clear of rectangles draws a triangle over the target, clipped to each rectangle by the scissor, with the
    // device's clear pipeline for this framebuffer: the color is the blend constant and the depth both ends of the
    // viewport. The caller's pipeline, viewport and scissor are set again on its next draw; its bind groups stay bound,
    // since the clear pipeline's layout has none.
    void WebGPUCommandList::clearRects(int32_t colorAttachment, const RenderColor &color, bool clearDepth, bool clearStencil, float depthValue, uint32_t stencilValue, const RenderRect *rects, uint32_t rectCount) {
        WGPURenderPipeline clearPipeline = device->getClearPipeline(framebuffer, colorAttachment, clearDepth, clearStencil);
        if ((clearPipeline == nullptr) || !beginRenderPass()) {
            return;
        }

        const WGPUColor blendConstant = { color.r, color.g, color.b, color.a };
        setPassPipeline(clearPipeline);
        wgpuRenderPassEncoderSetBlendConstant(renderPass, &blendConstant);
        wgpuRenderPassEncoderSetStencilReference(renderPass, stencilValue);
        wgpuRenderPassEncoderSetViewport(renderPass, 0.0f, 0.0f, float(framebuffer->width), float(framebuffer->height), depthValue, depthValue);
        pass.blendConstantSet = false;
        pass.stencilReference = stencilValue;
        pass.stencilReferenceSet = true;
        pass.viewportSet = false;
        pass.scissorSet = false;
        for (uint32_t i = 0; i < rectCount; i++) {
            const int32_t left = std::clamp<int32_t>(rects[i].left, 0, framebuffer->width);
            const int32_t top = std::clamp<int32_t>(rects[i].top, 0, framebuffer->height);
            const int32_t right = std::clamp<int32_t>(rects[i].right, left, framebuffer->width);
            const int32_t bottom = std::clamp<int32_t>(rects[i].bottom, top, framebuffer->height);
            if ((right > left) && (bottom > top)) {
                wgpuRenderPassEncoderSetScissorRect(renderPass, left, top, right - left, bottom - top);
                wgpuRenderPassEncoderDraw(renderPass, 3, 1, 0, 0);
            }
        }

        renderStateDirty = true;
    }

    // A clear of a whole target becomes the load op of the next render pass, so a pass already open is ended first.
    void WebGPUCommandList::clearColor(uint32_t attachmentIndex, RenderColor colorValue, const RenderRect *clearRects, uint32_t clearRectsCount) {
        assert(framebuffer != nullptr);

        if (!coversFramebuffer(framebuffer, clearRects, clearRectsCount)) {
            this->clearRects(int32_t(attachmentIndex), colorValue, false, false, 0.0f, 0, clearRects, clearRectsCount);
            return;
        }

        if (renderPass != nullptr) {
            endRenderPass();
        }
        if (pendingColorClears.size() <= attachmentIndex) {
            pendingColorClears.resize(attachmentIndex + 1, { false, RenderColor() });
        }

        pendingColorClears[attachmentIndex] = { true, colorValue };
    }

    void WebGPUCommandList::clearDepthStencil(bool clearDepth, bool clearStencil, float depthValue, uint32_t stencilValue, const RenderRect *clearRects, uint32_t clearRectsCount) {
        assert(framebuffer != nullptr);

        if (!coversFramebuffer(framebuffer, clearRects, clearRectsCount)) {
            this->clearRects(-1, RenderColor(), clearDepth, clearStencil, depthValue, stencilValue, clearRects, clearRectsCount);
            return;
        }

        if (renderPass != nullptr) {
            endRenderPass();
        }
        if (clearDepth) {
            pendingDepthClear = true;
            pendingDepthValue = depthValue;
        }

        if (clearStencil) {
            pendingStencilClear = true;
            pendingStencilValue = stencilValue;
        }
    }

    void WebGPUCommandList::copyBufferRegion(RenderBufferReference dstBuffer, RenderBufferReference srcBuffer, uint64_t size) {
        endRenderPass();
        const WebGPUBuffer *dst = static_cast<const WebGPUBuffer *>(dstBuffer.ref);
        const WebGPUBuffer *src = static_cast<const WebGPUBuffer *>(srcBuffer.ref);

        // Widened to whole words (see the header). Buffers are made a whole number of words long.
        const uint64_t before = std::min(srcBuffer.offset & 3, dstBuffer.offset & 3);
        const uint64_t srcOffset = srcBuffer.offset - before;
        const uint64_t dstOffset = dstBuffer.offset - before;
        uint64_t wordSize = alignUp(size + before, 4);
        wordSize = std::min({ wordSize, wgpuBufferGetSize(src->buffer) - srcOffset, wgpuBufferGetSize(dst->buffer) - dstOffset });
        assert((((srcOffset | dstOffset | wordSize) & 3) == 0) && "Both ends of a buffer copy must be equally misaligned.");
        wgpuCommandEncoderCopyBufferToBuffer(encoder, src->buffer, srcOffset, dst->buffer, dstOffset, wordSize);
        copiedInto(dst);
    }

    // Remembers a READBACK buffer this list copies into, to be mapped once the list is submitted.
    void WebGPUCommandList::copiedInto(const RenderBuffer *buffer) {
        WebGPUBuffer *webgpuBuffer = const_cast<WebGPUBuffer *>(static_cast<const WebGPUBuffer *>(buffer));
        if ((webgpuBuffer->readback != nullptr) && (std::find(readbackDestinations.begin(), readbackDestinations.end(), webgpuBuffer) == readbackDestinations.end())) {
            readbackDestinations.emplace_back(webgpuBuffer);
        }
    }

    void WebGPUCommandList::copyTextureRegion(const RenderTextureCopyLocation &dstLocation, const RenderTextureCopyLocation &srcLocation, uint32_t dstX, uint32_t dstY, uint32_t dstZ, const RenderBox *srcBox) {
        endRenderPass();

        // A depth and stencil texture copies to or from a buffer one aspect at a time, and to another texture whole.
        auto textureInfo = [](const RenderTextureCopyLocation &location, uint32_t x, uint32_t y, uint32_t z, bool withBuffer) {
            const WebGPUTexture *texture = static_cast<const WebGPUTexture *>(location.texture);
            WGPUTexelCopyTextureInfo info = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
            info.texture = texture->texture;
            info.mipLevel = location.subresource.mipLevel;
            info.origin = { x, y, (texture->desc.dimension == RenderTextureDimension::TEXTURE_3D) ? z : location.subresource.arrayIndex + z };
            if (withBuffer && RenderFormatIsDepth(texture->desc.format) && RenderFormatIsStencil(texture->desc.format)) {
                info.aspect = WGPUTextureAspect_DepthOnly;
            }

            return info;
        };

        auto bufferInfo = [](const RenderTextureCopyLocation &location) {
            const WebGPUBuffer *buffer = static_cast<const WebGPUBuffer *>(location.buffer);
            uint32_t bytesPerRow, rows;
            copyFootprint(location.placedFootprint.format, location.placedFootprint.rowWidth, location.placedFootprint.height, bytesPerRow, rows);
            WGPUTexelCopyBufferInfo info = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
            info.buffer = buffer->buffer;
            info.layout.offset = location.placedFootprint.offset;
            info.layout.bytesPerRow = bytesPerRow;
            info.layout.rowsPerImage = rows;
            return info;
        };

        const bool srcIsTexture = srcLocation.type == RenderTextureCopyType::SUBRESOURCE;
        const bool dstIsTexture = dstLocation.type == RenderTextureCopyType::SUBRESOURCE;
        WGPUExtent3D extent = {};
        if (srcIsTexture) {
            const WebGPUTexture *texture = static_cast<const WebGPUTexture *>(srcLocation.texture);
            const uint32_t mip = srcLocation.subresource.mipLevel;
            extent = { std::max(texture->desc.width >> mip, 1U), std::max(texture->desc.height >> mip, 1U), 1 };
            if (texture->desc.dimension == RenderTextureDimension::TEXTURE_3D) {
                extent.depthOrArrayLayers = std::max(texture->desc.depth >> mip, 1U);
            }
        }
        else {
            extent = { srcLocation.placedFootprint.width, srcLocation.placedFootprint.height, std::max(srcLocation.placedFootprint.depth, 1U) };
        }

        uint32_t srcX = 0, srcY = 0, srcZ = 0;
        if (srcBox != nullptr) {
            srcX = srcBox->left;
            srcY = srcBox->top;
            srcZ = srcBox->front;
            extent = { uint32_t(srcBox->right - srcBox->left), uint32_t(srcBox->bottom - srcBox->top), uint32_t(std::max(srcBox->back - srcBox->front, 1)) };
        }

        if (!srcIsTexture && dstIsTexture) {
            WGPUTexelCopyBufferInfo src = bufferInfo(srcLocation);
            WGPUTexelCopyTextureInfo dst = textureInfo(dstLocation, dstX, dstY, dstZ, true);
            wgpuCommandEncoderCopyBufferToTexture(encoder, &src, &dst, &extent);
        }
        else if (srcIsTexture && !dstIsTexture) {
            WGPUTexelCopyTextureInfo src = textureInfo(srcLocation, srcX, srcY, srcZ, true);
            WGPUTexelCopyBufferInfo dst = bufferInfo(dstLocation);
            wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &dst, &extent);
            copiedInto(dstLocation.buffer);
        }
        else if (srcIsTexture && dstIsTexture) {
            WGPUTexelCopyTextureInfo src = textureInfo(srcLocation, srcX, srcY, srcZ, false);
            WGPUTexelCopyTextureInfo dst = textureInfo(dstLocation, dstX, dstY, dstZ, false);
            wgpuCommandEncoderCopyTextureToTexture(encoder, &src, &dst, &extent);
        }
        else {
            assert(false && "A copy needs a texture on at least one side.");
        }
    }

    void WebGPUCommandList::copyBuffer(const RenderBuffer *dstBuffer, const RenderBuffer *srcBuffer) {
        endRenderPass();
        const WebGPUBuffer *dst = static_cast<const WebGPUBuffer *>(dstBuffer);
        const WebGPUBuffer *src = static_cast<const WebGPUBuffer *>(srcBuffer);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, src->buffer, 0, dst->buffer, 0, alignUp(std::min(dst->desc.size, src->desc.size), 4));
        copiedInto(dst);
    }

    void WebGPUCommandList::copyTexture(const RenderTexture *dstTexture, const RenderTexture *srcTexture) {
        endRenderPass();
        const WebGPUTexture *dst = static_cast<const WebGPUTexture *>(dstTexture);
        const WebGPUTexture *src = static_cast<const WebGPUTexture *>(srcTexture);
        const RenderTextureDesc &desc = src->desc;
        const uint32_t layers = (desc.dimension == RenderTextureDimension::TEXTURE_3D) ? std::max(desc.depth, 1U) : std::max(desc.arraySize, 1U);
        for (uint32_t mip = 0; mip < std::max(desc.mipLevels, 1U); mip++) {
            WGPUTexelCopyTextureInfo srcInfo = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
            srcInfo.texture = src->texture;
            srcInfo.mipLevel = mip;
            WGPUTexelCopyTextureInfo dstInfo = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
            dstInfo.texture = dst->texture;
            dstInfo.mipLevel = mip;
            WGPUExtent3D extent = { std::max(desc.width >> mip, 1U), std::max(desc.height >> mip, 1U), (desc.dimension == RenderTextureDimension::TEXTURE_3D) ? std::max(layers >> mip, 1U) : layers };
            wgpuCommandEncoderCopyTextureToTexture(encoder, &srcInfo, &dstInfo, &extent);
        }
    }

    // WebGPU resolves only at the end of a render pass, so a resolve is a pass that loads the multisampled texture
    // and draws nothing.
    void WebGPUCommandList::resolveTexture(const RenderTexture *dstTexture, const RenderTexture *srcTexture) {
        endRenderPass();
        WebGPUTexture *dst = const_cast<WebGPUTexture *>(static_cast<const WebGPUTexture *>(dstTexture));
        WebGPUTexture *src = const_cast<WebGPUTexture *>(static_cast<const WebGPUTexture *>(srcTexture));
        WGPURenderPassColorAttachment attachment = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
        attachment.view = src->getDefaultView();
        attachment.resolveTarget = dst->getDefaultView();
        attachment.loadOp = WGPULoadOp_Load;
        attachment.storeOp = WGPUStoreOp_Store;
        WGPURenderPassDescriptor passDesc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
        passDesc.colorAttachmentCount = 1;
        passDesc.colorAttachments = &attachment;
        WGPURenderPassEncoder resolvePass = wgpuCommandEncoderBeginRenderPass(encoder, &passDesc);
        wgpuRenderPassEncoderEnd(resolvePass);
        wgpuRenderPassEncoderRelease(resolvePass);
    }

    void WebGPUCommandList::resolveTextureRegion(const RenderTexture *dstTexture, uint32_t dstX, uint32_t dstY, const RenderTexture *srcTexture, const RenderRect *srcRect, RenderResolveMode resolveMode) {
        if ((dstX == 0) && (dstY == 0) && (srcRect == nullptr) && (resolveMode == RenderResolveMode::AVERAGE)) {
            resolveTexture(dstTexture, srcTexture);
        }
        else {
            PLUME_WEBGPU_UNSUPPORTED("resolving part of a texture, or not by averaging, is ignored");
        }
    }

    void WebGPUCommandList::buildBottomLevelAS(const RenderAccelerationStructure *dstAccelerationStructure, RenderBufferReference scratchBuffer, const RenderBottomLevelASBuildInfo &buildInfo) {
        assert(false && "WebGPU has no raytracing.");
    }

    void WebGPUCommandList::buildTopLevelAS(const RenderAccelerationStructure *dstAccelerationStructure, RenderBufferReference scratchBuffer, RenderBufferReference instancesBuffer, const RenderTopLevelASBuildInfo &buildInfo) {
        assert(false && "WebGPU has no raytracing.");
    }

    void WebGPUCommandList::discardTexture(const RenderTexture *texture) { }

    void WebGPUCommandList::resetQueryPool(const RenderQueryPool *queryPool, uint32_t queryFirstIndex, uint32_t queryCount) { }

    void WebGPUCommandList::writeTimestamp(const RenderQueryPool *queryPool, uint32_t queryIndex) { }

    // WebGPUCommandQueue

    WebGPUCommandQueue::WebGPUCommandQueue(WebGPUDevice *device) {
        assert(device != nullptr);

        this->device = device;
    }

    WebGPUCommandQueue::~WebGPUCommandQueue() { }

    std::unique_ptr<RenderCommandList> WebGPUCommandQueue::createCommandList() {
        return std::make_unique<WebGPUCommandList>(device);
    }

    std::unique_ptr<RenderSwapChain> WebGPUCommandQueue::createSwapChain(const RenderSwapChainDesc &desc) {
        return std::make_unique<WebGPUSwapChain>(this, desc);
    }

    void WebGPUCommandQueue::executeCommandLists(const RenderCommandList **commandLists, uint32_t commandListCount, RenderCommandSemaphore **waitSemaphores, uint32_t waitSemaphoreCount, RenderCommandSemaphore **signalSemaphores, uint32_t signalSemaphoreCount, RenderCommandFence *signalFence) {
        device->flushMappedBuffers();

        std::vector<WGPUCommandBuffer> commandBuffers;
        std::vector<WebGPUBuffer *> readbackBuffers;
        for (uint32_t i = 0; i < commandListCount; i++) {
            WebGPUCommandList *commandList = const_cast<WebGPUCommandList *>(static_cast<const WebGPUCommandList *>(commandLists[i]));
            assert(commandList->commandBuffer != nullptr && "The command list wasn't ended, or was already executed.");
            commandBuffers.emplace_back(commandList->commandBuffer);
            for (WebGPUBuffer *buffer : commandList->readbackDestinations) {
                if (std::find(readbackBuffers.begin(), readbackBuffers.end(), buffer) == readbackBuffers.end()) {
                    buffer->waitForReadback();
                    buffer->cancelReadback();
                    readbackBuffers.emplace_back(buffer);
                }
            }
        }

        wgpuQueueSubmit(device->queue, commandBuffers.size(), commandBuffers.data());
        for (WebGPUBuffer *buffer : readbackBuffers) {
            buffer->requestReadback();
        }

        for (uint32_t i = 0; i < commandListCount; i++) {
            WebGPUCommandList *commandList = const_cast<WebGPUCommandList *>(static_cast<const WebGPUCommandList *>(commandLists[i]));
            wgpuCommandBufferRelease(commandList->commandBuffer);
            commandList->commandBuffer = nullptr;
        }
    }

    void WebGPUCommandQueue::waitForCommandFence(RenderCommandFence *fence) {
        // Nothing to wait for: see plume_webgpu.h.
    }

    // WebGPUPool

    WebGPUPool::WebGPUPool(WebGPUDevice *device, const RenderPoolDesc &desc) {
        assert(device != nullptr);

        this->device = device;
        this->desc = desc;
    }

    WebGPUPool::~WebGPUPool() { }

    std::unique_ptr<RenderBuffer> WebGPUPool::createBuffer(const RenderBufferDesc &desc) {
        return std::make_unique<WebGPUBuffer>(device, this, desc);
    }

    std::unique_ptr<RenderTexture> WebGPUPool::createTexture(const RenderTextureDesc &desc) {
        return std::make_unique<WebGPUTexture>(device, this, desc);
    }

    // WebGPUQueryPool

    WebGPUQueryPool::WebGPUQueryPool(uint32_t queryCount) {
        results.resize(queryCount, 0);
    }

    WebGPUQueryPool::~WebGPUQueryPool() { }

    void WebGPUQueryPool::queryResults() { }

    const uint64_t *WebGPUQueryPool::getResults() const {
        return results.data();
    }

    uint32_t WebGPUQueryPool::getCount() const {
        return uint32_t(results.size());
    }

    // WebGPUDevice

    WebGPUDevice::WebGPUDevice(WebGPUInterface *renderInterface) {
        assert(renderInterface != nullptr);

        this->renderInterface = renderInterface;
        device = emscripten_webgpu_get_device();
        if (device == nullptr) {
            fprintf(stderr, "plume WebGPU: no device; the page must set Module.preinitializedWebGPUDevice.\n");
            return;
        }

        queue = wgpuDeviceGetQueue(device);
        float32Filterable = wgpuDeviceHasFeature(device, WGPUFeatureName_Float32Filterable);
        float32Blendable = wgpuDeviceHasFeature(device, WGPUFeatureName_Float32Blendable);
        depthClipControl = wgpuDeviceHasFeature(device, WGPUFeatureName_DepthClipControl);
        depth32FloatStencil8 = wgpuDeviceHasFeature(device, WGPUFeatureName_Depth32FloatStencil8);
        capabilities.dualSourceBlending = wgpuDeviceHasFeature(device, WGPUFeatureName_DualSourceBlending);

        WGPULimits limits = WGPU_LIMITS_INIT;
        wgpuDeviceGetLimits(device, &limits);
        capabilities.maxTextureSize = limits.maxTextureDimension2D;

        emptyLayout = getBindGroupLayout("", {});
        WGPUBindGroupDescriptor emptyGroupDesc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        emptyGroupDesc.layout = emptyLayout;
        emptyBindGroup = wgpuDeviceCreateBindGroup(device, &emptyGroupDesc);

        pushConstantsEntry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        pushConstantsEntry.binding = PushConstantsBinding;
        pushConstantsEntry.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment | WGPUShaderStage_Compute;
        pushConstantsEntry.buffer.type = WGPUBufferBindingType_Uniform;
        pushConstantsEntry.buffer.hasDynamicOffset = true;
        pushConstantsLayout = getBindGroupLayout("pushConstants", { pushConstantsEntry });
        capabilities.triangleFan = false;
        capabilities.dynamicDepthBias = false;
        capabilities.queryPools = false;
        capabilities.presentWait = true;
        capabilities.displayTiming = true;
        capabilities.backgroundPipelineBuilds = (plume_webgpu_background_pipeline_builds() != 0);
        capabilities.preferredSwapChainFormat = (plume_webgpu_bgra_swap_chain() != 0) ? RenderFormat::B8G8R8A8_UNORM : RenderFormat::R8G8B8A8_UNORM;
        description.name = "WebGPU";
        workDoneAfterSubmit = (plume_webgpu_work_done_after_submit() != 0);
    }

    WebGPUDevice::~WebGPUDevice() {
        for (auto &it : clearPipelines) {
            if (it.second != nullptr) {
                wgpuRenderPipelineRelease(it.second);
            }
        }

        if (clearShader != nullptr) {
            wgpuShaderModuleRelease(clearShader);
        }

        if (emptyBindGroup != nullptr) {
            wgpuBindGroupRelease(emptyBindGroup);
        }

        for (auto &it : pipelineLayouts) {
            wgpuPipelineLayoutRelease(it.second);
        }

        for (auto &it : bindGroups) {
            wgpuBindGroupRelease(it.second.first);
        }

        for (auto &it : bindGroupLayouts) {
            wgpuBindGroupLayoutRelease(it.second);
        }

        if (queue != nullptr) {
            wgpuQueueRelease(queue);
        }

        if (device != nullptr) {
            wgpuDeviceRelease(device);
        }
    }

    WGPUBindGroupLayout WebGPUDevice::getBindGroupLayout(const std::string &key, const std::vector<WGPUBindGroupLayoutEntry> &entries) {
        auto it = bindGroupLayouts.find(key);
        if (it != bindGroupLayouts.end()) {
            return it->second;
        }

        WGPUBindGroupLayoutDescriptor layoutDesc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
        layoutDesc.entryCount = entries.size();
        layoutDesc.entries = entries.data();
        WGPUBindGroupLayout layout = wgpuDeviceCreateBindGroupLayout(device, &layoutDesc);
        bindGroupLayouts[key] = layout;
        return layout;
    }

    // A bind group with this key (the layout and what each entry binds) made before, or a new one, with a reference for
    // the caller. The least recently used is released once there are more than BindGroupCacheSize.
    WGPUBindGroup WebGPUDevice::getBindGroup(const std::string &key, WGPUBindGroupLayout layout, const std::vector<WGPUBindGroupEntry> &entries) {
        auto it = bindGroups.find(key);
        if (it != bindGroups.end()) {
            bindGroupUses.splice(bindGroupUses.begin(), bindGroupUses, it->second.second);
            wgpuBindGroupAddRef(it->second.first);
            return it->second.first;
        }

        WGPUBindGroupDescriptor groupDesc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        groupDesc.layout = layout;
        groupDesc.entryCount = entries.size();
        groupDesc.entries = entries.data();
        WGPUBindGroup group = wgpuDeviceCreateBindGroup(device, &groupDesc);
        bindGroupUses.push_front(key);
        bindGroups[key] = { group, bindGroupUses.begin() };
        if (bindGroups.size() > BindGroupCacheSize) {
            auto oldest = bindGroups.find(bindGroupUses.back());
            wgpuBindGroupRelease(oldest->second.first);
            bindGroups.erase(oldest);
            bindGroupUses.pop_back();
        }

        wgpuBindGroupAddRef(group);
        return group;
    }

    WGPUPipelineLayout WebGPUDevice::getPipelineLayout(const std::vector<WGPUBindGroupLayout> &layouts) {
        auto it = pipelineLayouts.find(layouts);
        if (it != pipelineLayouts.end()) {
            return it->second;
        }

        WGPUPipelineLayoutDescriptor layoutDesc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
        layoutDesc.bindGroupLayoutCount = layouts.size();
        layoutDesc.bindGroupLayouts = layouts.data();
        WGPUPipelineLayout layout = wgpuDeviceCreatePipelineLayout(device, &layoutDesc);
        pipelineLayouts[layouts] = layout;
        return layout;
    }

    // Firefox's GPU process looks for finished GPU work, and calls mapAsync and onSubmittedWorkDone back, only after a
    // submission or on a timer every 100 ms (POLL_TIME_MS in its WebGPUParent.cpp). A page that waits for one of those
    // callbacks before it submits more, as a readback does, so waits for the timer whenever the GPU hadn't finished by
    // the submission's own look, which on a slow GPU is every time. While something waits, or a present is in flight
    // (kickWorkDone()), an empty submission at most once a millisecond makes it look again.
    void WebGPUDevice::pollQueue() {
        const double now = emscripten_get_now();
        if ((now - lastPollTime) >= 1.0) {
            lastPollTime = now;
            wgpuQueueSubmit(queue, 0, nullptr);
        }
    }

    void WebGPUDevice::flushMappedBuffers() {
        for (WebGPUBuffer *buffer : mappedBuffers) {
            buffer->flush(0, buffer->shadow.size());
        }
    }

    // The pipeline that clears rectangles of one color attachment (colorAttachment >= 0) or of the depth and stencil
    // (colorAttachment < 0) of framebuffers like this one: every other target is in it with nothing written, since a
    // pipeline has to match all of the pass's attachments. Null for color targets that can't blend: integer ones, and
    // 32-bit float ones without float32-blendable.
    WGPURenderPipeline WebGPUDevice::getClearPipeline(const WebGPUFramebuffer *framebuffer, int32_t colorAttachment, bool clearDepth, bool clearStencil) {
        WGPUBlendState blend = WGPU_BLEND_STATE_INIT;
        blend.color = { WGPUBlendOperation_Add, WGPUBlendFactor_Constant, WGPUBlendFactor_Zero };
        blend.alpha = { WGPUBlendOperation_Add, WGPUBlendFactor_Constant, WGPUBlendFactor_Zero };

        // A pipeline may not write the depth or stencil of a format without one.
        const RenderFormat depthFormat = (framebuffer->depthTexture != nullptr) ? framebuffer->depthTexture->desc.format : RenderFormat::UNKNOWN;
        clearDepth = clearDepth && RenderFormatIsDepth(depthFormat);
        clearStencil = clearStencil && RenderFormatIsStencil(depthFormat);

        std::string key = std::to_string(colorAttachment) + (clearDepth ? "d" : "") + (clearStencil ? "s" : "");
        uint32_t sampleCount = 1;
        std::vector<WGPUColorTargetState> targets;
        for (size_t i = 0; i < framebuffer->colorTextures.size(); i++) {
            const RenderTextureDesc &textureDesc = framebuffer->colorTextures[i]->desc;
            const WebGPUTextureView *view = framebuffer->colorViews[i];
            const RenderFormat format = ((view != nullptr) && (view->desc.format != RenderFormat::UNKNOWN)) ? view->desc.format : textureDesc.format;
            WGPUColorTargetState target = WGPU_COLOR_TARGET_STATE_INIT;
            target.format = toWGPU(format);
            target.writeMask = WGPUColorWriteMask_None;
            if (int32_t(i) == colorAttachment) {
                bool isSigned = false;
                if (isIntegerFormat(format, isSigned)) {
                    PLUME_WEBGPU_UNSUPPORTED("clears of rectangles of integer color targets are ignored");
                    return nullptr;
                }
                else if (isFloat32Format(format) && !float32Blendable) {
                    PLUME_WEBGPU_UNSUPPORTED("clears of rectangles of 32-bit float color targets need float32-blendable");
                    return nullptr;
                }

                target.blend = &blend;
                target.writeMask = WGPUColorWriteMask_All;
            }

            sampleCount = std::max(textureDesc.multisampling.sampleCount, 1U);
            targets.emplace_back(target);
            key += ";" + std::to_string(int(target.format));
        }

        WGPUDepthStencilState depthStencil = WGPU_DEPTH_STENCIL_STATE_INIT;
        if (framebuffer->depthTexture != nullptr) {
            depthStencil.format = toWGPU(framebuffer->depthTexture->desc.format);
            depthStencil.depthWriteEnabled = clearDepth ? WGPUOptionalBool_True : WGPUOptionalBool_False;
            depthStencil.depthCompare = WGPUCompareFunction_Always;
            if (clearStencil) {
                depthStencil.stencilFront = { WGPUCompareFunction_Always, WGPUStencilOperation_Keep, WGPUStencilOperation_Keep, WGPUStencilOperation_Replace };
                depthStencil.stencilBack = depthStencil.stencilFront;
                depthStencil.stencilWriteMask = 0xFF;
            }
            else {
                depthStencil.stencilReadMask = 0;
                depthStencil.stencilWriteMask = 0;
            }

            sampleCount = std::max(framebuffer->depthTexture->desc.multisampling.sampleCount, 1U);
            key += ";d" + std::to_string(int(depthStencil.format));
        }

        key += ";x" + std::to_string(sampleCount);
        auto it = clearPipelines.find(key);
        if (it != clearPipelines.end()) {
            return it->second;
        }

        if (clearShader == nullptr) {
            // A triangle over the whole viewport, and a fragment entry point per color attachment with an output at
            // its location only (fs_none for depth and stencil): it writes 1, which the blend multiplies by the constant.
            std::string code =
                "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {\n"
                "    return vec4f(f32(i & 1u) * 4.0 - 1.0, f32(i >> 1u) * 4.0 - 1.0, 0.0, 1.0);\n"
                "}\n"
                "@fragment fn fs_none() { }\n";
            for (uint32_t i = 0; i < RenderGraphicsPipelineDesc::MaxRenderTargets; i++) {
                code += "@fragment fn fs" + std::to_string(i) + "() -> @location(" + std::to_string(i) + ") vec4f { return vec4f(1.0); }\n";
            }

            WGPUShaderSourceWGSL wgslSource = WGPU_SHADER_SOURCE_WGSL_INIT;
            wgslSource.code = toStringView(code);
            WGPUShaderModuleDescriptor moduleDesc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
            moduleDesc.nextInChain = &wgslSource.chain;
            clearShader = wgpuDeviceCreateShaderModule(device, &moduleDesc);
        }

        const std::string fragmentEntryPoint = (colorAttachment >= 0) ? "fs" + std::to_string(colorAttachment) : "fs_none";
        WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
        fragment.module = clearShader;
        fragment.entryPoint = toStringView(fragmentEntryPoint);
        fragment.targetCount = targets.size();
        fragment.targets = targets.data();

        WGPURenderPipelineDescriptor pipelineDesc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
        pipelineDesc.label = { "plume clear", WGPU_STRLEN };
        pipelineDesc.layout = getPipelineLayout({});
        pipelineDesc.vertex.module = clearShader;
        pipelineDesc.vertex.entryPoint = { "vs", WGPU_STRLEN };
        pipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
        pipelineDesc.multisample.count = sampleCount;
        pipelineDesc.depthStencil = (framebuffer->depthTexture != nullptr) ? &depthStencil : nullptr;
        pipelineDesc.fragment = &fragment;
        WGPURenderPipeline clearPipeline = wgpuDeviceCreateRenderPipeline(device, &pipelineDesc);
        clearPipelines[key] = clearPipeline;
        return clearPipeline;
    }

    std::unique_ptr<RenderDescriptorSet> WebGPUDevice::createDescriptorSet(const RenderDescriptorSetDesc &desc) {
        return std::make_unique<WebGPUDescriptorSet>(this, desc);
    }

    std::unique_ptr<RenderShader> WebGPUDevice::createShader(const void *data, uint64_t size, const char *entryPointName, RenderShaderFormat format) {
        return std::make_unique<WebGPUShader>(this, data, size, entryPointName, format);
    }

    std::unique_ptr<RenderSampler> WebGPUDevice::createSampler(const RenderSamplerDesc &desc) {
        return std::make_unique<WebGPUSampler>(this, desc);
    }

    std::unique_ptr<RenderPipeline> WebGPUDevice::createComputePipeline(const RenderComputePipelineDesc &desc) {
        return std::make_unique<WebGPUComputePipeline>(this, desc);
    }

    std::unique_ptr<RenderPipeline> WebGPUDevice::createGraphicsPipeline(const RenderGraphicsPipelineDesc &desc) {
        return std::make_unique<WebGPUGraphicsPipeline>(this, desc);
    }

    std::unique_ptr<RenderPipeline> WebGPUDevice::createRaytracingPipeline(const RenderRaytracingPipelineDesc &desc, const RenderPipeline *previousPipeline) {
        assert(false && "WebGPU has no raytracing.");
        return nullptr;
    }

    std::unique_ptr<RenderCommandQueue> WebGPUDevice::createCommandQueue(RenderCommandListType type) {
        // WebGPU has one queue; every plume queue submits to it.
        return std::make_unique<WebGPUCommandQueue>(this);
    }

    std::unique_ptr<RenderBuffer> WebGPUDevice::createBuffer(const RenderBufferDesc &desc) {
        return std::make_unique<WebGPUBuffer>(this, nullptr, desc);
    }

    std::unique_ptr<RenderTexture> WebGPUDevice::createTexture(const RenderTextureDesc &desc) {
        return std::make_unique<WebGPUTexture>(this, nullptr, desc);
    }

    std::unique_ptr<RenderAccelerationStructure> WebGPUDevice::createAccelerationStructure(const RenderAccelerationStructureDesc &desc) {
        assert(false && "WebGPU has no raytracing.");
        return nullptr;
    }

    std::unique_ptr<RenderPool> WebGPUDevice::createPool(const RenderPoolDesc &desc) {
        return std::make_unique<WebGPUPool>(this, desc);
    }

    std::unique_ptr<RenderPipelineLayout> WebGPUDevice::createPipelineLayout(const RenderPipelineLayoutDesc &desc) {
        return std::make_unique<WebGPUPipelineLayout>(this, desc);
    }

    std::unique_ptr<RenderCommandFence> WebGPUDevice::createCommandFence() {
        return std::make_unique<WebGPUCommandFence>();
    }

    std::unique_ptr<RenderCommandSemaphore> WebGPUDevice::createCommandSemaphore() {
        return std::make_unique<WebGPUCommandSemaphore>();
    }

    std::unique_ptr<RenderFramebuffer> WebGPUDevice::createFramebuffer(const RenderFramebufferDesc &desc) {
        return std::make_unique<WebGPUFramebuffer>(desc);
    }

    std::unique_ptr<RenderQueryPool> WebGPUDevice::createQueryPool(uint32_t queryCount) {
        return std::make_unique<WebGPUQueryPool>(queryCount);
    }

    void WebGPUDevice::setBottomLevelASBuildInfo(RenderBottomLevelASBuildInfo &buildInfo, const RenderBottomLevelASMesh *meshes, uint32_t meshCount, bool preferFastBuild, bool preferFastTrace) {
        assert(false && "WebGPU has no raytracing.");
    }

    void WebGPUDevice::setTopLevelASBuildInfo(RenderTopLevelASBuildInfo &buildInfo, const RenderTopLevelASInstance *instances, uint32_t instanceCount, bool preferFastBuild, bool preferFastTrace) {
        assert(false && "WebGPU has no raytracing.");
    }

    void WebGPUDevice::setShaderBindingTableInfo(RenderShaderBindingTableInfo &tableInfo, const RenderShaderBindingGroups &groups, const RenderPipeline *pipeline, RenderDescriptorSet **descriptorSets, uint32_t descriptorSetCount) {
        assert(false && "WebGPU has no raytracing.");
    }

    const RenderDeviceCapabilities &WebGPUDevice::getCapabilities() const {
        return capabilities;
    }

    const RenderDeviceDescription &WebGPUDevice::getDescription() const {
        return description;
    }

    RenderSampleCounts WebGPUDevice::getSampleCountsSupported(RenderFormat format) const {
        // Core WebGPU guarantees 1 and 4 samples for every renderable format. D32_FLOAT_S8_UINT is an optional feature,
        // depth32float-stencil8, and a device without it has no such format.
        if ((format == RenderFormat::D32_FLOAT_S8_UINT) && !depth32FloatStencil8) {
            return 0;
        }

        return RenderSampleCount::COUNT_1 | RenderSampleCount::COUNT_4;
    }

    bool WebGPUDevice::beginCapture() {
        return false;
    }

    bool WebGPUDevice::endCapture() {
        return false;
    }

    void WebGPUDevice::setReadbackCallback(void (*callback)()) {
        readbackCallback = callback;
    }

    // WebGPUInterface

    WebGPUInterface::WebGPUInterface() {
        capabilities.shaderFormat = RenderShaderFormat::WGSL;
        deviceNames.emplace_back("WebGPU");
    }

    WebGPUInterface::~WebGPUInterface() { }

    std::unique_ptr<RenderDevice> WebGPUInterface::createDevice(const std::string &preferredDeviceName) {
        auto createdDevice = std::make_unique<WebGPUDevice>(this);
        return (createdDevice->device != nullptr) ? std::move(createdDevice) : nullptr;
    }

    const RenderInterfaceCapabilities &WebGPUInterface::getCapabilities() const {
        return capabilities;
    }

    const std::vector<std::string> &WebGPUInterface::getDeviceNames() const {
        return deviceNames;
    }

    bool WebGPUInterface::isValid() const {
        return true;
    }

    std::unique_ptr<RenderInterface> CreateWebGPUInterface() {
        return std::make_unique<WebGPUInterface>();
    }
};
