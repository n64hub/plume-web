//
// plume
//
// Copyright (c) 2024 renderbag and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file for details.
//

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace plume {
    // PLUME_LOSE_DEVICE_AFTER=<seconds>: a developer's test of what an application does when its D3D12 or Vulkan device
    // is lost, which can't be made to happen on a GPU at will (a forced TDR resets the GPU under every other program on
    // it). The first device of the process that is asked isLost() says so from that many seconds after it was first
    // asked, and its buffers map() to nullptr, as a removed D3D12 device's do. On D3D12 its buffers, textures and query
    // heaps made from then on fail to be made, and its swap chain's presents and resizes fail, with
    // DXGI_ERROR_DEVICE_REMOVED, as a removed device's do, so what uses one is tested too (a Vulkan device lost for real
    // still makes them). The device itself goes on working otherwise,
    // and a device made after it, to carry on with, is never lost this way. With PLUME_LOSE_DEVICES=<n> the first n
    // devices are, each that many seconds after it was first asked, so what an application does when the device
    // that replaced a lost one is lost too is tested.
    struct LostDeviceTest {
        // Nanoseconds of the steady clock at which the device is lost; 0 while it isn't the device being tested.
        mutable std::atomic<int64_t> lossTime { 0 };
        mutable std::atomic<bool> lossLogged { false };

        static int64_t now() {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        }

        // Called from isLost(): the first devices to be asked take the test, if the variable is set.
        void arm() const {
            static std::atomic<int> taken { 0 };
            static const int devices = (std::getenv("PLUME_LOSE_DEVICES") != nullptr) ? std::max(std::atoi(std::getenv("PLUME_LOSE_DEVICES")), 1) : 1;
            if ((lossTime.load(std::memory_order_relaxed) != 0) || (taken.load(std::memory_order_relaxed) >= devices)) {
                return;
            }

            const char *seconds = std::getenv("PLUME_LOSE_DEVICE_AFTER");
            if ((seconds == nullptr) || (seconds[0] == '\0') || (taken.fetch_add(1) >= devices)) {
                return;
            }

            lossTime = now() + int64_t(std::atof(seconds) * 1e9);
        }

        bool lost() const {
            const int64_t time = lossTime.load(std::memory_order_relaxed);
            if ((time == 0) || (now() < time)) {
                return false;
            }

            if (!lossLogged.exchange(true)) {
                fprintf(stderr, "plume: the device is lost, as PLUME_LOSE_DEVICE_AFTER asks.\n");
            }

            return true;
        }
    };
};
