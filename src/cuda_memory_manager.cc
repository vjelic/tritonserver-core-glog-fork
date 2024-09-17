// Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// 
// Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
// 
#include "cuda_memory_manager.h"

#ifndef TRITON_ENABLE_ROCM
#include <cnmem.h>
#endif
#include <string.h>

#include <set>

#include "cuda_utils.h"
#include "triton/common/logging.h"

namespace {

#define RETURN_IF_HIP_ERROR(S, MSG)                      \
  do {                                                   \
    auto status__ = (S);                                 \
    if (status__ != hipSuccess) {                        \
      return Status(                                     \
          Status::Code::INTERNAL,                        \
          (MSG) + ": " + hipGetErrorString(status__));   \
    }                                                    \
  } while (false)

std::string
PointerToString(void* ptr)
{
  std::stringstream ss;
  ss << ptr;
  return ss.str();
}

}  // namespace

namespace triton { namespace core {

std::unique_ptr<CudaMemoryManager> CudaMemoryManager::instance_;
std::mutex CudaMemoryManager::instance_mu_;

CudaMemoryManager::~CudaMemoryManager()
{
#ifndef TRITON_ENABLE_ROCM
  if (has_allocation_) {
    auto status = cnmemFinalize();
    if (status != CNMEM_STATUS_SUCCESS) {
      LOG_ERROR << "Failed to finalize ROCM memory manager: [" << status << "] "
                << cnmemGetErrorString(status);
    }
  }
#endif
}

void
CudaMemoryManager::Reset()
{
  std::lock_guard<std::mutex> lock(instance_mu_);
  instance_.reset();
}

Status
CudaMemoryManager::Create(const CudaMemoryManager::Options& options)
{
  // Ensure thread-safe creation of ROCM memory pool
  std::lock_guard<std::mutex> lock(instance_mu_);
  if (instance_ != nullptr) {
    LOG_WARNING << "New ROCM memory pools could not be created since they "
                   "already exists";
    return Status::Success;
  }

  std::set<int> supported_gpus;
  auto status = GetSupportedGPUs(
      &supported_gpus, options.min_supported_compute_capability_);
  if (status.IsOk()) {
#ifndef TRITON_ENABLE_ROCM
    std::vector<cnmemDevice_t> devices;
    for (auto gpu : supported_gpus) {
      const auto it = options.memory_pool_byte_size_.find(gpu);
      if ((it != options.memory_pool_byte_size_.end()) && (it->second != 0)) {
        devices.emplace_back();
        auto& device = devices.back();
        memset(&device, 0, sizeof(device));
        device.device = gpu;
        device.size = it->second;

        LOG_INFO << "ROCM memory pool is created on device " << device.device
                 << " with size " << device.size;
      }
    }

    if (!devices.empty()) {
      RETURN_IF_HIP_ERROR(
          cnmemInit(devices.size(), devices.data(), CNMEM_FLAGS_CANNOT_GROW),
          std::string("Failed to finalize ROCM memory manager"));
    } else {
      LOG_INFO << "ROCM memory pool disabled";
    }
#endif

    // Use to finalize CNMeM properly when out of scope
    instance_.reset(new CudaMemoryManager(!supported_gpus.empty()));
  } else {
    return Status(
        status.ErrorCode(),
        "Failed to initialize ROCM memory manager: " + status.Message());
  }

  return Status::Success;
}

Status
CudaMemoryManager::Alloc(void** ptr, uint64_t size, int64_t device_id)
{
  if (instance_ == nullptr) {
    return Status(
        Status::Code::UNAVAILABLE, "CudaMemoryManager has not been created");
  } else if (!instance_->has_allocation_) {
    return Status(
        Status::Code::UNAVAILABLE,
        "CudaMemoryManager has no preallocated ROCM memory");
  }

  int current_device;
  RETURN_IF_ROCM_ERR(
      hipGetDevice(&current_device), std::string("Failed to get device"));
  bool overridden = (current_device != device_id);
  if (overridden) {
    RETURN_IF_ROCM_ERR(
        hipSetDevice(device_id), std::string("Failed to set device"));
  }

  // Defer returning error to make sure the device is recovered
  auto err = hipMalloc(ptr, size);

  if (overridden) {
    hipSetDevice(current_device);
  }

  RETURN_IF_HIP_ERROR(
      err, std::string("Failed to allocate ROCM memory with byte size ") +
               std::to_string(size) + " on GPU " + std::to_string(device_id));
  return Status::Success;
}

Status
CudaMemoryManager::Free(void* ptr, int64_t device_id)
{
  if (instance_ == nullptr) {
    return Status(
        Status::Code::UNAVAILABLE, "CudaMemoryManager has not been created");
  } else if (!instance_->has_allocation_) {
    return Status(
        Status::Code::UNAVAILABLE,
        "CudaMemoryManager has no preallocated ROCM memory");
  }

  int current_device;
  RETURN_IF_ROCM_ERR(
      hipGetDevice(&current_device), std::string("Failed to get device"));
  bool overridden = (current_device != device_id);
  if (overridden) {
    RETURN_IF_ROCM_ERR(
        hipSetDevice(device_id), std::string("Failed to set device"));
  }

  // Defer returning error to make sure the device is recovered
  auto err = hipFree(ptr);

  if (overridden) {
    hipSetDevice(current_device);
  }

  RETURN_IF_HIP_ERROR(
      err, std::string("Failed to deallocate ROCM memory at address ") +
               PointerToString(ptr) + " on GPU " + std::to_string(device_id));
  return Status::Success;
}

}}  // namespace triton::core
