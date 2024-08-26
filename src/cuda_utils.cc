// Copyright 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cuda_utils.h"

#include "model_config_utils.h"
#include "triton/common/nvtx.h"

namespace triton { namespace core {

#ifdef TRITON_ENABLE_ROCM
static void
MemcpyHost(void* args)
{
  auto* copy_params = reinterpret_cast<CopyParams*>(args);
  memcpy(copy_params->dst_, copy_params->src_, copy_params->byte_size_);
  delete copy_params;
}
#endif  // TRITON_ENABLE_GPU

Status
GetDeviceMemoryInfo(const int device_id, size_t* free, size_t* total)
{
  *free = 0;
  *total = 0;
#ifdef TRITON_ENABLE_ROCM
  // Make sure that correct device is set before creating stream and
  // then restore the device to what was set by the caller.
  int current_device;
  auto cuerr = hipGetDevice(&current_device);
  bool overridden = false;
  if (cuerr == hipSuccess) {
    overridden = (current_device != device_id);
    if (overridden) {
      cuerr = hipSetDevice(device_id);
    }
  }

  if (cuerr == hipSuccess) {
    cuerr = hipMemGetInfo(free, total);
  }

  if (overridden) {
    hipSetDevice(current_device);
  }

  if (cuerr != hipSuccess) {
    return Status(
        Status::Code::INTERNAL,
        (std::string("unable to get memory info for device ") +
         std::to_string(device_id) + ": " + hipGetErrorString(cuerr)));
  }
#endif  // TRITON_ENABLE_GPU
  return Status::Success;
}

Status
EnablePeerAccess(const double min_compute_capability)
{
#ifdef TRITON_ENABLE_ROCM
  // If we can't enable peer access for one device pair, the best we can
  // do is skipping it...
  std::set<int> supported_gpus;
  bool all_enabled = false;
  if (GetSupportedGPUs(&supported_gpus, min_compute_capability).IsOk()) {
    all_enabled = true;
    int can_access_peer = false;
    for (const auto& host : supported_gpus) {
      auto cuerr = hipSetDevice(host);

      if (cuerr == hipSuccess) {
        for (const auto& peer : supported_gpus) {
          if (host == peer) {
            continue;
          }

          cuerr = hipDeviceCanAccessPeer(&can_access_peer, host, peer);
          if ((cuerr == hipSuccess) && (can_access_peer == 1)) {
            cuerr = hipDeviceEnablePeerAccess(peer, 0);
          }

          all_enabled &= ((cuerr == hipSuccess) && (can_access_peer == 1));
        }
      }
    }
  }
  if (!all_enabled) {
    return Status(
        Status::Code::UNSUPPORTED,
        "failed to enable peer access for some device pairs");
  }
#endif  // TRITON_ENABLE_GPU
  return Status::Success;
}

Status
CopyBuffer(
    const std::string& msg, const TRITONSERVER_MemoryType src_memory_type,
    const int64_t src_memory_type_id,
    const TRITONSERVER_MemoryType dst_memory_type,
    const int64_t dst_memory_type_id, const size_t byte_size, const void* src,
    void* dst, hipStream_t hip_stream, bool* rocm_used, bool copy_on_stream)
{
  NVTX_RANGE(nvtx_, "CopyBuffer");

  *rocm_used = false;

  // For ROCM memcpy, all host to host copy will be blocked in respect to the
  // host, so use memcpy() directly. In this case, need to be careful on whether
  // the src buffer is valid.
  if ((src_memory_type != TRITONSERVER_MEMORY_GPU) &&
      (dst_memory_type != TRITONSERVER_MEMORY_GPU)) {
#ifdef TRITON_ENABLE_ROCM
    if (copy_on_stream) {
      auto params = new CopyParams(dst, src, byte_size);
      hipLaunchHostFunc(
          hip_stream, MemcpyHost, reinterpret_cast<void*>(params));
      *rocm_used = true;
    } else {
      memcpy(dst, src, byte_size);
    }
#else
    memcpy(dst, src, byte_size);
#endif  // TRITON_ENABLE_GPU
  } else {
#ifdef TRITON_ENABLE_ROCM
    RETURN_IF_ROCM_ERR(
        hipMemcpyAsync(dst, src, byte_size, hipMemcpyDefault, hip_stream),
        msg + ": failed to perform ROCM copy");

    *rocm_used = true;
#else
    return Status(
        Status::Code::INTERNAL,
        msg + ": try to use ROCM copy while GPU is not supported");
#endif  // TRITON_ENABLE_GPU
  }

  return Status::Success;
}

void
CopyBufferHandler(
    const std::string& msg, const TRITONSERVER_MemoryType src_memory_type,
    const int64_t src_memory_type_id,
    const TRITONSERVER_MemoryType dst_memory_type,
    const int64_t dst_memory_type_id, const size_t byte_size, const void* src,
    void* dst, hipStream_t hip_stream, void* response_ptr,
    triton::common::SyncQueue<std::tuple<Status, bool, void*>>*
        completion_queue)
{
  bool rocm_used = false;
  Status status = CopyBuffer(
      msg, src_memory_type, src_memory_type_id, dst_memory_type,
      dst_memory_type_id, byte_size, src, dst, hip_stream, &rocm_used);
  completion_queue->Put(std::make_tuple(status, rocm_used, response_ptr));
}

#ifdef TRITON_ENABLE_ROCM
Status
CheckGPUCompatibility(const int gpu_id, const double min_compute_capability)
{
  // Query the compute capability from the device
  hipDeviceProp_t cuprops;
  hipError_t cuerr = hipGetDeviceProperties(&cuprops, gpu_id);
  if (cuerr != hipSuccess) {
    return Status(
        Status::Code::INTERNAL,
        "unable to get ROCM device properties for GPU ID" +
            std::to_string(gpu_id) + ": " + hipGetErrorString(cuerr));
  }

  double compute_compability = cuprops.major + (cuprops.minor / 10.0);
  if ((compute_compability > min_compute_capability) ||
      (abs(compute_compability - min_compute_capability) < 0.01)) {
    return Status::Success;
  } else {
    return Status(
        Status::Code::UNSUPPORTED,
        "gpu " + std::to_string(gpu_id) + " has compute capability '" +
            std::to_string(cuprops.major) + "." +
            std::to_string(cuprops.minor) +
            "' which is less than the minimum supported of '" +
            std::to_string(min_compute_capability) + "'");
  }
}

Status
GetSupportedGPUs(
    std::set<int>* supported_gpus, const double min_compute_capability)
{
  // Make sure set is empty before starting
  supported_gpus->clear();

  int device_cnt;
  hipError_t cuerr = hipGetDeviceCount(&device_cnt);
  if ((cuerr == hipErrorNoDevice) || (cuerr == hipErrorInsufficientDriver)) {
    device_cnt = 0;
  } else if (cuerr != hipSuccess) {
    return Status(
        Status::Code::INTERNAL, "unable to get number of ROCM devices: " +
                                    std::string(hipGetErrorString(cuerr)));
  }

  // populates supported_gpus
  for (int gpu_id = 0; gpu_id < device_cnt; gpu_id++) {
    Status status = CheckGPUCompatibility(gpu_id, min_compute_capability);
    if (status.IsOk()) {
      supported_gpus->insert(gpu_id);
    }
  }
  return Status::Success;
}

Status
SupportsIntegratedZeroCopy(const int gpu_id, bool* zero_copy_support)
{
  // Query the device to check if integrated
  hipDeviceProp_t cuprops;
  hipError_t cuerr = hipGetDeviceProperties(&cuprops, gpu_id);
  if (cuerr != hipSuccess) {
    return Status(
        Status::Code::INTERNAL,
        "unable to get ROCM device properties for GPU ID" +
            std::to_string(gpu_id) + ": " + hipGetErrorString(cuerr));
  }

  // Zero-copy supported only on integrated GPU when it can map host memory
  if (cuprops.integrated && cuprops.canMapHostMemory) {
    *zero_copy_support = true;
  } else {
    *zero_copy_support = false;
  }

  return Status::Success;
}

#endif

}}  // namespace triton::core
