// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "buffer_attributes.h"

#include <cstring>

#include "constants.h"

namespace triton { namespace core {
void
BufferAttributes::SetByteSize(const size_t& byte_size)
{
  byte_size_ = byte_size;
}

void
BufferAttributes::SetMemoryType(const TRITONSERVER_MemoryType& memory_type)
{
  memory_type_ = memory_type;
}

void
BufferAttributes::SetMemoryTypeId(const int64_t& memory_type_id)
{
  memory_type_id_ = memory_type_id;
}

void
BufferAttributes::SetCudaIpcHandle(void* rocm_ipc_handle)
{
  char* lrocm_ipc_handle = reinterpret_cast<char*>(rocm_ipc_handle);
  rocm_ipc_handle_.clear();
  std::copy(
      lrocm_ipc_handle, lrocm_ipc_handle + ROCM_IPC_STRUCT_SIZE,
      std::back_inserter(rocm_ipc_handle_));
}

void*
BufferAttributes::CudaIpcHandle()
{
  if (rocm_ipc_handle_.empty()) {
    return nullptr;
  } else {
    return reinterpret_cast<void*>(rocm_ipc_handle_.data());
  }
}

size_t
BufferAttributes::ByteSize() const
{
  return byte_size_;
}

TRITONSERVER_MemoryType
BufferAttributes::MemoryType() const
{
  return memory_type_;
}

int64_t
BufferAttributes::MemoryTypeId() const
{
  return memory_type_id_;
}

BufferAttributes::BufferAttributes(
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id, char* rocm_ipc_handle)
    : byte_size_(byte_size), memory_type_(memory_type),
      memory_type_id_(memory_type_id)
{
  // rocm ipc handle size
  rocm_ipc_handle_.reserve(ROCM_IPC_STRUCT_SIZE);

  if (rocm_ipc_handle != nullptr) {
    std::copy(
        rocm_ipc_handle, rocm_ipc_handle + ROCM_IPC_STRUCT_SIZE,
        std::back_inserter(rocm_ipc_handle_));
  }
}
}}  // namespace triton::core
