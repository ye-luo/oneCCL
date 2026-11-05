/*
 Copyright 2016-2026 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/


#include <unistd.h>
#include <cstdint>
#include <utility>
#include "comm/comm.hpp"
#include "comm/windows.hpp"

std::vector<std::vector<void *>> do_simple_ipc_exchange(ccl_comm *comm,
                                                        const std::vector<void *> &ptrs,
                                                        const std::vector<size_t> &sizes,
                                                        std::vector<std::vector<size_t>> *out_sizes = nullptr);

int ccl_window::register_buf(void *ptr, size_t size) {
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    int comm_size = comm->size();
    int comm_rank = comm->rank();

    if (ptr == nullptr || size == 0) {
        LOG_WARN("invalid pointer ", ptr, " / size ", size, " for window registration");
        return 0;
    }
    {
        ze_context_handle_t context{};
        ze_device_handle_t device{};
        ze_memory_allocation_properties_t mem_alloc_props{};
        if (!ccl::ze::get_buffer_context_and_device(ptr, &context, &device, &mem_alloc_props)) {
            LOG_WARN("unable to get context from ptr\n");
            return 0;
        }
        void *base_ptr;
        size_t total_size;
        ZE_CALL(zeMemGetAddressRange, (context, ptr, &base_ptr, &total_size));
        if ((size_t)((char *)ptr + size) > (size_t)((char *)base_ptr + total_size)) {
            LOG_WARN("Invalid pointer registered. ");
            return 0;
        }
    }

    std::vector<void *> in_buffers(1);
    std::vector<size_t> in_buffers_sizes(1);
    in_buffers[0] = ptr;
    in_buffers_sizes[0] = size;
    LOG_DEBUG("calling do_simple_ipc_exchange on:", std::hex, ptr);
    std::vector<std::vector<size_t>> ipc_sizes;
    std::vector<std::vector<void *>> ipc_ptrs =
        do_simple_ipc_exchange(comm, in_buffers, in_buffers_sizes, &ipc_sizes);

    registered_ptrs.resize(comm_size);
    for (int r = 0; r < comm_size; r++) {
        registered_ptrs[r].ptr = r == comm_rank ? ptr : ipc_ptrs[r][0];
        registered_ptrs[r].size = (!ipc_sizes.empty() && !ipc_sizes[r].empty()) ? ipc_sizes[r][0] : size;
        LOG_DEBUG("register ipc ptr: rank: ", r, " ptr: ", std::hex, registered_ptrs[r].ptr, " size: ", registered_ptrs[r].size);
    }
    return 1;
#else
    return 0;
#endif
}

int ccl_window::deregister_buf() {
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    int comm_rank = comm->rank();
    int comm_size = comm->size();

    ze_context_handle_t context{};
    ze_device_handle_t device{};
    ze_memory_allocation_properties_t mem_alloc_props{};
    if (!ccl::ze::get_buffer_context_and_device(
            registered_ptrs[comm_rank].ptr, &context, &device, &mem_alloc_props)) {
        LOG_WARN("unable to get context from ptr\n");
        return 0;
    }

    // This function is invoked from noexcept contexts (move-assignment and
    // window deregistration). ZE_CALL throws a ccl::exception on failure,
    // which would call std::terminate() if it escaped. Swallow it here: the return
    // value is a status code that all callers already ignore.
    try {
        for (int r = 0; r < comm_size; r++) {
            if (r == comm->rank() || registered_ptrs[r].ptr == nullptr)
                continue;
            if (context) {
                ZE_CALL(zeMemCloseIpcHandle, (context, registered_ptrs[r].ptr));
            }
        }
    }
    catch (const ccl::exception& e) {
        LOG_WARN("failed to deregister window buffer: ", e.what());
        return 0;
    }
    return 1;
#else
    return 0;
#endif
}

bool ccl_window::is_registered(const void *ptr, size_t size, size_t &offset) {
    int comm_rank = comm->rank();
    if (ptr >= registered_ptrs[comm_rank].ptr &&
        static_cast<const char *>(ptr) + size <=
            static_cast<char *>(registered_ptrs[comm_rank].ptr) + registered_ptrs[comm_rank].size) {
        offset =
            static_cast<const char *>(ptr) - static_cast<char *>(registered_ptrs[comm_rank].ptr);
        LOG_DEBUG("found registration ptr: ",
                  std::hex,
                  registered_ptrs[comm_rank].ptr,
                  " offset: ",
                  offset);
        return true;
    }
    return false;
}

std::vector<void *> ccl_window::get_ptrs(size_t offset) {
    std::vector<void *> ptrs(comm->size());
    for (int r = 0; r < comm->size(); r++) {
        ptrs[r] = registered_ptrs[r].ptr ? (static_cast<char *>(registered_ptrs[r].ptr) + offset) : nullptr;
        LOG_DEBUG("get ptr: rank: ",
                  r,
                  " ptr: ",
                  std::hex,
                  registered_ptrs[r].ptr,
                  " with offset: ",
                  offset);
    }
    return ptrs;
}

std::vector<size_t> ccl_window::get_sizes() const {
    std::vector<size_t> sizes(comm->size());
    for (int r = 0; r < comm->size(); r++) {
        sizes[r] = registered_ptrs[r].size;
    }
    return sizes;
}

size_t ccl_window::get_size(int rank) const {
    if (rank >= 0 && rank < static_cast<int>(registered_ptrs.size())) {
        return registered_ptrs[rank].size;
    }
    return 0;
}
