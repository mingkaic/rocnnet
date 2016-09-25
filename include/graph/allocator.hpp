//
//  allocator.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef allocator_hpp
#define allocator_hpp

#include <algorithm>
#include <complex>

namespace nnet {

struct alloc_attrib {

};

class allocator {
    private:
        // allow floats and doubles: is_trivial
        // allow complex
        // add other allowed types here
        template <typename T>
        struct is_allowed {
            static constexpr bool value =
            std::is_trivial<T>::value ||
            std::is_same<T, std::complex<size_t> >::value ||
            std::is_same<T, std::complex<double> >::value;
        };

        virtual void* get_raw (size_t alignment,
            size_t num_bytes) {
            return get_raw(alignment, num_bytes, alloc_attrib());
        }

    protected:
        virtual void* get_raw (size_t alignment,
            size_t num_bytes,
            alloc_attrib const & attrib) = 0;

        virtual void del_raw (void* ptr) = 0;

    public:
        static constexpr size_t alloc_alignment = 32;

        virtual ~allocator (void) {}

        virtual allocator* copy (void) = 0;

        virtual size_t id (void) = 0; // update to boost uuid

        template <typename T>
        T* allocate (size_t num_elements) {
            alloc_attrib attr;
            return allocate<T>(num_elements, attr);
        }

        template <typename T>
        T* allocate (size_t num_elements, alloc_attrib const & attrib) {
            static_assert(is_allowed<T>::value, "T is not an allowed type.");

            if (num_elements > (std::numeric_limits<size_t>::max() / sizeof(T))) {
                return nullptr;
            }

            void* p = get_raw(alloc_alignment,
                sizeof(T) * num_elements,
                attrib);
            T* typedptr = reinterpret_cast<T*>(p);
            return typedptr;
        }

        template <typename T>
        void dealloc(T* ptr, size_t num_elements) {
            if (nullptr != ptr) {
                del_raw(ptr);
            }
        }

        // does the implementation of allocator track the allocated size
        virtual bool tracks_size (void) { return false; }

        // requires empty tensors to allocate
        virtual bool alloc_empty (void) { return false; }

        // gets allocated size if tracking enabled
        virtual size_t requested_size (void* ptr) {
            throw std::bad_function_call();
            return 0;
        }

        // alloc id if tracking enabled
        // 0 otherwise
        virtual size_t alloc_id (void* ptr) { return 0; }

        // virtual size_t requested_size_slowly (void* ptr) {
        //     if (TracksAllocationSizes()) {
        //         return AllocatedSize(ptr);
        //     }
        //     return 0;
        // }

        // // Fills in stat gatherer with statistics collected by this allocator.
        // virtual void gather_stat (alloc_stat* stats) { stats->clear(); }

};

class memory_alloc : public allocator {
    protected:
        virtual void* get_raw (size_t alignment,
            size_t num_bytes,
            alloc_attrib const & attrib) {
            // str8 2 memory, ignore attributes
            void* ptr = malloc(num_bytes);
            return ptr;
        }

        virtual void del_raw (void* ptr) {
            // str8 from memory
            free (ptr);
        }

    public:
        virtual allocator* copy (void) {
            return new memory_alloc();
        }

        virtual size_t id (void) { return 0; } // update to boost uuid
};

}

#endif /* allocator_hpp */
