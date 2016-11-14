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

class iallocator {
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

		virtual void* get_raw (
			size_t alignment,
			size_t num_bytes);

	protected:
		virtual void* get_raw (size_t alignment,
			size_t num_bytes,
			const alloc_attrib& attrib) = 0;

		virtual void del_raw (void* ptr) = 0;

		virtual iallocator* clone_impl (void) = 0;

	public:
		static constexpr size_t alloc_alignment = 32;

		virtual ~iallocator (void) {}

		iallocator* clone (void) {
			return clone_impl();
		}

		virtual size_t id (void) = 0;

		template <typename T>
		T* allocate (size_t num_elements) {
			alloc_attrib attr;
			return allocate<T>(num_elements, attr);
		}

		template <typename T>
		T* allocate (size_t num_elements, const alloc_attrib& attrib) {
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
		virtual size_t requested_size (void* ptr);
		// alloc id if tracking enabled
		// 0 otherwise
		virtual size_t alloc_id (void* ptr) { return 0; }

		// virtual size_t requested_size_slowly (void* ptr) {
		//	 if (TracksAllocationSizes()) {
		//		 return AllocatedSize(ptr);
		//	 }
		//	 return 0;
		// }

		// // Fills in stat gatherer with statistics collected by this allocator.
		// virtual void gather_stat (alloc_stat* stats) { stats->clear(); }

};

}

#endif /* allocator_hpp */
