//
//  operation.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <unordered_map>
#include <algorithm>
#include <random>
#include <functional>
#include <limits>
#include <memory>

#pragma once
#ifndef operation_hpp
#define operation_hpp

#include "../variable/variable.hpp"
#include "../observer/node.hpp"

namespace nnet {

template <typename T>
using BUILD_DERIVE = std::function<ivariable<T>*(std::vector<ivariable<T>*>)>;

// INTERFACE OPERATION

template <typename T>
class iunar_ops;
template <typename T>
class ibin_ops;
template <typename T>
class univar_func;

template <typename T>
class ioperation : public ivariable<T>, public ccoms::inode {
	protected:
		ivariable<T>* grad = nullptr;

		// DEPRECATED
		// used in calc_gradient to toggle operations between returning eval and returning one
		// bool derive_this = false;
		
		// virtual void set_gradient (ivariable<T>* g) {
		// 	if (nullptr == grad && nullptr != g) {
		// 		grad = g;
		// 		ivariable<T>::set_gradient(grad);
		// 	}
		// }

		// TODO: find a way to move shape evaluation functions out of here
		// utility shape operations
		tensor_shape get_element_shape (
			const tensor<T>& t1,
			const tensor<T>& t2) const;

		tensor_shape transpose_shape (const tensor_shape& ins) const;

		tensor_shape change_shape (
			const tensor_shape& ins,
			size_t index,
			double multiplier,
			size_t& below_dim,
			size_t& at_idx) const;

		tensor_shape get_matrix_shape (
			const tensor<T>& t1, const tensor<T>& t2,
			bool transposeA, bool transposeB,
			size_t& common_dim) const;

		// operator wrapper functions that handle variable scalar and shape
		// differences. returned tensor's ownership transfers to caller
		// reduce/filter functional
		template <typename U>
		void util_op (
			U& out, const tensor<T>& in,
			std::function<void(U&, T)> op) const;
		// collector functions while maintaining tensor shape
		void elem_op (
			tensor<T>& out,
			const tensor<T>& in,
			std::function<void(T&, T)> op) const;

		// TODO get rid of these in favor of collectors (no need to allocate new tensors)
		tensor<T>* get_trace (const ivariable<T>& in) const;
		// 1 to 1 map functional
		tensor<T>* util_op (
			const tensor<T>& in,
			std::function<T(T)> op) const;
		// 2 to 1 map functional
		tensor<T>* util_op (
			const tensor<T>& a,
			const tensor<T>& b,
			std::function<T(T, T)> op) const;
		// operator wrapper functions that restricts shapes to matrices and
		// retrieve raw value
		tensor<T>* transpose_op (
		    tensor<T> const & in) const;
		tensor<T>* matmul_op (
			const tensor<T>& a,
			const tensor<T>& b,
		    bool transposeA,
		    bool transposeB) const;
		tensor<T>* extend_op (const tensor<T>& in, size_t index, size_t multiplier) const;
		tensor<T>* compress_op (
			const tensor<T>& in,
			signed index,
			std::function<T(const std::vector<T>&)> collector) const;

		// consume control
		// clears input
//		void deconsume (ivariable<T>& food) { this->remove_consumer(food, *this); }
//		// note keeping: record self as consumer of food
//		void consume (ivariable<T>& food) { this->add_consumer(food, *this); }

		// check if candidate_shape is worth propagating to consumers
		// before assigning consumers with new shapes to consume
		void update (tensor_shape candidate_shape);
		// implement unique method of consuming input variables
		// to extract shape info
		virtual void shape_eval (void) = 0;
		
		// CONSTRUCTS THE GRADIENT TREE AND STORE ROOT IN MEMBER GRAD
		virtual void setup_gradient (void) = 0;

		friend class ivariable<T>;
		friend class placeholder<T>;
		friend class univar_func<T>;

	public:
		ioperation (std::vector<ccoms::subject*> dependencies, std::string name) :
				ivariable<T>(std::vector<size_t>{}, name) {
			this->short_circuit = false;
		}
		virtual ~ioperation (void) {
			if (grad) {
				delete grad;
			}
		}
		
		virtual const tensor<T>& get_eval (void) {
			if (this->short_circuit) {
				return this->ones;
			}
			return this->out_;
		}

		virtual ivariable<T>* get_gradient (void) {
			if (nullptr == grad) {
				setup_gradient();
			}
			return grad;
		}
};

}

#include "../../../src/graph/operation/ioperation.ipp"

#endif /* operation_hpp */
