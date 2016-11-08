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

#include "graph/variable/variable.hpp"

namespace nnet {

// INTERFACE OPERATION

template <typename T>
class iunar_ops;
template <typename T>
class ibin_ops;
template <typename T>
class univar_func;

template <typename T>
class ioperation : public ivariable<T> {
	protected:
		// used in calc_gradient to toggle operations between returning eval and returning one
		bool derive_this = false;
		VAR_PTR<T> grad = nullptr;

		virtual void set_gradient (VAR_PTR<T> g) {
			if (nullptr == grad && nullptr != g) {
				grad = g;
				ivariable<T>::set_gradient(grad);
			}
		}

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
		void deconsume (ivariable<T>& food) { this->remove_consumer(food, *this); }
		// note keeping: record self as consumer of food
		void consume (ivariable<T>& food) { this->add_consumer(food, *this); }

		// changes input
		virtual void replace (ivariable<T>* food, VAR_PTR<T> newfood) = 0;

		// check if candidate_shape is worth propagating to consumers
		// before assigning consumers with new shapes to consume
		void update (tensor_shape candidate_shape);
		// implement unique method of consuming input variables
		// to extract shape info
		virtual void shape_eval (void) = 0;
		virtual void make_gradient (VAR_PTR<T>& safety_ref) = 0;

		friend class ivariable<T>;
		friend class placeholder<T>;
		friend class univar_func<T>;

	public:
		// clone remains abstract
		virtual ~ioperation (void) {}
		// eval remains abstract

		virtual VAR_PTR<T> get_gradient (void) {
			VAR_PTR<T> safety_ref;
			if (nullptr == this->grad) make_gradient(safety_ref);
			else safety_ref = this->grad;
			return safety_ref;
		}
};

}

#include "../../../src/graph/operation/ioperation.ipp"

#endif /* operation_hpp */
