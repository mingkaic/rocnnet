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
#include <queue>
#include <memory>

#pragma once
#ifndef operation_hpp
#define operation_hpp

#include "variable.hpp"
#include "shared_ops.hpp"

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
		// TODO: find a way to move these functions out of here
		// operator wrapper functions that handle variable scalar and shape
		// differences. returned tensor's ownership transfers to caller
		// reduce/filter functional
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

		std::vector<T> get_vec (const tensor<T>& in) const {
			T* raw = in.raw_data;
			size_t limit = in.n_elems();
			return std::vector<T>(raw, raw + limit);
		}
		// share friend priviledge with ivariable and tensor to descendants
		// retrieve the last evaluated tensor
		tensor<T>& get_eval (VAR_PTR<T> var) const {
			return var->out;
		}

		// clears input
		void deconsume (ivariable<T>& food) {
			food.consumers.erase(this);
		}

		// note keeping: record self as consumer of food
		void consume (ivariable<T>& food) {
			food.consumers.emplace(this);
		}

		// changes input
		virtual void replace (ivariable<T>* food, VAR_PTR<T> newfood) = 0;

		// check if candidate_shape is worth propagating to consumers
		// before assigning consumers with new shapes to consume
		void update (tensor_shape candidate_shape);
		// implement unique method of consuming input variables
		// to extract shape info
		virtual void shape_eval (void) = 0;
		virtual void make_gradient (void) = 0;

		friend class ivariable<T>;
		friend class placeholder<T>;
		friend class univar_func<T>;

	public:
		ioperation (void) {}
		// clone remains abstract
		virtual ~ioperation (void) {}
		// eval remains abstract
};

template <typename T>
using ELEMENTARY_DERIV = std::function<VAR_PTR<T>(std::vector<VAR_PTR<T> >)>;

template <typename T>
class elementary : public ioperation<T> {
	protected:
		std::vector<VAR_PTR<T> > args; // order matters
		std::function<void(T&, T)> op;
		ELEMENTARY_DERIV<T> der;

		virtual void make_gradient (void);
		virtual EVOKER_PTR<T> clone_impl (std::string name);
		virtual void replace (ivariable<T>* food, VAR_PTR<T> newfood);
		virtual void shape_eval (void);
		virtual const tensor<T>& calc_eval (VAR_PTR<T> active);

	public:
		elementary (VAR_PTR<T> arg,
					std::function<void(T&, T)> op,
					ELEMENTARY_DERIV<T> der,
					std::string name = "");
		elementary (VAR_PTR<T> arga, VAR_PTR<T> argb,
					std::function<void(T&, T)> op,
					ELEMENTARY_DERIV<T> der,
					std::string name = "");
		elementary (std::vector<VAR_PTR<T> > args,
					std::function<void(T&, T)> op,
					ELEMENTARY_DERIV<T> der,
					std::string name = "");
		std::shared_ptr<elementary<T> > clone (std::string name = "") {
			return std::static_pointer_cast<elementary<T>, ievoker<T> >(clone_impl(name));
		}

		virtual const tensor<T>& eval (void);
};

// template <typename T>
// tensor scatter_sub (IndexedSlices sparse_delta, use_locking = false);
//
// template <typename T>
// void count_up_to (variable<T> value, size_t limit);
//
// template <typename T>
// tensor<T>* product (const tensor<T>& t1, const tensor<T>& t2);
//
// template <typename T>
// void contract (const tensor<T>& t);
//
// template <typename T>
// void raise (const tensor<T>& t, size_t idx);
//
// template <typename T>
// void lower (const tensor<T>& t, size_t idx);

}

#include "../../src/variable/operation.tpp"

#endif /* operation_hpp */
