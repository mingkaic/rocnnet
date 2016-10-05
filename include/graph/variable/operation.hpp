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

#pragma once
#ifndef operation_hpp
#define operation_hpp

#include "variable.hpp"
#include "shared_ops.hpp"

namespace nnet {

// DEPRECATED (TODO: replace with ioperation)
struct adhoc_operation {
	std::function<double(double)> act;
	std::function<double(double)> grad;

	adhoc_operation (void) {}

	adhoc_operation (std::function<double(double)> act,
		std::function<double(double)> grad)
		: act(act), grad(grad) {}

	double operator () (double in) { return act(in); }
};

// INTERFACE OPERATION

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
			size_t index,
			std::function<T(const std::vector<T>&)> collector) const;

		std::vector<T> get_vec (const tensor<T>& in) const {
			T* raw = in.raw_data;
			return std::vector<T>(raw, raw + in.n_elems());
		}
		// share friend priviledge with ivariable and tensor to descendants
		// retrieve the last evaluated tensor
		tensor<T>& get_eval (ivariable<T>& var) const { return var.out; }
		// note keeping: record self as consumer of food
		void consume (ivariable<T>& food) {
			food.consumers.insert(this);
		}
		virtual void decompose (ivariable<T>& food) = 0;

		// check if candidate_shape is worth propagating to consumers
		// before assigning consumers with new shapes to consume
		void update (tensor_shape candidate_shape);
		// implement unique method of consuming input variables
		// to extract shape info
		virtual void shape_eval (void) = 0;

		friend class ivariable<T>;

	public:
		// clone remains abstract
		virtual ~ioperation (void) {}
		// eval remains abstract
};

template <typename T>
class iunar_ops;
template <typename T>
class ibin_ops;
template <typename T>
class univar_func;

// SCALAR using operation functions
// TODO change base class to ivariable
template <typename T>
class scalar : public ioperation<T> {
	private:
		static T equivalent (T a, T b);
		static void atleast1 (bool& reduce, T value);

	protected:
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;
		virtual void decompose (ivariable<T>& food) {}

		scalar (const scalar<T>& other, std::string name);
		virtual void shape_eval (void) { /* do nothing */ }

	public:
		scalar (T value);
		virtual scalar<T>* clone (std::string name = "");
		virtual ~scalar (void) {}
		virtual const tensor<T>& eval (void) { return this->out; }
};

// MATRIX OPERATIONS

// UNARY MATRIX OPERATIONS

template <typename T>
class iunar_mat_ops : public ioperation<T> {
	protected:
		ivariable<T>* var = nullptr;

		virtual void decompose (ivariable<T>& food) {
			if (var == &food) var = nullptr;
		}
		virtual std::string get_symb (void) = 0;

		virtual void shape_eval (void) = 0;
		void copy (const ivariable<T>& other, std::string name = ""); // virtually identical to iunar_ops

	public:
		virtual ~iunar_mat_ops (void) { if (var) var->get_consumers().erase(this); }
		virtual ivariable<T>& operator () (ivariable<T>& in); // identical to iunar_ops
		virtual iunar_mat_ops<T>& operator = (const ivariable<T>& other);

		// eval remains abstract
};

// TENSOR EXTENSION

template <typename T>
class extend : public iunar_mat_ops<T> {
	private:
		size_t index = 0;
		size_t multiplier = 1;

	protected:
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;
		virtual std::string get_symb (void) { return "extend"; }

		virtual void shape_eval (void);
		void copy (const ivariable<T>& other, std::string name = "");
		extend (const ivariable<T>& other, std::string name) { this->copy(other, name); }

	public:
		extend (void) {}
		extend (ivariable<T>& in);
		extend (ivariable<T>& in, size_t index, size_t multiplier);
		void set_ext_info (size_t index, size_t multiplier);
		virtual extend<T>* clone (std::string name = "");
		virtual extend<T>& operator = (const ivariable<T>& other);

		virtual const tensor<T>& eval (void);
};

// TENSOR COMPRESSION ALONG ONE DIMENSION

template <typename T>
class compress : public iunar_mat_ops<T> {
	private:
		size_t index = 0;
		// first parameter is the collecting buffer, second is the gathered data
		std::function<T(const std::vector<T>&)> collector; // default to average sum

	protected:
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;
		virtual std::string get_symb (void) { return "compress"; }

		virtual void shape_eval (void);
		void copy (const ivariable<T>& other, std::string name = "");
		compress (const ivariable<T>& other, std::string name) { this->copy(other, name); }

	public:
		compress (void) {}
		compress (ivariable<T>& in);
		compress (ivariable<T>& in, size_t index);
		compress (ivariable<T>& in, size_t index, std::function<T(const std::vector<T>&)> collector);
		virtual compress<T>* clone (std::string name = "");
		virtual compress<T>& operator = (const ivariable<T>& other);

		void set_cmpr_info (size_t index, std::function<T(const std::vector<T>&)> collector);

		virtual const tensor<T>& eval (void);
};

// MATRIX TRANSPOSE

template <typename T>
class transpose : public iunar_mat_ops<T> {
	protected:
		// backward chaining for AD
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;
		virtual std::string get_symb (void) { return "transpose"; }

		virtual void shape_eval (void);
		transpose (const ivariable<T>& other, std::string name) { this->copy(other, name); }

	public:
		transpose (void) {}
		transpose (ivariable<T>& in);
		virtual transpose<T>* clone (std::string name = "");

		virtual const tensor<T>& eval (void);
};

// MATRIX MULTIPLICATION

// restricted to 2-d matrices with proper shapes
// dimension 1 denote number of columns,
// dimension 2 denote number of rows
// TODO implement calc_deriv for matmul ops
template <typename T>
class matmul : public ioperation<T> {
	private:
		ivariable<T>* a = nullptr;
		ivariable<T>* b = nullptr;
		bool transposeA;
		bool transposeB;

	protected:
		// backward chaining for AD
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;
		virtual void decompose (ivariable<T>& food) {
			if (a == &food) a = nullptr;
			if (b == &food) b = nullptr;
		}

		virtual void shape_eval (void);
		matmul (const matmul<T>& other, std::string name);

	public:
		matmul (void) : transposeA(false), transposeB(false) {}
		matmul (
			ivariable<T>& a,
			ivariable<T>& b,
			bool transposeA = false,
			bool transposeB = false);
		virtual matmul<T>* clone (std::string name = "");
		virtual ivariable<T>& operator () (
			ivariable<T>& a,
			ivariable<T>& b,
			bool transposeA = false,
			bool transposeB = false);
		virtual ~matmul (void) {
			if (a) a->get_consumers().erase(this);
			if (b) b->get_consumers().erase(this);
		}
		virtual matmul<T>& operator = (const ivariable<T>& other);

		virtual const tensor<T>& eval (void);
};

// template <typename T>
// tensor<T> assign (variable<T> value, variable<T> delta, bool use_locking = false);
//
// template <typename T>
// tensor<T> assign_add (variable<T> value, variable<T> delta, bool use_locking = false);
//
// template <typename T>
// tensor<T> assign_sub (variable<T> value, variable<T> delta, bool use_locking = false);
//
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

#include "../../../src/graph/variable/operation.tpp"

#endif /* operation_hpp */
