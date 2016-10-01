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

#include "variable.hpp"

#pragma once
#ifndef operation_hpp
#define operation_hpp

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
		// operator wrapper functions that handle variable scalar and shape
		// differences. returned tensor's ownership transfers to caller
		// reduce/filter functional
		tensor_shape get_element_shape (
			const tensor<T>& t1,
			const tensor<T>& t2) const;

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
			tensor<T> const & a,
			tensor<T> const & b,
		    bool transposeA,
		    bool transposeB) const;

		std::vector<T> get_vec (const tensor<T>& in) const {
			T* raw = in.raw_data;
			return std::vector<T>(raw, raw + in.n_elems());
		}
		// share friend priviledge with ivariable and tensor to descendants
		// retrieve the last evaluated tensor
		tensor<T>& get_eval (ivariable<T>& var) const { return var.out; }
		// note keeping: record self as consumer of food
		void consume (ivariable<T>& food) {
			food.consumers.push_back(this);
		}
		// check if candidate_shape is worth propagating to consumers
		// before assigning consumers with new shapes to consume
		void update (tensor_shape candidate_shape);
		// implement unique method of consuming input variables
		// to extract shape info
		virtual void shape_eval (void) = 0;

	public:
		// clone remains abstract
		virtual ~ioperation (void) {}
		// eval remains abstract
};

// FUNCTION WRAPPER

template <typename T>
class univar_func : ioperation<T> {
	private:
		// do not own fanin or out
		ivariable<T>* fanin = nullptr;
		ioperation<T>* fanout = nullptr;
		// no longer need if use sharedptr

		void clear (void);
		void copy (ivariable<T> const & other, std::string name = "");

	protected:
		std::vector<ioperation<T>*> ownout;
		virtual void shape_eval (void);

		univar_func (const univar_func<T>& other, std::string name);

	public:
		// declare
		univar_func (std::function<void(ioperation<T>*&)> declare);
		// currently shallow copy
		// TODO implement graph object manager for deep copy cloner
		virtual univar_func<T>* clone (std::string name = "");
		virtual ~univar_func (void) { clear(); }
		// connect input to fanin ivariables according
		// to declared equation ordered by function parameters
		virtual ivariable<T>& operator () (ivariable<T>* input);
		virtual univar_func<T>& operator = (ivariable<T> const & other);

		// calls derive from fanout
		virtual tensor<T>* derive (ivariable<T>* over) const;
		// calls derive on single input
		tensor<T>* derive (void) const;
		virtual const tensor<T>& eval (void);
};

// ACTIVATION FUNCTIONS

template <typename T>
class sigmoid : public univar_func<T> {
	public:
		sigmoid (void);
};

template <typename T>
class tanh : public univar_func<T> {
	public:
		tanh (void);
};

// SCALAR using operation functions
// TODO change base class to ivariable
template <typename T>
class scalar : public ioperation<T> {
	private:
		static T equivalent (T a, T b);
		static void atleast1 (bool& reduce, T value);

	protected:
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;

		scalar (scalar<T> const & other, std::string name);
		virtual void shape_eval (void) { /* do nothing */ }

	public:
		scalar (T value);
		virtual scalar<T>* clone (std::string name = "");
		virtual ~scalar (void) {}
		virtual const tensor<T>& eval (void) { return this->out; }
};

// UNARY OPERATIONS

template <typename T>
class unar_ops : public ioperation<T> {
	protected:
		// avoid calling ivariable's assignment multiple time
		ivariable<T>* var = nullptr;

		// backward chaining for AD
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;
		void copy (const ivariable<T>& other, std::string name = "");
		virtual void shape_eval (void);

		// operator () getters
		virtual std::string get_symb (void) = 0;
		virtual std::function<T(T)> get_op (void) = 0;

		friend class univar_func<T>;

	public:
		virtual ~unar_ops (void) {}
		virtual ivariable<T>& operator () (ivariable<T>& in);
		virtual unar_ops<T>& operator = (ivariable<T> const & other);

		virtual const tensor<T>& eval (void);
};

// BINARY OPERATIONS

template <typename T>
class bin_ops : public ioperation<T> {
	protected:
		ivariable<T>* a = nullptr;
		ivariable<T>* b = nullptr;
		ivariable<T>* own = nullptr;

		// calc_derive remains abstract
		void copy (const ivariable<T>& other, std::string name = "");
		virtual void shape_eval (void);

		// operator () getters
		virtual std::string get_symb (void) = 0;
		virtual std::function<T(T, T)> get_op (void) = 0;

		friend class univar_func<T>;

	public:
		virtual ~bin_ops (void) { if (own) delete own; }
		virtual ivariable<T>& operator () (ivariable<T>& a, ivariable<T>& b);
		virtual ivariable<T>& operator () (ivariable<T>& a, T b);
		virtual ivariable<T>& operator () (T a, ivariable<T>& b);
		virtual bin_ops<T>& operator = (ivariable<T> const & other);

		virtual const tensor<T>& eval (void);
};

// MATRIX OPERATIONS
// restricted to 2-d matrices with proper shapes
// dimension 1 denote number of columns,
// dimension 2 denote number of rows
// TODO implement calc_deriv for transpose and matmul ops
// MATRIX TRANSPOSE

template <typename T>
class transpose : public ioperation<T> {
	private:
		ivariable<T>* var = nullptr;

	protected:
		// backward chaining for AD
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;
		virtual void shape_eval (void);

		transpose (transpose<T> const & other, std::string name);

	public:
		transpose (void) {}
		transpose (ivariable<T>& in);
		virtual transpose<T>* clone (std::string name = "");
		virtual ivariable<T>& operator () (ivariable<T>& in);
		virtual ~transpose (void) {}
		virtual transpose<T>& operator = (ivariable<T> const & other);

		virtual const tensor<T>& eval (void);
};

// MATRIX MULTIPLICATION

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
		virtual void shape_eval (void);

		matmul (matmul<T> const & other, std::string name);

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
		virtual ~matmul (void) {}
		virtual matmul<T>& operator = (ivariable<T> const & other);

		virtual const tensor<T>& eval (void);
};

// OUT NODE

template <typename T>
class expose : public unar_ops<T> {
	protected:
		// inherits calc_derive from unar_ops
		expose (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "expose"; }
		std::function<T(T)> get_op (void);

	public:
		expose (void) {}
		expose (ivariable<T>& var) { (*this)(var); }
		virtual expose<T>* clone (std::string name = "");

		virtual const tensor<T>& eval (void) { return this->var->eval(); }

		// non-inheriteds
		// evaluates consumed operation
		virtual std::vector<T> get_raw (void);
		// extracts derivative based on the LAST evaluation
		// doesn't evaluate
		virtual std::vector<T> get_derive (ivariable<T>& over) const;
};

// NEGATION

template <typename T>
class neg : public unar_ops<T> {
	protected:
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;
		neg (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "-"; }
		std::function<T(T)> get_op (void);

	public:
		neg (void) {}
		neg (ivariable<T>& var) { (*this)(var); }
		virtual neg<T>* clone (std::string name = "");
};

// SINE

template <typename T>
class sin : public unar_ops<T> {
	protected:
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;
		sin (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "sin"; }
		std::function<T(T)> get_op (void);

	public:
		sin (void) {}
		sin (ivariable<T>& var) { (*this)(var); }
		virtual sin<T>* clone (std::string name = "");
};

// COSINE

template <typename T>
class cos : public unar_ops<T> {
	protected:
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;
		cos (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "cos"; }
		std::function<T(T)> get_op (void);

	public:
		cos (void) {}
		cos (ivariable<T>& var) { (*this)(var); }
		virtual cos<T>* clone (std::string name = "");
};

// TANGENT

template <typename T>
class tan : public unar_ops<T> {
	protected:
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;
		tan (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "tan"; }
		std::function<T(T)> get_op (void);

	public:
		tan (void) {}
		tan (ivariable<T>& var) { (*this)(var); }
		virtual tan<T>* clone (std::string name = "");
};

// COSECANT

template <typename T>
class csc : public unar_ops<T> {
	protected:
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;
		csc (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "csc"; }
		std::function<T(T)> get_op (void);

	public:
		csc (void) {}
		csc (ivariable<T>& var) { (*this)(var); }
		virtual csc<T>* clone (std::string name = "");
};

// SECANT

template <typename T>
class sec : public unar_ops<T> {
	protected:
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;
		sec (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "sec"; }
		std::function<T(T)> get_op (void);

	public:
		sec (void) {}
		sec (ivariable<T>& var) { (*this)(var); }
		virtual sec<T>* clone (std::string name = "");
};

// COTANGENT

template <typename T>
class cot : public unar_ops<T> {
	protected:
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;
		cot (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "cot"; }
		std::function<T(T)> get_op (void);

	public:
		cot (void) {}
		cot (ivariable<T>& var) { (*this)(var); }
		virtual cot<T>* clone (std::string name = "");
};

// EXPONENT OF E

template <typename T>
class exp : public unar_ops<T> {
	protected:
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;
		exp (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "exp"; }
		std::function<T(T)> get_op (void);

	public:
		exp (void) {}
		exp (ivariable<T>& var) { (*this)(var); }
		virtual exp<T>* clone (std::string name = "");
};

// ADDITION

template <typename T>
class add : public bin_ops<T> {
	protected:
		// backward chaining for AD
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;
		add (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "+"; }
		std::function<T(T, T)> get_op (void);

	public:
		add (void) {}
		add (ivariable<T>& a, ivariable<T>& b) { (*this)(a, b); }
		add (ivariable<T>& a, T b) { (*this)(a, b); }
		add (T a, ivariable<T>& b) { (*this)(a, b); }
		virtual add<T>* clone (std::string name = "");
};

// SUBTRACTION

template <typename T>
class sub : public bin_ops<T> {
	protected:
		// backward chaining for AD
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;
		sub (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "-"; }
		std::function<T(T, T)> get_op (void);

	public:
		sub (void) {}
		sub (ivariable<T>& a, ivariable<T>& b) { (*this)(a, b); }
		sub (ivariable<T>& a, const T b) { (*this)(a, b); }
		sub (const T a, ivariable<T>& b) { (*this)(a, b); }
		virtual sub<T>* clone (std::string name = "");
};

// element wise multiplication
template <typename T>
class mul : public bin_ops<T> {
	protected:
		// backward chaining for AD
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;
		mul (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "*"; }
		std::function<T(T, T)> get_op (void);

	public:
		mul (void) {}
		mul (ivariable<T>& a, ivariable<T>& b) { (*this)(a, b); }
		mul (ivariable<T>& a, const T b) { (*this)(a, b); }
		mul (const T a, ivariable<T>& b) { (*this)(a, b); }
		virtual mul<T>* clone (std::string name = "");
};

// element wise division
template <typename T>
class div : public bin_ops<T> {
	protected:
		// backward chaining for AD
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;
		div (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "/"; }
		std::function<T(T, T)> get_op (void);

	public:
		div (void) {}
		div (ivariable<T>& a, ivariable<T>& b) { (*this)(a, b); }
		div (ivariable<T>& a, const T b) { (*this)(a, b); }
		div (const T a, ivariable<T>& b) { (*this)(a, b); }
		virtual div<T>* clone (std::string name = "");
};

template <typename T>
tensor<T> assign (variable<T> value, variable<T> delta, bool use_locking = false) {
	return tensor<T>();
}

template <typename T>
tensor<T> assign_add (variable<T> value, variable<T> delta, bool use_locking = false) {
	return tensor<T>();
}

template <typename T>
tensor<T> assign_sub (variable<T> value, variable<T> delta, bool use_locking = false) {
	return tensor<T>();
}

// tensor scatter_sub (IndexedSlices sparse_delta, use_locking = false) {}

template <typename T>
void count_up_to (variable<T> value, size_t limit) {

}

// template <typename T>
// tensor<T>* product (tensor<T> const & t1, tensor<T> const & t2);
//
// template <typename T>
// void contract (tensor<T> const & t);
//
// template <typename T>
// void raise (tensor<T> const & t, size_t idx);
//
// template <typename T>
// void lower (tensor<T> const & t, size_t idx);

}

#include "../../src/graph/operation.tpp"

#endif /* operation_hpp */
