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

// DEPRECATED (todo: replace with ioperation)
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
		template <typename U>
		void util_op (
			U& out,
			tensor<T> const & in,
			std::function<void(U&, T)> op) const;
		// 1 to 1 map functional
		tensor<T>* util_op (
			tensor<T> const & in,
			std::function<T(T)> op) const;
		// 2 to 1 map functional
		tensor<T>* util_op (
			tensor<T> const & a,
			tensor<T> const & b,
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
		void copy (univar_func<T> const & other, std::string name = "");
		univar_func (univar_func<T> const & other, std::string name);

	protected:
		std::vector<ioperation<T>*> ownout;

	public:
		// declare
		univar_func (std::function<void(ioperation<T>*)> declare);
		// shallow copy
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
class sigmoid : univar_func<T> {
	public:
		sigmoid (void);
};

template <typename T>
class tanh : univar_func<T> {
	public:
		tanh (void);
};

// SCALAR using operation functions

template <typename T>
class scalar : public ioperation<T> {
	private:
		static T equivalent (T a, T b);
		static void atleast1 (bool& reduce, T value);

	protected:
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;

		scalar (scalar<T> const & other, std::string name);

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
		std::function<T(T)> op;

		virtual void init (std::string op, ivariable<T>& var);
		// backward chaining for AD
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;

		unar_ops (void) { /* default construction propagates to ioperation */ }
		unar_ops (unar_ops<T> const & other, std::string name);

		friend class univar_func<T>;

	public:
		virtual ~unar_ops (void) {}
		virtual unar_ops<T>* clone (std::string name = "");
		virtual ivariable<T>& operator () (ivariable<T>& in) { return *this; }
		virtual unar_ops<T>& operator = (ivariable<T> const & other);

		virtual const tensor<T>& eval (void);
};

// BINARY OPERATIONS

template <typename T>
class bin_ops : public ioperation<T> {
	protected:
		ivariable<T>* a = nullptr;
		ivariable<T>* b = nullptr;
		std::function<T(T, T)> op;

		virtual void init (std::string op, ivariable<T>& a, ivariable<T>& b);
		virtual void init (std::string op, ivariable<T>& a, T b);
		virtual void init (std::string op, T a, ivariable<T>& b);
		// calc_derive remains abstract

		bin_ops (void) { /* default construction propagates to ioperation */ }
		bin_ops (bin_ops<T> const & other, std::string name);

		friend class univar_func<T>;

	public:
		virtual bin_ops<T>* clone (std::string name = "");
		virtual ~bin_ops (void) {}
		virtual ivariable<T>& operator () (ivariable<T>& a, ivariable<T>& b) { return *this; }
		virtual ivariable<T>& operator () (ivariable<T>& a, T b) { return *this; }
		virtual ivariable<T>& operator () (T a, ivariable<T>& b) { return *this; }
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

	public:
		expose (void) {}
		expose (ivariable<T>& var) { (*this)(var); }
		ivariable<T>& operator () (ivariable<T>& in);

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

	public:
		neg (void) {}
		neg (ivariable<T>& var) { (*this)(var); }
		ivariable<T>& operator () (ivariable<T>& in);
};

// SINE

template <typename T>
class sin : public unar_ops<T> {
	protected:
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;

	public:
		sin (void) {}
		virtual sin<T>* clone (std::string name = "") {
			return dynamic_cast<sin<T>*>(unar_ops<T>::clone(name));
		}
		sin (ivariable<T>& var) { (*this)(var); }
		ivariable<T>& operator () (ivariable<T>& in);
};

// COSINE

template <typename T>
class cos : public unar_ops<T> {
	protected:
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;

	public:
		cos (void) {}
		virtual cos<T>* clone (std::string name = "") {
			return dynamic_cast<cos<T>*>(unar_ops<T>::clone(name));
		}
		cos (ivariable<T>& var) { (*this)(var); }
		ivariable<T>& operator () (ivariable<T>& in);
};

// TANGENT

template <typename T>
class tan : public unar_ops<T> {
	protected:
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;

	public:
		tan (void) {}
		virtual tan<T>* clone (std::string name = "") {
			return dynamic_cast<tan<T>*>(unar_ops<T>::clone(name));
		}
		tan (ivariable<T>& var) { (*this)(var); }
		ivariable<T>& operator () (ivariable<T>& in);
};

// COSECANT

template <typename T>
class csc : public unar_ops<T> {
	protected:
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;

	public:
		csc (void) {}
		virtual csc<T>* clone (std::string name = "") {
			return dynamic_cast<csc<T>*>(unar_ops<T>::clone(name));
		}
		csc (ivariable<T>& var) { (*this)(var); }
		ivariable<T>& operator () (ivariable<T>& in);
};

// SECANT

template <typename T>
class sec : public unar_ops<T> {
	protected:
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;

	public:
		sec (void) {}
		virtual sec<T>* clone (std::string name = "") {
			return dynamic_cast<sec<T>*>(unar_ops<T>::clone(name));
		}
		sec (ivariable<T>& var) { (*this)(var); }
		ivariable<T>& operator () (ivariable<T>& in);
};

// COTANGENT

template <typename T>
class cot : public unar_ops<T> {
	protected:
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;

	public:
		cot (void) {}
		virtual cot<T>* clone (std::string name = "") {
			return dynamic_cast<cot<T>*>(unar_ops<T>::clone(name));
		}
		cot (ivariable<T>& var) { (*this)(var); }
		ivariable<T>& operator () (ivariable<T>& in);
};

// EXPONENT OF E

template <typename T>
class exp : public unar_ops<T> {
	protected:
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;

	public:
		exp (void) {}
		virtual exp<T>* clone (std::string name = "") {
			return dynamic_cast<exp<T>*>(unar_ops<T>::clone(name));
		}
		exp (ivariable<T>& var) { (*this)(var); }
		ivariable<T>& operator () (ivariable<T>& in);
};

// ADDITION

template <typename T>
class add : public bin_ops<T> {
	protected:
		// backward chaining for AD
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;

	public:
		add (void) {}
		virtual add<T>* clone (std::string name = "") {
			return dynamic_cast<add<T>*>(bin_ops<T>::clone(name));
		}
		add (ivariable<T>& a, ivariable<T>& b) { (*this)(a, b); }
		add (ivariable<T>& a, T b) { (*this)(a, b); }
		add (T a, ivariable<T>& b) { (*this)(a, b); }

		ivariable<T>& operator () (ivariable<T>& a, ivariable<T>& b);
		ivariable<T>& operator () (ivariable<T>& a, T b);
		ivariable<T>& operator () (T a, ivariable<T>& b);
};

// SUBTRACTION

template <typename T>
class sub : public bin_ops<T> {
	protected:
		// backward chaining for AD
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;

	public:
		sub (void) {}
		virtual sub<T>* clone (std::string name = "") {
			return dynamic_cast<sub<T>*>(bin_ops<T>::clone(name));
		}
		sub (ivariable<T>& a, ivariable<T>& b) { (*this)(a, b); }
		sub (ivariable<T>& a, const T b) { (*this)(a, b); }
		sub (const T a, ivariable<T>& b) { (*this)(a, b); }

		ivariable<T>& operator () (ivariable<T>& a, ivariable<T>& b);
		ivariable<T>& operator () (ivariable<T>& a, T b);
		ivariable<T>& operator () (T a, ivariable<T>& b);
};

// element wise multiplication
template <typename T>
class mul : public bin_ops<T> {
	protected:
		// backward chaining for AD
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;

	public:
		mul (void) {}
		virtual mul<T>* clone (std::string name = "") {
			return dynamic_cast<mul<T>*>(bin_ops<T>::clone(name));
		}
		mul (ivariable<T>& a, ivariable<T>& b) { (*this)(a, b); }
		mul (ivariable<T>& a, const T b) { (*this)(a, b); }
		mul (const T a, ivariable<T>& b) { (*this)(a, b); }

		ivariable<T>& operator () (ivariable<T>& a, ivariable<T>& b);
		ivariable<T>& operator () (ivariable<T>& a, T b);
		ivariable<T>& operator () (T a, ivariable<T>& b);
};

// element wise division
template <typename T>
class div : public bin_ops<T> {
	protected:
		// backward chaining for AD
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;

	public:
		div (void) {}
		virtual div<T>* clone (std::string name = "") {
			return dynamic_cast<div<T>*>(bin_ops<T>::clone(name));
		}
		div (ivariable<T>& a, ivariable<T>& b) { (*this)(a, b); }
		div (ivariable<T>& a, const T b) { (*this)(a, b); }
		div (const T a, ivariable<T>& b) { (*this)(a, b); }

		ivariable<T>& operator () (ivariable<T>& a, ivariable<T>& b);
		ivariable<T>& operator () (ivariable<T>& a, T b);
		ivariable<T>& operator () (T a, ivariable<T>& b);
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
