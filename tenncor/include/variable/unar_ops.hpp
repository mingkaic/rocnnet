//
//  unar_ops.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-09-30.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef unar_ops_hpp
#define unar_ops_hpp
#include <iostream>

#include "operation.hpp"

namespace nnet {

// UNARY OPERATIONS

template <typename T>
class iunar_ops : public ioperation<T> {
	protected:
		// avoid calling ivariable's assignment multiple time
		ivariable<T>* var = nullptr;

		// backward chaining for AD
		virtual tensor<T>* calc_derive (ivariable<T>* over) const;
		void copy (const ivariable<T>& other, std::string name = "");
		virtual void decompose (ivariable<T>& food) {
			if (var == &food) var = nullptr;
		}

		virtual void shape_eval (void);
		// operator () getters
		virtual std::string get_symb (void) = 0;
		virtual std::function<T(T)> get_op (void) = 0;

		friend class univar_func<T>;

	public:
		virtual ~iunar_ops (void) {
			if (var) var->get_consumers().erase(this);
		}
		virtual ivariable<T>& operator () (ivariable<T>& in);
		virtual iunar_ops<T>& operator = (const ivariable<T>& other);

		virtual const tensor<T>& eval (void);
};

// OUT NODE

// TODO extend ioperation interface for unique operations like derive and expose
template <typename T>
class expose : public iunar_ops<T> {
	protected:
		// inherits calc_derive from iunar_ops
		expose (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "expose"; }
		std::function<T(T)> get_op (void);

	public:
		expose (void) {}
		expose (ivariable<T>& var) { (*this)(var); }
		virtual expose<T>* clone (std::string name = "");

		virtual const tensor<T>& eval (void) {
			this->out = this->var->eval();
			return this->out;
		}

		// non-inheriteds
		// evaluates consumed operation
		virtual std::vector<T> get_raw (void);
		// extracts derivative based on the LAST evaluation
		// doesn't evaluate
		virtual std::vector<T> get_derive (ivariable<T>& over) const;
};

// NEGATION

template <typename T>
class neg : public iunar_ops<T> {
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
class sin : public iunar_ops<T> {
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
class cos : public iunar_ops<T> {
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
class tan : public iunar_ops<T> {
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
class csc : public iunar_ops<T> {
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
class sec : public iunar_ops<T> {
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
class cot : public iunar_ops<T> {
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
class exp : public iunar_ops<T> {
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

}

#include "../../src/variable/unar_ops.tpp"

#endif /* unar_ops_hpp */
