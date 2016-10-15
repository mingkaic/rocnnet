//
//  bin_ops.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-09-30.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef bin_ops_hpp
#define bin_ops_hpp

#include "operation.hpp"

namespace nnet {

// forward declarations
template <typename T>
class ioperation;

template <typename T>
class univar_func;

// ELEMENT-WISE BINARY OPERATIONS

template <typename T>
class ibin_ops : public ioperation<T> {
	protected:
		ivariable<T>* a = nullptr;
		ivariable<T>* b = nullptr;
		ivariable<T>* own = nullptr;

		// calc_derive remains abstract
		void copy (const ivariable<T>& other, std::string name = "");
		virtual void replace (
			const ivariable<T>& food,
			const ivariable<T>* newfood) {
			if (a == &food) a = const_cast<ivariable<T>*>(newfood);
			if (b == &food) b = const_cast<ivariable<T>*>(newfood);
		}

		virtual void shape_eval (void);
		// operator () getters
		virtual std::string get_symb (void) = 0;
		virtual std::function<T(T, T)> get_op (void) = 0;

		friend class univar_func<T>;

	public:
		virtual ~ibin_ops (void) {
			if (own) delete own;
			if (a) a->get_consumers().erase(this);
			if (b) b->get_consumers().erase(this);
		}
		virtual ivariable<T>& operator () (ivariable<T>& a, ivariable<T>& b);
		virtual ivariable<T>& operator () (ivariable<T>& a, T b);
		virtual ivariable<T>& operator () (T a, ivariable<T>& b);
		virtual ibin_ops<T>& operator = (const ivariable<T>& other);

		virtual const tensor<T>& eval (void);
};

// DERIVATION

// TODO extend ioperation interface for unique operations like derive and expose
// TODO change ALL derivative elementary operations to use ioperations as to
// enable n-th derivative (derivative of derivative of derivative...)
// TODO test derivation
template <typename T>
class derive : public ibin_ops<T> {
	protected:
		virtual tensor<T>* calc_derive (ivariable<T>* over) const {
			// TODO implement calc_derive
			return nullptr;
		}
		derive (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "/Derive?"; }
		std::function<T(T, T)> get_op (void);

	public:
		derive (void) {}
		derive (ivariable<T>& func, ivariable<T>& over) { (*this)(func, over); }
		virtual derive<T>* clone (std::string name = "");

		virtual const tensor<T>& eval (void) {
			tensor<T>* prime = this->a->derive(this->b);
			this->out = *prime;
			delete prime;
			return this->out;
		}
};

// addition

template <typename T>
class add : public ibin_ops<T> {
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

// subtraction

template <typename T>
class sub : public ibin_ops<T> {
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
class mul : public ibin_ops<T> {
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
class div : public ibin_ops<T> {
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

}

#include "../../src/variable/bin_ops.tpp"

#endif /* bin_ops_hpp */
