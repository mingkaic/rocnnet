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
		VAR_PTR<T> a = nullptr;
		VAR_PTR<T> b = nullptr;
		VAR_PTR<T> own = nullptr;

		// calc_gradient remains abstract
		void copy (const ivariable<T>& other, std::string name = "");
		virtual void replace (ivariable<T>* food, VAR_PTR<T> newfood) {
			if (a.get() == food) a = newfood;
			if (b.get() == food) b = newfood;
		}

		virtual void shape_eval (void);
		// operator () getters
		virtual std::string get_symb (void) = 0;
		virtual std::function<T(T, T)> get_op (void) = 0;

		friend class univar_func<T>;

	public:
		virtual ~ibin_ops (void) {}
		std::shared_ptr<ibin_ops<T> > clone (std::string name = "") {
			return std::static_pointer_cast<ibin_ops<T>, ievoker<T> >(this->clone_impl(name));
		}

		virtual ivariable<T>& operator () (VAR_PTR<T> a, VAR_PTR<T> b);
		virtual ivariable<T>& operator () (VAR_PTR<T> a, T b);
		virtual ivariable<T>& operator () (T a, VAR_PTR<T> b);

		virtual ibin_ops<T>& operator = (const ivariable<T>& other);

		virtual const tensor<T>& eval (void);
};

// addition

template <typename T>
class add : public ibin_ops<T> {
	protected:
		// backward chaining for AD
		virtual tensor<T>* calc_gradient (WEAK_VAR_PTR<T> over) const;
		add (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "+"; }
		std::function<T(T, T)> get_op (void);

		virtual EVOKER_PTR<T> clone_impl (std::string name);

	public:
		add (void) {}
		add (VAR_PTR<T> a, VAR_PTR<T> b) { (*this)(a, b); }
		add (VAR_PTR<T> a, T b) { (*this)(a, b); }
		add (T a, VAR_PTR<T> b) { (*this)(a, b); }

		std::shared_ptr<add<T> > clone (std::string name = "") {
			return std::static_pointer_cast<add<T>, ievoker<T> >(clone_impl(name));
		}
};

// subtraction

template <typename T>
class sub : public ibin_ops<T> {
	protected:
		// backward chaining for AD
		virtual tensor<T>* calc_gradient (WEAK_VAR_PTR<T> over) const;
		sub (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "-"; }
		std::function<T(T, T)> get_op (void);

		virtual EVOKER_PTR<T> clone_impl (std::string name);

	public:
		sub (void) {}
		sub (VAR_PTR<T> a, VAR_PTR<T> b) { (*this)(a, b); }
		sub (VAR_PTR<T> a, T b) { (*this)(a, b); }
		sub (T a, VAR_PTR<T> b) { (*this)(a, b); }

		std::shared_ptr<sub<T> > clone (std::string name = "") {
			return std::static_pointer_cast<sub<T>, ievoker<T> >(clone_impl(name));
		}
};

// element wise multiplication

template <typename T>
class mul : public ibin_ops<T> {
	protected:
		// backward chaining for AD
		virtual tensor<T>* calc_gradient (WEAK_VAR_PTR<T> over) const;
		mul (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "*"; }
		std::function<T(T, T)> get_op (void);

		virtual EVOKER_PTR<T> clone_impl (std::string name);

	public:
		mul (void) {}
		mul (VAR_PTR<T> a, VAR_PTR<T> b) { (*this)(a, b); }
		mul (VAR_PTR<T> a, T b) { (*this)(a, b); }
		mul (T a, VAR_PTR<T> b) { (*this)(a, b); }

		std::shared_ptr<mul<T> > clone (std::string name = "") {
			return std::static_pointer_cast<mul<T>, ievoker<T> >(clone_impl(name));
		}
};

// element wise division

template <typename T>
class div : public ibin_ops<T> {
	protected:
		// backward chaining for AD
		virtual tensor<T>* calc_gradient (WEAK_VAR_PTR<T> over) const;
		div (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "/"; }
		std::function<T(T, T)> get_op (void);

		virtual EVOKER_PTR<T> clone_impl (std::string name);

	public:
		div (void) {}
		div (VAR_PTR<T> a, VAR_PTR<T> b) { (*this)(a, b); }
		div (VAR_PTR<T> a, T b) { (*this)(a, b); }
		div (T a, VAR_PTR<T> b) { (*this)(a, b); }

		std::shared_ptr<div<T> > clone (std::string name = "") {
			return std::static_pointer_cast<div<T>, ievoker<T> >(clone_impl(name));
		}
};

}

#include "../../src/variable/bin_ops.tpp"

#endif /* bin_ops_hpp */
