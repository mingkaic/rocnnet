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
		VAR_PTR<T> var = nullptr;

		// backward chaining for AD
		virtual tensor<T>* calc_gradient (WEAK_VAR_PTR<T> over) const {
			return this->var->gradient(over);
		}
		void copy (const ivariable<T>& other, std::string name = "");
		virtual void replace (ivariable<T>* food, VAR_PTR<T> newfood);
		virtual void shape_eval (void);
		// operator () getters
		virtual std::string get_symb (void) = 0;

		friend class univar_func<T>;

	public:
		virtual ~iunar_ops (void) {}
		virtual ivariable<T>& operator () (VAR_PTR<T> in);
		virtual iunar_ops<T>& operator = (const ivariable<T>& other);
};

// OUT NODE

template <typename T>
class expose : public iunar_ops<T> {
	protected:
		expose (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "expose"; }
		virtual std::shared_ptr<ivariable<T> > clone_impl (std::string name);

	public:
		expose (void) {}
		expose (VAR_PTR<T> var) { (*this)(var); }
		virtual const tensor<T>& eval (void);

		std::shared_ptr<expose<T> > clone (std::string name = "") {
			return std::static_pointer_cast<expose<T>, ivariable<T> >(clone_impl(name));
		}

		// non-inheriteds
		// evaluates consumed operation
		virtual std::vector<T> get_raw (void);
		// extracts derivative based on the LAST evaluation
		// doesn't evaluate
		virtual std::vector<T> get_derive (WEAK_VAR_PTR<T> over) const;
};

// GRADIENT

// TODO extend ioperation interface for unique operations like gradient and expose
// TODO change ALL derivative elementary operations to use ioperations as to
// enable n-th derivative (derivative of derivative of derivative...)
// TODO test derivation
template <typename T>
class gradient : public iunar_ops<T> {
	private:
		VAR_PTR<T> over = nullptr;

	protected:
		virtual tensor<T>* calc_gradient (WEAK_VAR_PTR<T> over) const {
			// TODO implement calc_gradient
			return nullptr;
		}
		gradient (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "/gradient?"; }
		virtual std::shared_ptr<ivariable<T> > clone_impl (std::string name);

	public:
		gradient (VAR_PTR<T> func) { (*this)(func); }
		gradient (VAR_PTR<T> func, VAR_PTR<T> over) : over(over) { (*this)(func); }
		virtual const tensor<T>& eval (void);

		void set_over (VAR_PTR<T> over) { this->over = over; }

		std::shared_ptr<gradient<T> > clone (std::string name = "") {
			return std::static_pointer_cast<gradient<T>, ivariable<T> >(clone_impl(name));
		}
};

// USED FOR ELEMENT WISE OPERATIONS ONLY

template <typename T>
class iunar_elem_ops : public iunar_ops<T> {
	protected:
		virtual std::function<T(T)> get_op (void) = 0; // these are for elementary and simple operations

	public:
		virtual const tensor<T>& eval (void);
};

// NEGATION

template <typename T>
class neg : public iunar_elem_ops<T> {
	protected:
		virtual tensor<T>* calc_gradient (WEAK_VAR_PTR<T> over) const;
		neg (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "-"; }
		std::function<T(T)> get_op (void);
		virtual std::shared_ptr<ivariable<T> > clone_impl (std::string name);

	public:
		neg (void) {}
		neg (VAR_PTR<T> var) { (*this)(var); }

		std::shared_ptr<neg<T> > clone (std::string name = "") {
			return std::static_pointer_cast<neg<T>, ivariable<T> >(clone_impl(name));
		}
};

// SINE

template <typename T>
class sin : public iunar_elem_ops<T> {
	protected:
		virtual tensor<T>* calc_gradient (WEAK_VAR_PTR<T> over) const;
		sin (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "sin"; }
		std::function<T(T)> get_op (void);
		virtual std::shared_ptr<ivariable<T> > clone_impl (std::string name);

	public:
		sin (void) {}
		sin (VAR_PTR<T> var) { (*this)(var); }

		std::shared_ptr<sin<T> > clone (std::string name = "") {
			return std::static_pointer_cast<sin<T>, ivariable<T> >(clone_impl(name));
		}
};

// COSINE

template <typename T>
class cos : public iunar_elem_ops<T> {
	protected:
		virtual tensor<T>* calc_gradient (WEAK_VAR_PTR<T> over) const;
		cos (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "cos"; }
		std::function<T(T)> get_op (void);
		virtual std::shared_ptr<ivariable<T> > clone_impl (std::string name);

	public:
		cos (void) {}
		cos (VAR_PTR<T> var) { (*this)(var); }

		std::shared_ptr<cos<T> > clone (std::string name = "") {
			return std::static_pointer_cast<cos<T>, ivariable<T> >(clone_impl(name));
		}
};

// TANGENT

template <typename T>
class tan : public iunar_elem_ops<T> {
	protected:
		virtual tensor<T>* calc_gradient (WEAK_VAR_PTR<T> over) const;
		tan (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "tan"; }
		std::function<T(T)> get_op (void);
		virtual std::shared_ptr<ivariable<T> > clone_impl (std::string name);

	public:
		tan (void) {}
		tan (VAR_PTR<T> var) { (*this)(var); }

		std::shared_ptr<tan<T> > clone (std::string name = "") {
			return std::static_pointer_cast<tan<T>, ivariable<T> >(clone_impl(name));
		}
};

// COSECANT

template <typename T>
class csc : public iunar_elem_ops<T> {
	protected:
		virtual tensor<T>* calc_gradient (WEAK_VAR_PTR<T> over) const;
		csc (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "csc"; }
		std::function<T(T)> get_op (void);
		virtual std::shared_ptr<ivariable<T> > clone_impl (std::string name);

	public:
		csc (void) {}
		csc (VAR_PTR<T> var) { (*this)(var); }

		std::shared_ptr<csc<T> > clone (std::string name = "") {
			return std::static_pointer_cast<csc<T>, ivariable<T> >(clone_impl(name));
		}
};

// SECANT

template <typename T>
class sec : public iunar_elem_ops<T> {
	protected:
		virtual tensor<T>* calc_gradient (WEAK_VAR_PTR<T> over) const;
		sec (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "sec"; }
		std::function<T(T)> get_op (void);
		virtual std::shared_ptr<ivariable<T> > clone_impl (std::string name);

	public:
		sec (void) {}
		sec (VAR_PTR<T> var) { (*this)(var); }

		std::shared_ptr<sec<T> > clone (std::string name = "") {
			return std::static_pointer_cast<sec<T>, ivariable<T> >(clone_impl(name));
		}
};

// COTANGENT

template <typename T>
class cot : public iunar_elem_ops<T> {
	protected:
		virtual tensor<T>* calc_gradient (WEAK_VAR_PTR<T> over) const;
		cot (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "cot"; }
		std::function<T(T)> get_op (void);
		virtual std::shared_ptr<ivariable<T> > clone_impl (std::string name);

	public:
		cot (void) {}
		cot (VAR_PTR<T> var) { (*this)(var); }

		std::shared_ptr<cot<T> > clone (std::string name = "") {
			return std::static_pointer_cast<cot<T>, ivariable<T> >(clone_impl(name));
		}
};

// EXPONENT OF E

template <typename T>
class exp : public iunar_elem_ops<T> {
	protected:
		virtual tensor<T>* calc_gradient (WEAK_VAR_PTR<T> over) const;
		exp (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "exp"; }
		std::function<T(T)> get_op (void);
		virtual std::shared_ptr<ivariable<T> > clone_impl (std::string name);

	public:
		exp (void) {}
		exp (VAR_PTR<T> var) { (*this)(var); }

		std::shared_ptr<exp<T> > clone (std::string name = "") {
			return std::static_pointer_cast<exp<T>, ivariable<T> >(clone_impl(name));
		}
};

// CLIP

template <typename T>
class clip_by_value : public iunar_elem_ops<T> {
	private:
		T min;
		T max;

	protected:
		clip_by_value (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "clip_val"; }
		std::function<T(T)> get_op (void);
		virtual std::shared_ptr<ivariable<T> > clone_impl (std::string name);

	public:
		clip_by_value (void) {}
		clip_by_value (VAR_PTR<T> var, T min, T max) : min(min), max(max) { (*this)(var); }
		virtual ~clip_by_value (void) {}

		void set_bounds (T min, T max) { this->min = min; this->max = max; }

		std::shared_ptr<clip_by_value<T> > clone (std::string name = "") {
			return std::static_pointer_cast<clip_by_value<T>, ivariable<T> >(clone_impl(name));
		}
};

template <typename T>
class clip_by_norm : public iunar_elem_ops<T> {
	private:
		T cap;

	protected:
		clip_by_norm (ivariable<T>& var, std::string name) { this->copy(var, name); }

		std::string get_symb (void) { return "clip_norm"; }
		std::function<T(T)> get_op (void);
		virtual std::shared_ptr<ivariable<T> > clone_impl (std::string name);

	public:
		clip_by_norm (void) {}
		clip_by_norm (VAR_PTR<T> var, T cap) : cap(cap) { (*this)(var); }
		virtual ~clip_by_norm (void) {}

		void set_bounds (T min, T max) { this->min = min; this->max = max; }

		std::shared_ptr<clip_by_norm<T> > clone (std::string name = "") {
			return std::static_pointer_cast<clip_by_norm<T>, ivariable<T> >(clone_impl(name));
		}
};

}

#include "../../src/variable/unar_ops.tpp"

#endif /* unar_ops_hpp */
