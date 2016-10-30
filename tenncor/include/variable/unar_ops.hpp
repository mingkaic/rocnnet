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

#include "operation.hpp"

namespace nnet {

// UNARY OPERATIONS

template <typename T>
class iunar_ops : public ioperation<T> {
	protected:
		// avoid calling ivariable's assignment multiple time
		VAR_PTR<T> var = nullptr;

		void copy (const ivariable<T>& other, std::string name = "");
		virtual void replace (ivariable<T>* food, VAR_PTR<T> newfood);
		virtual void shape_eval (void);
		// operator () getters
		virtual std::string get_symb (void) = 0;

		void init (VAR_PTR<T> var);

		friend class univar_func<T>;

	public:
		virtual ~iunar_ops (void) {}

		virtual iunar_ops<T>& operator = (const ivariable<T>& other);

		std::shared_ptr<iunar_ops<T> > clone (std::string name = "") {
			return std::static_pointer_cast<iunar_ops<T>, ievoker<T> >(this->clone_impl(name));
		}
};

// OUT NODE

template <typename T>
class expose : public iunar_ops<T> {
	protected:
		// backward chaining for AD
		virtual void make_gradient (VAR_PTR<T>& safety_ref) {}

		expose (ivariable<T>& var, std::string name) { this->copy(var, name); }
		expose (VAR_PTR<T> var) { this->init(var); }

		std::string get_symb (void) { return "expose"; }
		virtual EVOKER_PTR<T> clone_impl (std::string name);

	public:
		static std::shared_ptr<expose<T> > make (VAR_PTR<T> var) {
			VAR_PTR<T> inst = ivariable<T>::make_shared(new expose(var));
			return std::static_pointer_cast<expose<T> >(inst);
		}

		virtual const tensor<T>& eval (void);

		std::shared_ptr<expose<T> > clone (std::string name = "") {
			return std::static_pointer_cast<expose<T>, ievoker<T> >(clone_impl(name));
		}

		// non-inheriteds
		// evaluates consumed operation
		virtual std::vector<T> get_raw (void);
		// extracts derivative based on the LAST evaluation
		// doesn't evaluate
		virtual std::vector<T> get_derive (VAR_PTR<T> over) const;
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
		gradient (VAR_PTR<T> func, VAR_PTR<T> over) { this->over = over; this->init(func); }

		std::string get_symb (void) { return "/gradient?"; }
		virtual EVOKER_PTR<T> clone_impl (std::string name);

		virtual void make_gradient (VAR_PTR<T>& safety_ref) {}

	public:
		static VAR_PTR<T> make (VAR_PTR<T> func, VAR_PTR<T> over = nullptr) {
			return ivariable<T>::make_shared(new gradient(func, over));
		}

		virtual const tensor<T>& eval (void);
		void set_over (VAR_PTR<T> over) { this->over = over; }

		std::shared_ptr<gradient<T> > clone (std::string name = "") {
			return std::static_pointer_cast<gradient<T>, ievoker<T> >(clone_impl(name));
		}
};

// USED FOR ELEMENT WISE OPERATIONS ONLY

template <typename T>
class iunar_elem_ops : public iunar_ops<T> {
	protected:
		virtual std::function<T(T)> get_op (void) = 0; // these are for elementary and simple operations

	public:
		virtual ~iunar_elem_ops (void) {}
		std::shared_ptr<iunar_elem_ops<T> > clone (std::string name = "") {
			return std::static_pointer_cast<iunar_elem_ops<T>, ievoker<T> >(this->clone_impl(name));
		}

		virtual const tensor<T>& eval (void);
};

// CLIP

template <typename T>
class clip_by_value : public iunar_elem_ops<T> {
	private:
		T min;
		T max;

	protected:
		// backward chaining for AD
		virtual void make_gradient (VAR_PTR<T>& safety_ref) {
			this->set_gradient(clip_by_value<T>::make(this->var->get_gradient(), min, max));
			safety_ref = this->grad;
		}

		clip_by_value (ivariable<T>& var, std::string name) { this->copy(var, name); }
		clip_by_value (VAR_PTR<T> var, T min, T max) : min(min), max(max) { this->init(var); }

		std::string get_symb (void) { return "clip_val"; }
		std::function<T(T)> get_op (void);
		virtual EVOKER_PTR<T> clone_impl (std::string name);

	public:
		static VAR_PTR<T> make (VAR_PTR<T> var, T min, T max) {
			return ivariable<T>::make_shared(new clip_by_value(var, min, max));
		}
		virtual ~clip_by_value (void) {}

		void set_bounds (T min, T max) { this->min = min; this->max = max; }

		std::shared_ptr<clip_by_value<T> > clone (std::string name = "") {
			return std::static_pointer_cast<clip_by_value<T>, ievoker<T> >(clone_impl(name));
		}
};

template <typename T>
class clip_by_norm : public iunar_elem_ops<T> {
	private:
		T cap;

	protected:
		// backward chaining for AD
		virtual void make_gradient (VAR_PTR<T>& safety_ref) {
			this->set_gradient(clip_by_norm<T>::make(this->var->get_gradient(), cap));
			safety_ref = this->grad;
		}

		clip_by_norm (ivariable<T>& var, std::string name) { this->copy(var, name); }
		clip_by_norm (VAR_PTR<T> var, T cap) : cap(cap) { this->init(var); }

		std::string get_symb (void) { return "clip_norm"; }
		std::function<T(T)> get_op (void);
		virtual EVOKER_PTR<T> clone_impl (std::string name);

	public:
		static VAR_PTR<T> make (VAR_PTR<T> var, T cap) {
			return ivariable<T>::make_shared(new clip_by_norm(var, cap));
		}
		virtual ~clip_by_norm (void) {}

		void set_bounds (T min, T max) { this->min = min; this->max = max; }

		std::shared_ptr<clip_by_norm<T> > clone (std::string name = "") {
			return std::static_pointer_cast<clip_by_norm<T>, ievoker<T> >(clone_impl(name));
		}
};

}

#include "../../src/variable/unar_ops.tpp"

#endif /* unar_ops_hpp */
