//
//  variable.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <list>
#include <ctime>
#include <random>
#include <new>
#include <memory>

#include "../evoker.hpp"

#pragma once
#ifndef variable_hpp
#define variable_hpp

namespace nnet {

template <typename T>
class initializer {
	protected:
		void delegate_task(tensor<T>& value,
			std::function<void(T*, size_t)> op) {
			op(value.raw_data, value.n_elems());
		}

	public:
		virtual ~initializer (void) {}

		virtual void operator () (tensor<T>& in) = 0;
		virtual initializer<T>* clone (void) = 0;
};

template <typename T>
class const_init : public initializer<T> {
	private:
		T value;

	public:
		const_init (T value) : value(value) {}

		virtual void operator () (tensor<T>& in);
		virtual initializer<T>* clone (void) {
			return new const_init(value);
		}
};

template <typename T>
class random_uniform : public initializer<T> {
	private:
		std::uniform_real_distribution<T> distribution;

	public:
		random_uniform (T min, T max) {
			distribution = std::uniform_real_distribution<T>(min, max);
		}

		virtual void operator () (tensor<T>& in);
		virtual initializer<T>* clone (void) {
			return new random_uniform(distribution.min(), distribution.max());
		}
};

template <typename T>
class ivar_init;

template <typename T>
class elementary;

template <typename T>
class ivariable : public ievoker<T> {
	protected:
		// weak pointer to this
		WEAK_VAR_PTR<T> self_ref;

		tensor<T> out;
		std::string name;
		WEAK_VAR_PTR<T> integral;

		size_t grad_order = 0; // TODO IMPLEMENT ON GRADIENTS TO DISTINGUISH INTEGRALS AND GRADS

		// TODO make weak
		std::unordered_set<ioperation<T>*> consumers; // next

		// backward chaining for AD
		void copy (const ivariable<T>& other, std::string name = "");

		virtual void make_gradient (VAR_PTR<T>& safety_ref) = 0;
		// different depending on leaf node or not
		// TODO get rid of
		virtual void set_gradient (VAR_PTR<T> g) = 0;

		static VAR_PTR<T> make_shared(ivariable<T>* ptr) {
			VAR_PTR<T> inst = VAR_PTR<T>(ptr);
			inst->self_ref = inst;
			return inst;
		}

		ivariable (void);

		// FOR LEAVES only
		// flip between 2 tensor states
		class gradient_leaf;

		// protected members need to be accessed by other operations
		friend class ivar_init<T>;
		friend class ioperation<T>;
		friend class update<T>;
		friend class placeholder<T>;
		friend class elementary<T>;

	public:
		virtual ~ivariable (void);
		virtual ivariable<T>& operator = (const ivariable<T>& other);

		std::shared_ptr<ivariable<T> > clone (std::string name = "") {
			return std::static_pointer_cast<ivariable<T>, ievoker<T> >(this->clone_impl(name));
		}

		std::string get_name (void) const { return name; }
		virtual tensor_shape get_shape (void) const { return this->out.get_shape(); }

		std::unordered_set<ioperation<T>*>& get_consumers (void) { return consumers; }

		virtual const tensor<T>& eval (void) = 0;

		virtual VAR_PTR<T> get_gradient (void) = 0;

		virtual const tensor<T>& calc_gradient (VAR_PTR<T> active) {
			if (VAR_PTR<T> g = this->get_gradient()) {
				if (std::shared_ptr<gradient_leaf> leaf_ptr =
						std::dynamic_pointer_cast<gradient_leaf>(active->get_gradient())) {
					leaf_ptr->activate(active);
					const tensor<T> &res = g->eval();
					leaf_ptr->deactivate();
					return res;
				} else if (std::shared_ptr<ioperation<T> > op_ptr =
						std::dynamic_pointer_cast<ioperation<T> >(active->get_gradient())) { // if active isn't a leaf
					op_ptr->derive_this = true;
					const tensor<T> &res = g->eval();
					op_ptr->derive_this = false;
					return res;
				}
			}
			static tensor<T> zero(0);
			return zero;
		}
};

// interface for managing initializers
template <typename T>
class ivar_init : public ivariable<T> {
	protected:
		bool is_init = false;
		initializer<T>* init = nullptr;
		WEAK_VAR_PTR<T> grad;

		virtual void make_gradient (VAR_PTR<T>& safety_ref) {
			this->integral = this->grad = this->self_ref;
			safety_ref = this->self_ref.lock();
		}

		virtual void set_gradient (VAR_PTR<T> g) {
			if (grad.expired() && nullptr != g) {
				grad = g;
				g->integral = this->self_ref;
			}
		}

		// used by assignment operators to freely initialized inner tensor
		struct open_init;

		void copy (const ivar_init<T>& other,
					std::string name = "") {
			init = other.init->clone();
			ivariable<T>::copy(other, name);
		}

	public:
		virtual ~ivar_init (void) {
			if (nullptr != this->init) {
				delete this->init;
			}
		}

		virtual ivar_init<T>& operator = (const VAR_PTR<T>& other);

		std::shared_ptr<ivar_init<T> > clone (std::string name = "") {
			return std::static_pointer_cast<ivar_init<T>, ievoker<T> >(this->clone_impl(name));
		}

		bool can_init (void) const { return init != nullptr; }
		virtual const tensor<T>& eval (void) {
			assert(is_init);
			return this->out;
		}

		virtual VAR_PTR<T> get_gradient (void) {
			VAR_PTR<T> safety_ref;
			if (this->grad.expired()) make_gradient(safety_ref);
			else safety_ref = this->grad.lock();
			return safety_ref;
		}
};

template <typename T>
class constant : public ivar_init<T> {
	private:
		constant (T scalar);
		constant (std::vector<T> raw, tensor_shape shape);
		constant (VAR_PTR<T> get_out);

	protected:
		constant (const constant<T>& other, std::string name);
		virtual EVOKER_PTR<T> clone_impl (std::string name);

		virtual void make_gradient (VAR_PTR<T>& safety_ref) {
			VAR_PTR<T> g = std::shared_ptr<constant<T> >(new constant(0));
			this->set_gradient(g);
			safety_ref = g;
		}

	public:
		static VAR_PTR<T> make (T scalar) {
			return ivariable<T>::make_shared(new constant(scalar));
		}
		static VAR_PTR<T> make (std::vector<T> raw, tensor_shape shape) {
			return ivariable<T>::make_shared(new constant(raw, shape));
		}
		static VAR_PTR<T> make (VAR_PTR<T> get_out) {
			return ivariable<T>::make_shared(new constant(get_out));
		}
		virtual ~constant (void) {}

		std::shared_ptr<constant<T> > clone (std::string name = "") {
			return std::static_pointer_cast<constant<T>, ievoker<T> >(clone_impl(name));
		}
};

// extend tensors by composition
// also holds initializer (in operation)f
template <typename T>
class variable : public ivar_init<T> {
	private:
		variable (T scalar);
		variable (std::string name = ""); // this requires initializer later on
		variable (const tensor_shape& shape, std::string name = "");
		variable (const tensor_shape& shape, initializer<T>& init, std::string name = "");

	protected:
		variable (const variable<T>& other, std::string name);
		virtual EVOKER_PTR<T> clone_impl (std::string name);

		virtual void make_gradient (VAR_PTR<T>& safety_ref) {
			// no need to set_gradient, gradient sets this as integral
			this->grad = safety_ref = ivariable<T>::gradient_leaf::make(this->self_ref);
		}

	public:
		static VAR_PTR<T> make (T scalar) {
			return ivariable<T>::make_shared(new variable(scalar));
		}
		static VAR_PTR<T> make (std::string name = "") {
			return ivariable<T>::make_shared(new variable(name));
		}
		static VAR_PTR<T> make (const tensor_shape& shape, std::string name = "") {
			return ivariable<T>::make_shared(new variable(shape, name));
		}
		static VAR_PTR<T> make (const tensor_shape& shape, initializer<T>& init, std::string name = "") {
			return ivariable<T>::make_shared(new variable(shape, init, name));
		}
		virtual ~variable (void) {}

		std::shared_ptr<variable<T> > clone (std::string name = "") {
			return std::static_pointer_cast<variable<T>, ievoker<T> >(clone_impl(name));
		}

		// required by variables using initializer (not by placeholder)
		// initializer can be call multiple times to reset values
		// TODO: allow session to flag variables as init once only to ensure safety
		virtual tensor<T>& initialize (void);
		virtual tensor<T>& initialize (tensor_shape alloc_shape);
};

template <typename T>
class placeholder : public ivar_init<T> {
	private:
		void consumer_reshape (void);
		placeholder (const placeholder<T>& other, std::string name);
		placeholder (std::string name); // super generic placeholder
		placeholder (const tensor_shape& shape, std::string name);

	protected:
		virtual EVOKER_PTR<T> clone_impl (std::string name);

	public:
		static std::shared_ptr<placeholder<T> > make (std::string name = "") {
			VAR_PTR<T> inst = ivariable<T>::make_shared(new placeholder(name));
			return std::static_pointer_cast<placeholder<T> >(inst);
		}
		static std::shared_ptr<placeholder<T> > make (const tensor_shape& shape, std::string name = "") {
			VAR_PTR<T> inst = ivariable<T>::make_shared(new placeholder(shape, name));
			return std::static_pointer_cast<placeholder<T> >(inst);
		}
		virtual ~placeholder (void) {}

		std::shared_ptr<placeholder<T> > clone (std::string name = "") {
			return std::static_pointer_cast<placeholder<T>, ievoker<T> >(clone_impl(name));
		}

		// assign raw data according to 1 dimension representation of inner tensor
		virtual ivariable<T>& assign (VAR_PTR<T> other);
		virtual ivariable<T>& operator = (std::vector<T> data);
		virtual ivariable<T>& operator = (const tensor<T>& data);

		// replace with shared_ptr<unique_ptr<placeholder<T> > >...
		void replace (const placeholder<T>& other);
};

template <typename T>
using PLACEHOLDER_PTR = std::shared_ptr<placeholder<T> >;

template <typename T>
constexpr auto PLACEHOLDER_TO_VAR = std::static_pointer_cast<ivariable<T>, placeholder<T> >;

}

#include "../../src/variable/variable.tpp"

#endif /* variable_hpp */
