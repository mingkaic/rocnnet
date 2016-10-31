//
//  ivariable.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef ivariable_hpp
#define ivariable_hpp

#include "../evoker.hpp"

namespace nnet {

// INITIALIZERS

template <typename T>
class initializer {
	protected:
		void delegate_task(tensor<T>& value,
						   std::function<void(T*, size_t)> op) {
			op(value._raw_data, value.n_elems());
		}

	public:
		virtual ~initializer (void) {}

		virtual void operator () (tensor<T>& in) = 0;
		virtual initializer<T>* clone (void) = 0;
};

template <typename T>
class const_init : public initializer<T> {
	private:
		T _value;

	public:
		const_init (T value) : _value(value) {}

		virtual void operator () (tensor<T>& in);
		virtual initializer<T>* clone (void) {
			return new const_init(_value);
		}
};

template <typename T>
class random_uniform : public initializer<T> {
	private:
		std::uniform_real_distribution<T> _distribution;

	public:
		random_uniform (T min, T max) {
			_distribution = std::uniform_real_distribution<T>(min, max);
		}

		virtual void operator () (tensor<T>& in);
		virtual initializer<T>* clone (void) {
			return new random_uniform(_distribution.min(), _distribution.max());
		}
};

template <typename T>
class ivar_init;

template <typename T>
class ioperation;

template <typename T>
class update;

template <typename T>
class elementary;

template <typename T>
class ioptimizer;

// VARIABLE INTERFACE

template <typename T>
class ivariable : public ievoker<T> {
	protected:
		// weak pointer to this
		WEAK_VAR_PTR<T> _self_ref;

		size_t _grad_order = 0; // TODO IMPLEMENT ON GRADIENTS TO DISTINGUISH INTEGRALS AND GRADS
		tensor<T> _out;
		std::string name;
		WEAK_VAR_PTR<T> integral;

		nnutils::WEAK_SET<ivariable<T> > _leaves;
		// TODO make weak
		std::unordered_set<ioperation<T>*> consumers; // next

		// backward chaining for AD
		void copy (const ivariable<T>& other, std::string name = "");

		virtual void make_gradient (VAR_PTR<T>& safety_ref) = 0;
		// different depending on leaf node or not
		virtual void set_gradient (VAR_PTR<T> g) = 0;

		static VAR_PTR<T> make_shared(ivariable<T>* ptr, bool add_leaf = false) {
			VAR_PTR<T> inst = VAR_PTR<T>(ptr);
			inst->_self_ref = inst;
			if (add_leaf) {
				inst->_leaves.insert(inst);
			}
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
		friend class ioptimizer<T>;

	public:
		virtual ~ivariable (void);
		virtual ivariable<T>& operator = (const ivariable<T>& other);

		std::shared_ptr<ivariable<T> > clone (std::string name = "") {
			return std::static_pointer_cast<ivariable<T>, ievoker<T> >(this->clone_impl(name));
		}

		std::string get_name (void) const { return name; }
		virtual tensor_shape get_shape (void) const { return this->_out.get_shape(); }

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

// INITIALIZER MANAGING INTERFACE

template <typename T>
class ivar_init : public ivariable<T> {
	protected:
		bool is_init = false;
		initializer<T>* init = nullptr;
		WEAK_VAR_PTR<T> grad;

		virtual void make_gradient (VAR_PTR<T>& safety_ref) {
			this->integral = this->grad = this->_self_ref;
			safety_ref = this->_self_ref.lock();
		}

		virtual void set_gradient (VAR_PTR<T> g) {
			if (grad.expired() && nullptr != g) {
				grad = g;
				g->integral = this->_self_ref;
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
			return this->_out;
		}

		virtual VAR_PTR<T> get_gradient (void) {
			VAR_PTR<T> safety_ref;
			if (this->grad.expired()) make_gradient(safety_ref);
			else safety_ref = this->grad.lock();
			return safety_ref;
		}
};

}

#include "../../../src/graph/variable/ivariable.ipp"

#endif /* ivariable_hpp */