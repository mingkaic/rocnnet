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
			op(value.raw_data_, value.n_elems());
		}

	public:
		virtual ~initializer (void) {}

		virtual void operator () (tensor<T>& in) = 0;
		virtual initializer<T>* clone (void) = 0;
};

template <typename T>
class const_init : public initializer<T> {
	private:
		T value_;

	public:
		const_init (T value) : value_(value) {}

		virtual void operator () (tensor<T>& in);
		virtual initializer<T>* clone (void) {
			return new const_init(value_);
		}
};

template <typename T>
class random_uniform : public initializer<T> {
	private:
		std::uniform_real_distribution<T>  distribution_;

	public:
		random_uniform (T min, T max) {
			 distribution_ = std::uniform_real_distribution<T>(min, max);
		}

		virtual void operator () (tensor<T>& in);
		virtual initializer<T>* clone (void) {
			return new random_uniform( distribution_.min(),  distribution_.max());
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

template <typename T>
class constant;

// VARIABLE INTERFACE

template <typename T>
class ivariable : public ievoker<T> {
	protected:
		// weak pointer to this
		WEAK_VAR_PTR<T>  self_ref_;

		size_t grad_order_ = 0; // TODO IMPLEMENT ON GRADIENTS TO DISTINGUISH INTEGRALS AND GRADS
		tensor<T>  out_;
		std::string name;
		WEAK_VAR_PTR<T> integral;

		nnutils::WEAK_SET<ivariable<T> > leaves_;
		// TODO make weak
		std::unordered_set<ioperation<T>*> consumers; // next

		// backward chaining for AD
		void copy (const ivariable<T>& other, std::string name = "");

		virtual void make_gradient (VAR_PTR<T>& safety_ref) = 0;
		// different depending on leaf node or not
		virtual void set_gradient (VAR_PTR<T> g) = 0;

		static VAR_PTR<T> make_shared(ivariable<T>* ptr, bool add_leaf = false) {
			VAR_PTR<T> inst = VAR_PTR<T>(ptr);
			inst-> self_ref_ = inst;
			if (add_leaf) {
				inst-> leaves_.insert(inst);
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
		virtual tensor_shape get_shape (void) const { return this->out_.get_shape(); }

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
		
		// push in_grad through all shape modifying operations until in_grad reaches end_node
		// end_node must be a leaf node (no children)
		virtual VAR_PTR<T> push_to (VAR_PTR<T> in_grad, VAR_PTR<T> end_node) = 0;
};

// INITIALIZER MANAGING INTERFACE

template <typename T>
class ivar_init : public ivariable<T> {
	protected:
		bool is_init = false;
		initializer<T>* init = nullptr;
		WEAK_VAR_PTR<T> grad;

		virtual void make_gradient (VAR_PTR<T>& safety_ref) {
			safety_ref = constant<T>::make(0);
			this->set_gradient(safety_ref);
		}

		virtual void set_gradient (VAR_PTR<T> g) {
			if (grad.expired() && nullptr != g) {
				grad = g;
				g->integral = this-> self_ref_;
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
			return this->out_;
		}

		virtual VAR_PTR<T> get_gradient (void) {
			VAR_PTR<T> safety_ref;
			if (this->grad.expired()) make_gradient(safety_ref);
			else safety_ref = this->grad.lock();
			return safety_ref;
		}
		
		virtual VAR_PTR<T> push_to (VAR_PTR<T> in_grad, VAR_PTR<T> end_node) {
			if (this->self_ref_.lock() == end_node) {
				return in_grad;
			}
			return nullptr;
		}
};

}

#include "../../../src/graph/variable/ivariable.ipp"

#endif /* ivariable_hpp */
