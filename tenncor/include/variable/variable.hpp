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
		virtual initializer<T>* copy (void) = 0;
};

template <typename T>
class const_init : public initializer<T> {
	private:
		T value;

	public:
		const_init (T value) : value(value) {}

		virtual void operator () (tensor<T>& in);
		virtual initializer<T>* copy (void) {
			return new const_init(value);
		}
};

template <typename T>
class elementary;

template <typename T>
class random_uniform : public initializer<T> {
	private:
		std::uniform_real_distribution<T> distribution;

	public:
		random_uniform (T min, T max) {
			distribution = std::uniform_real_distribution<T>(min, max);
		}

		virtual void operator () (tensor<T>& in);
		virtual initializer<T>* copy (void) {
			return new random_uniform(distribution.min(), distribution.max());
		}
};

template <typename T>
class ivariable : public ievoker<T> {
	protected:
		tensor<T> out;
		ivariable<T>* integral = nullptr;
		VAR_PTR<T> grad = nullptr;
		std::string name;

		// TODO make shared
		std::unordered_set<ioperation<T>*> consumers; // next

		// backward chaining for AD
		void copy (const ivariable<T>& other, std::string name = "");

		virtual const tensor<T>& calc_eval (VAR_PTR<T> active) = 0;

		virtual void make_gradient (void) = 0;

		virtual void set_gradient (VAR_PTR<T> g) {
			if (nullptr == grad) {
				grad = g;
				grad->integral = this;
			}
		}

		// protected members need to be accessed by other operations
		friend class ioperation<T>;
		friend class update<T>;
		friend class placeholder<T>;
		friend class elementary<T>;

	public:
		ivariable (void);
		virtual ~ivariable (void);
		virtual ivariable<T>& operator = (const ivariable<T>& other);

		std::shared_ptr<ivariable<T> > clone (std::string name = "") {
			return std::static_pointer_cast<ivariable<T>, ievoker<T> >(this->clone_impl(name));
		}

		std::string get_name (void) const { return name; }
		virtual tensor_shape get_shape (void) const { return this->out.get_shape(); }

		std::unordered_set<ioperation<T>*>& get_consumers (void) { return consumers; }

		virtual const tensor<T>& eval (void) = 0;

		virtual VAR_PTR<T> get_gradient (void) {
			if (nullptr == this->grad) make_gradient();
			return this->grad;
		}

		// ignores all leaf nodes that are not active in evaluation (used for the gradient tree)
		virtual const tensor<T>& eval (VAR_PTR<T> active) {
			static tensor<T> ones = tensor<T>(1);
			if (nullptr == integral) {
				return eval();
			} else if (integral == active.get()) {
				return ones;
			}
			return calc_eval(active);
		}
};

// extend tensors by composition
// also holds initializer (in operation)f
template <typename T>
class variable : public ivariable<T> {
	protected:
		bool is_init = false;
		initializer<T>* init = nullptr;

		void copy (const variable<T>& other, std::string name="");

		variable (const variable<T>& other, std::string name);
		virtual EVOKER_PTR<T> clone_impl (std::string name);

		const tensor<T>& calc_eval (VAR_PTR<T> active) {
			static tensor<T> zero = tensor<T>(0);
			// TODO: calculate matrix calculus
			return zero;
		}

		virtual void make_gradient (void) {
			this->set_gradient(std::make_shared<variable<T> >(1));
		}

	public:
		variable (T scalar);
		variable (std::string name = "");
		variable (const tensor_shape& shape, std::string name = "");
		variable (const tensor_shape& shape, initializer<T>& init, std::string name = "");
		virtual ~variable (void);
		virtual variable<T>& operator = (const ivariable<T>& other);

		std::shared_ptr<variable<T> > clone (std::string name = "") {
			return std::static_pointer_cast<variable<T>, ievoker<T> >(clone_impl(name));
		}

		bool can_init (void) const { return init != nullptr; }

		// required by variables using initializer (not by placeholder)
		// initializer can be call multiple times to reset values
		// TODO: allow session to flag variables as init once only to ensure safety
		virtual tensor<T>& initialize (void);
		virtual tensor<T>& initialize (tensor_shape alloc_shape);

		virtual const tensor<T>& eval (void);

		// tensor<T> scatter_sub (IndexedSlices sparse_delta, use_locking = false);
		// variable specific operations (inherit?)
		// void eval(feed_dict=None, session=None);
		// void eval (Session session);
		// graph get_graph (void);
};

template <typename T>
class placeholder : public variable<T> {
	private:
		// used by assignment operators to freely initialized inner tensor
		struct open_init;
		void consumer_reshape (void);
		placeholder (const placeholder<T>& other, std::string name);

	protected:
		virtual EVOKER_PTR<T> clone_impl (std::string name);

	public:
		placeholder (std::string name = ""); // super generic placeholder
		placeholder (const tensor_shape& shape, std::string name = "");
		// assign raw data according to 1 dimension representation of inner tensor
		virtual variable<T>& assign (VAR_PTR<T> other);
		virtual variable<T>& operator = (std::vector<T> data);
		virtual variable<T>& operator = (const tensor<T>& data);

		std::shared_ptr<placeholder<T> > clone (std::string name = "") {
			return std::static_pointer_cast<placeholder<T>, ievoker<T> >(clone_impl(name));
		}

		// replace with shared_ptr<unique_ptr<placeholder<T> > >...
		void replace (const placeholder<T>& other);

		// initialize does nothing
		virtual tensor<T>& initialize (void) { return this->out; }
		virtual tensor<T>& initialize (tensor_shape alloc_shape) { return this->out; }
};

template <typename T>
using PLACEHOLDER_PTR = std::shared_ptr<placeholder<T> >;

template <typename T>
constexpr auto PLACEHOLDER_TO_VAR = std::static_pointer_cast<ivariable<T>, placeholder<T> >;

}

#include "../../src/variable/variable.tpp"

#endif /* variable_hpp */
