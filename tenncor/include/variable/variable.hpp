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
#include <iostream>

#include "../memory/session.hpp"
#include "../tensor.hpp"

#pragma once
#ifndef variable_hpp
#define variable_hpp

namespace nnet {

template <typename T>
class initializer {
	protected:
		// anti pattern?
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
class placeholder;

template <typename T>
class ivariable {
	protected:
		tensor<T> out;
		std::string name;
		std::unordered_set<ioperation<T>*> consumers; // next
		// backward chaining for AD
		virtual tensor<T>* calc_derive (ivariable<T>* over) const {
			return nullptr;
		}
		void copy (
			ivariable<T> const & other,
			std::string name = "");
		// protected members need to be accessed by other operations
		friend class ioperation<T>;
		friend class placeholder<T>;

	public:
		ivariable (void) {
			session& sess = session::get_instance();
			sess.register_obj(*this);
		}
		virtual ivariable<T>* clone (std::string name = "") = 0;
		virtual ~ivariable (void) {
			session& sess = session::get_instance();
			sess.unregister_obj(*this);
			for (ioperation<T>* con : consumers) {
				con->decompose(*this);
			}
		}
		virtual ivariable<T>& operator = (const ivariable<T>& other);

		// TODO implement
		// operators that will replace elementary operation objects
		ivariable<double> operator + (void) const;
		ivariable<double> operator - (void) const;
		ivariable<double> operator + (const ivariable<double>& b) const;
		ivariable<double> operator - (const ivariable<double>& b) const;
		ivariable<double> operator * (const ivariable<double>& b) const;
		ivariable<double> operator / (const ivariable<double>& b) const;

		std::string get_name (void) const { return name; }
		virtual tensor_shape get_shape (void) const
			{ return this->out.get_shape(); }

		std::unordered_set<ioperation<T>*>& get_consumers (void) { return consumers; }
		// calculate the derivative over input variable given values
		// from the last evaluation. no forward evaluation takes place
		// currently doesn't handle the case of bad evaluation
		// (uninitialized variables) before derive is called
		virtual tensor<T>* derive (ivariable<T>* over) const;
		virtual const tensor<T>& eval (void) = 0;
};

// extend tensors by composition
// also holds initializer (in operation)
template <typename T>
class variable : public ivariable<T> {
	private:
		// Graph graph; // variable manager

	protected:
		bool is_init = false;
		initializer<T>* init = nullptr;

		void copy (const variable<T>& other, std::string name="");
		variable (const variable<T>& other, std::string name);

		// track all variables and placeholders for
		// static/singleton function initialize all
		// static session_manager manager;

	public:
		variable (const tensor_shape& shape, std::string name = "");
		variable (const tensor_shape& shape,
			initializer<T>& init, std::string name = "");
		virtual variable<T>* clone (std::string name = "");
		// variable (const variable<T>& other, std::string name="");
		virtual ~variable (void);
		virtual variable<T>& operator = (const ivariable<T>& other);

		bool can_init (void) const { return init != nullptr; }

		// required by variables using initializer (not by placeholder)
		// initializer can be call multiple times to reset values
		// TODO: allow session to flag variables as init once only to ensure safety
		virtual tensor<T>& initialize (void);
		virtual tensor<T>& initialize (tensor_shape alloc_shape);

		// update raw values with new tensor and some operation on the old and new values
		// where old value is the first parameter and new value is the second parameter
		virtual void update (const tensor<T>& in, std::function<T(T,T)> op);

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

	public:
		placeholder (const tensor_shape& shape, std::string name = "");

		// assign raw data according to 1 dimension representation of inner tensor
		virtual variable<T>& assign (const ivariable<T>& other);
		virtual variable<T>& operator = (std::vector<T> data);
		virtual variable<T>& operator = (const tensor<T>& data);

		// initialize does nothing
		virtual tensor<T>& initialize (void) { return this->out; }
		virtual tensor<T>& initialize (tensor_shape alloc_shape) { return this->out; }
};

}

#include "../../src/variable/variable.tpp"

#endif /* variable_hpp */
