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

#include "evoker.hpp"
#include "graph/observer/subject.hpp"

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
class ileaf;
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
template <typename T>
class matmul;
template <typename T>
class iunar_ops;

// deprecated once evaluation identification is implemented
template <typename T>
class var_buffer : public ivariable<T> {
	protected:
		ivariable<T>* var_;

		virtual ievoker<T>* clone_impl (std::string name) {
			return new var_buffer(var_);
		}

		// do nothing
		virtual void make_gradient (ivariable<T>*& safety_ref) {}
		virtual void set_gradient (ivariable<T>* g) {}

	public:
		var_buffer (ivariable<T>*& in) : var_(in) {}

        var_buffer<T>* clone (std::string name = "") {
			return static_cast<var_buffer<T>*>(clone_impl(name));
		}

		virtual const tensor<T>& get_eval (void) {
			return var_->get_eval();
		}

		virtual ivariable<T>* get_gradient (void) { return nullptr; }
};

// VARIABLE INTERFACE

// DEFAULTS TO DOWN-UP VARIABLE (INFORMATION IS OBTAINED FROM LEAF NODES: Synthesized Attribute as oppose to Inherited)

template <typename T>
class ivariable : public ievoker<T>, public ccoms::subject {
	private:
		std::string name_;
		
	protected:
	    // ZEROS, ONES, TODO remove once get_eval returns pointer
        tensor<T> zeros;
        tensor<T> ones;

		// GRADIENT INFO
		bool short_circuit_ = true;
		
		// WRAPPER CONTENT
		tensor<T>  out_;

		// backward chaining for AD
		void copy (const ivariable<T>& other, std::string name = "");

		virtual ievoker<T>* clone_impl (std::string name) = 0;

		// protected members need to be accessed by other operations
		friend class update<T>;
		friend class ioptimizer<T>;

	public:
		ivariable (const tensor_shape& shape, std::string name) : out_(shape), name_(name), zeros(0), ones(1) {
            session& sess = session::get_instance();
            sess.register_obj(*this);
		}
		virtual ~ivariable (void);
		
		// COPY
		// call abstract cloner
		ivariable<T>* clone (std::string name = "") {
			return static_cast<ivariable<T>*>(clone_impl(name));
		}
		virtual ivariable<T>& operator = (const ivariable<T>& other);
		
		// MOVER
		// TODO Implement

		std::string get_name (void) const { return name_; }
		virtual tensor_shape get_shape (void) const { return this->out_.get_shape(); }
		
		// DATA EXPOSURE TO PARENT/DEPENDENT NODES
		virtual const tensor<T>& get_eval (void) const {
			if (short_circuit_) {
				return this->out_;
			}
			return zeros;
		}

		virtual ivariable<T>* get_gradient (void) = 0;
};

}

#include "../../src/graph/ivariable.ipp"

#endif /* ivariable_hpp */
