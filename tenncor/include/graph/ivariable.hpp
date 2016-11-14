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

#include "../memory/session.hpp"
#include "tensor/tensor.hpp"
#include "tensor/tensor_op.hpp"
#include "graph/ccoms/subject.hpp"

namespace nnet {

// INITIALIZERS

template <typename T>
class initializer {
	protected:
		void delegate_task(tensor<T>& ten, std::function<void(T*, size_t)> op) {
			op(ten.get_raw(), ten.n_elems());
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

// VARIABLE INTERFACE

// DEFAULTS TO DOWN-UP VARIABLE (INFORMATION IS OBTAINED FROM LEAF NODES: Synthesized Attribute as oppose to Inherited)

template <typename T>
class ivariable : public ccoms::subject {
	private:
		std::string name_;
		
	protected:
		// WRAPPER CONTENT
		std::unique_ptr<tensor<T> > out_ = nullptr;

		// GRADIENT STATE
		// TODO: somehow differentiate gradient order (0 = non-gradient node, 1st order, etc.)

		// backward chaining for AD
		void copy (const ivariable<T>& other, std::string name = "");

		virtual ivariable<T>* clone_impl (std::string name) = 0;

		ivariable (const ivariable<T>& other, std::string name) {
			copy(other, name);
		}

		// protected members need to be accessed by other operations
		friend class update<T>;
		friend class ioptimizer<T>;

	public:
		ivariable (const tensorshape& shape, std::string name) : name_(name) {
			if (shape.is_fully_defined()) {
				out_ = std::make_unique<tensor<T> >(shape);
			}
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
		virtual tensorshape get_shape (void) const { return this->out_->get_shape(); }
		
		// DATA EXPOSURE TO PARENT/DEPENDENT NODES
		virtual tensor<T>* get_eval (void) {
			return this->out_.get();
		}

		virtual ivariable<T>* get_gradient (void) = 0;
};

}

#include "../../src/graph/ivariable.ipp"

#endif /* ivariable_hpp */
