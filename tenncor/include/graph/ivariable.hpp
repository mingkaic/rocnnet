//
//  ivariable.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "../memory/session.hpp"
#include "initializer.hpp"
#include "tensor/tensor_op.hpp"
#include "graph/ccoms/subject.hpp"

#pragma once
#ifndef ivariable_hpp
#define ivariable_hpp

namespace nnet
{

template <typename T>
class assign;
template <typename T>
class ioptimizer;

// VARIABLE INTERFACE

// DEFAULTS TO DOWN-UP VARIABLE (INFORMATION IS OBTAINED FROM LEAF NODES: Synthesized Attribute as oppose to Inherited)

template <typename T>
class ivariable : public ccoms::subject
{
	private:
		std::string name_;
		
	protected:
		// WRAPPER CONTENT
		std::unique_ptr<tensor<T> > out_ = nullptr;

		// GRADIENT STATE
		// TODO: somehow differentiate gradient order (0 = non-gradient node, 1st order, etc.)

		void copy (const ivariable<T>& other, std::string name = "");
		ivariable (const ivariable<T>& other, std::string name);

		virtual ivariable<T>* clone_impl (std::string name) = 0;

		// protected members need to be accessed by other operations
		friend class assign<T>;
		friend class ioptimizer<T>;

	public:
		ivariable (const tensorshape& shape, std::string name);
		virtual ~ivariable (void);
		
		// COPY
		// call abstract cloner
		ivariable<T>* clone (std::string name = "");
		virtual ivariable<T>& operator = (const ivariable<T>& other);
		
		// MOVE
		// TODO Implement

		std::string get_name (void) const { return name_; }
		virtual tensorshape get_shape (void) const { return this->out_->get_shape(); }
		
		// DATA EXPOSURE TO PARENT/DEPENDENT NODES
		virtual tensor<T>* get_eval (void) { return this->out_.get(); }

		virtual ivariable<T>* get_gradient (void) = 0;
};

template <typename T>
std::vector<T> expose (ivariable<T>* var)
{
	tensor<T>* ten = var->get_eval();
	T* raw = ten->get_raw();
	return std::vector<T>(raw, raw + ten->n_elems());
}

}

#include "../../src/graph/ivariable.ipp"

#endif /* ivariable_hpp */
