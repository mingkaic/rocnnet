/*!
 *
 *  generator.hpp
 *  cnnet
 *
 *  Purpose:
 *  generate values using init given shape dependency on shape_dep node
 *
 *  Created by Mingkai Chen on 2017-07-18.
 *  Copyright © 2017 Mingkai Chen. All rights reserved.
 *
 */

#include "graph/connector/iconnector.hpp"

#pragma once
#ifndef ROCNNET_GENERATOR_HPP
#define ROCNNET_GENERATOR_HPP

namespace nnet
{

template <typename T>
class generator : public iconnector<T>
{
public:
	virtual ~generator (void)
	{
		clean_up();
	}

	// >>>> BUILDER TO FORCE HEAP ALLOCATION <<<<
	//! builder for generator, clones init
	static generator<T>* get (inode<T>* shape_dep,
		const initializer<T>& init, std::string name = "generator")
	{
		return new generator(shape_dep, init, name);
	}

	// >>>> CLONER & ASSIGNMENT OPERATORS <<<<
	//! clone function
	generator<T>* clone (void) const
	{
		return static_cast<generator<T>*>(this->clone_impl());
	}

	//! move function
	generator<T>* move (void)
	{
		return static_cast<generator<T>*>(this->move_impl());
	}

	//! declare copy assignment to copy over ?
	virtual generator<T>& operator = (const generator<T>& other)
	{
		if (this != &other)
		{
			iconnector<T>::operator = (other);
			clean_up();
			copy_helper(other);
			this->notify(UPDATE);
		}
		return *this;
	}

	//! declare move assignment to move over ?
	virtual generator<T>& operator = (generator<T>&& other)
	{
		if (this != &other)
		{
			iconnector<T>::operator = (std::move(other));
			clean_up();
			move_helper(std::move(other));
			this->notify(UPDATE);
		}
		return *this;
	}

	// >>>> FORWARD & BACKWARD DATA <<<<
	//! grab a temporary value traversing top-down
	//! allocates out tensor. caller owns out
	virtual void temporary_eval (const iconnector<T>* target, inode<T>*& out) const
	{
		out = constant<T>::get(1);
	}

	//! get gradient wrt some node, applies jacobians before evaluting resulting tensor
	//! may call get_gradient
	virtual varptr<T> derive (inode<T>* wrt)
	{
		if (this != wrt)
		{
			return constant<T>::get_shared_zero();
		}
		return constant<T>::get_shared_one();
	}

	//! Utility function: get data shape
	virtual tensorshape get_shape (void) const
	{
		return data_->get_shape();
	}

	// >>>> GRAPH STATUS <<<<
	//! get gradient leaves
	virtual std::unordered_set<ileaf<T>*> get_leaves (void) const
	{
		return {};
	}

	// >>>> NODE STATUS <<<<
	//! check if the arguments are good; data is available
	virtual bool good_status (void) const
	{
		return nullptr != data_;
	}

	//! Inherited from inode: data_ takes data from proto
	virtual bool read_proto (const tenncor::tensor_proto&) {}

	// >>>> CALLED BY OBSERVER TO UPDATE <<<<
	//! Inherited from iobserver: update data
	//! Updates gcache_ and data_
	virtual void update (std::unordered_set<size_t> argidx)
	{
		inode<T>* dep = dynamic_cast<inode<T>*>(this->dependencies_[0]);
		if (nullptr == dep)
		{
			// self destroy
			this->notify(UNSUBSCRIBE);
		}
		tensorshape depshape = dep->get_shape();
		if (false == dep->good_status() || false == depshape.is_fully_defined())
		{
			return;
		}
		if (nullptr == data_)
		{
			// init
			data_ = new tensor<T>(depshape);
			(*init_)(data_);
			this->notify(UPDATE);
		}
		else if (false == data_->get_shape().is_compatible_with(depshape))
		{
			// reshape
			data_->set_shape(depshape);
			(*init_)(data_);
			this->notify(UPDATE);
		}
	}

protected:
	generator (inode<T>* shape_dep, const initializer<T>& init, std::string name) :
		iconnector<T>({shape_dep}, name)
	{
		this->init_ = init.clone();
		this->mergible_ = false;
	}

	generator (const generator<T>& other)
	{
		copy_helper(other);
	}

	generator (generator<T>&& other)
	{
		move_helper(std::move(other));
	}

	// >>>> POLYMORPHIC CLONERS <<<<
	//! clone abstraction function
	virtual inode<T>* clone_impl (void) const
	{
		return new generator(*this);
	}

	//! move abstraction function
	virtual inode<T>* move_impl (void)
	{
		return new generator(std::move(*this));
	}

	virtual const tensor<T>* get_eval (void) const
	{
		return data_;
	}

	virtual inode<T>* get_gradient (variable<T>* leaf)
	{
		return constant<T>::get_shared_zero();
	}

	virtual void death_on_broken (void)
	{
		delete this;
	}

	virtual void death_on_noparent (void)
	{
		delete this;
	}

	virtual typename iconnector<T>::summary_series summarize (void) const
	{
		return {};
	}

private:
	void copy_helper (const generator<T>& other) const
	{
		init_ = other.init_->clone();
		data_ = other.data_->clone();
	}

	void move_helper (generator<T>&& other)
	{
		init_ = other.init_->move();
		data_ = other.data_->move();
	}

	void clean_up (void)
	{
		if (init_) delete init_;
		if (data_) delete data_;
		init_ = nullptr;
		data_ = nullptr;
	}

	//! initialization handler, owns this
	initializer<T>* init_ = nullptr;

	//! tensor data
	tensor<T>* data_ = nullptr;
};

}

#endif /* ROCNNET_GENERATOR_HPP */
