//
//  operation.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/leaf/variable.hpp"

#ifdef ioperation_hpp

namespace nnet
{

template <typename T>
void* operation<T>::operator new (size_t size,
	std::vector<inode<T>*> args,
	SHAPER shaper, FORWARD_OP<T> Nf,
	BACK_MAP F, std::string name)
{
	operation<T>* op = static_cast<operation<T>*>(
		::operator new(size, args, shaper, Nf, F, name));
	op->onheap_ = true;
	return op;
}

template <typename T>
operation<T>::operation (
	std::vector<inode<T>*> args,
	SHAPER shaper, FORWARD_OP<T> forward,
	BACK_MAP<T> F, std::string name) :
immutable (args, F, name),
Nf_(shaper, forward)
{
	update(nullptr); // update data_ initially
}

template <typename T>
operation<T>* operation<T>::clone (void) const
{
	return static_cast<operation<T>*>(clone_impl());
}

template <typename T>
operation<T>::operation (operation<T>&& other) :
	immutable<T>(other)
{
	Nf_ = std::move(other.Nf_);
	data_ = std::move(other.data_);
}

template <typename T>
operation<T>& operation<T>::operator = (const operation<T>& other)
{
	if (this != &other)
	{
		immutable<T>::operator = (other);
		Nf_ = other.Nf_;
		data_ = other.data_->clone();
	}
	return *this;
}

template <typename T>
operation<T>& operation<T>::operator = (operation<T>&& other)
{
	if (this != &other)
	{
		immutable<T>::operator = (other);
		Nf_ = std::move(other.Nf_);
		data_ = std::move(other.data_);
	}
	return *this;
}

template <typename T>
tensorshape operation<T>::get_shape (void) const
{
	return data_->get_shape();
}

template <typename T>
const tensor<T>* operation<T>::get_eval (void) const
{
	return *data_;
}

template <typename T>
void operation<T>::temporary_eval (const iconnector<T>* target, tensor<T>*& out) const
{
	// base case
	if (this == target)
	{
		// return 1
		out = new tensor<T>(1);
		return;
	}
	// traverse towards target by looking at leaf sets
	std::vector<tensor<T>*> allocated;
	std::vector<const tensor<T>*> tens;
	this->access_dependency(
	[&tens, &allocated, target](const subject* s)
	{
		inode<T>* a = static_cast<inode<T>*>(s);
		if ((iconnector<T>* con = dynamic_cast<iconnector<T>*>(a)) &&
			(con->potential_descendent(target)))
		{
			tensor<T>* t;
			con->temporary_eval(target, *t);

			allocated.push_back(t);
			tens.push_back(t);
		}
		else
		{
			tens.push_back(*(a->get_eval()));
		}
	});

	out = new tensor<T>(this->get_shape());
	// out is the shape of the resulting shape
	Nf_(*out, tens);
	for (tensor<T>* at : allocated)
	{
		delete t;
	}
}

template <typename T>
void operation<T>::update (react::subject* arg)
{
	bool badstate = false;
	std::vector<const tensor<T>*> tens;
	std::vector<tensorshape> ts;
	this->access_dependency(
	[&badstate, &tens, &ts](const subject* sub)
	{
		if (inode<T>* a = dynamic_cast<inode<T>*>(sub))
		{
			tens.push_back(a->get_eval());
			ts.push_back(a->get_shape());
		}
		else
		{
			badstate = true;
		}
	});

	if (badstate)
	{
		// self destroy
		this->notify(UNSUBSCRIBE);
	}
	else
	{
		// forward pass
		if (data_ == nullptr)
		{
			// initialize data in the expected resulting shape
			data_ = std::make_unique<tensor<T> >(Nf_.shaper_(ts));
		}
		Nf_(*data_, tens);
		this->notify(UPDATE);
	}
}

template <typename T>
void operation<T>::commit_sudoku (void)
{
	if (onheap_)
	{
		delete this;
	}
}

template <typename T>
inode<T>* operation<T>::clone_impl (void) const
{
	return new operation<T>(*this);
}

template <typename T>
operation<T>::operation (const operation<T>& other) :
	immutable<T>(other)
{
	Nf_ = other.Nf_;
	data_ = other.data_->clone();
}

}

#endif
