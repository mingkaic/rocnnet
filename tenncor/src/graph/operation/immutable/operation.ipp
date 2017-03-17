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
operation<T>* operation<T>::get (std::vector<inode<T>*> args,
	SHAPER shaper, FORWARD_OP<T> Nf, BACK_MAP F, std::string name)
{
	return new operation<T>(args, shaper, Nf, F, name);
}

template <typename T>
operation<T>* operation<T>::get (operation<T>&& other)
{
	return new operation<T>(std::move(other));
}

template <typename T>
operation<T>* operation<T>::clone (void) const
{
	return static_cast<operation<T>*>(clone_impl());
}

template <typename T>
operation<T>* operation<T>::move (void)
{
	return static_cast<operation<T>*>(move_impl());
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
	for (const subject* sub : this->dependencies_)
	{
		inode<T>* a = static_cast<inode<T>*>(sub);
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
	};

	out = new tensor<T>(this->get_shape());
	// out is the shape of the resulting shape
	Nf_(*out, tens);
	for (tensor<T>* at : allocated)
	{
		delete t;
	}
}

template <typename T>
void operation<T>::update (subject* arg)
{
	bool badstate = false;
	std::vector<const tensor<T>*> tens;
	std::vector<tensorshape> ts;
	for (const subject* sub : this->dependencies_)
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
	};

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

	graph_node* master = nullptr;
	gid_->get_master(master);
	if (gid_ != master)
	{
		master->replace(gid_); // gid_ = lhs
	}
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
void operation<T>::commit_sudoku (void)
{
	delete this;
}

template <typename T>
inode<T>* operation<T>::clone_impl (void) const
{
	return new operation<T>(*this);
}

template <typename T>
inode<T>* operation<T>::move_impl (void)
{
	return new operation<T>(std::move(*this));
}

template <typename T>
operation<T>::operation (const operation<T>& other) :
	immutable<T>(other)
{
	Nf_ = other.Nf_;
	data_ = other.data_->clone();
}

template <typename T>
operation<T>::operation (operation<T>&& other) :
	immutable<T>(other)
{
	Nf_ = std::move(other.Nf_);
	data_ = std::move(other.data_);
}

}

#endif
