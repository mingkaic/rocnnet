//
//  constant.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_CONSTANT_HPP

namespace nnet
{

template <typename T>
constant<T> constant<T>::shared_zero(0);

template <typename T>
constant<T> constant<T>::shared_one(1);

template <typename T>
constant<T>* constant<T>::get_shared_zero (void)
{
	shared_zero.is_managed_ = true;
	return &shared_zero;
}

template <typename T>
constant<T>* constant<T>::get_shared_one (void)
{
	shared_one.is_managed_ = true;
	return &shared_one;
}

template <typename T>
constant<T>* constant<T>::get (T scalar)
{
	return new constant<T>(scalar);
}

template <typename T>
constant<T>* constant<T>::get (std::vector<T> raw, tensorshape shape)
{
	return new constant<T>(raw, shape);
}

template <typename T>
constant<T>* constant<T>::clone (void) const
{
	return static_cast<constant<T>*>(clone_impl());
}

template <typename T>
constant<T>* constant<T>::move (void)
{
	return static_cast<constant<T>*>(move_impl());
}

template <typename T>
varptr<T> constant<T>::get_gradient (inode<T>*)
{
	return constant<T>::get_shared_zero();
}

template <typename T>
void constant<T>::get_leaf (inode<T>*& out, variable<T>*)
{
	out = constant<T>::get_shared_zero();
}

template <typename T>
void constant<T>::get_leaves (
	typename inode<T>::GRAD_CACHE&) const {}

template <typename T>
constant<T>::constant (T scalar) :
	ileaf<T>(std::vector<size_t>{1},
		nnutils::formatter() << scalar)
{
	const_init<T> init(scalar);
	this->data_->allocate();
	init(this->data_);
	this->is_init_ = true;
}

template <typename T>
constant<T>::constant (std::vector<T> raw, tensorshape shape) :
	ileaf<T>(shape, raw.empty() ? "<empty>" :
		(nnutils::formatter() << raw.front() << ".." << raw.back()).str())
{
	size_t rawn = raw.size();
	typename ileaf<T>::assignment assigner;
	if (false == this->data_->is_alloc())
	{
		// loosely guess fails if n_elems/n_known > raw size
		// we ensure this will never happen by padding with zeros
		if (shape.n_known() > rawn)
		{
			size_t deficiency = shape.n_known() - rawn;
			raw.insert(raw.end(), deficiency, 0);
		}
		optional<tensorshape> propershape = this->data_->loosely_guess_shape(raw);
		assert((bool) propershape);
		this->data_->allocate(*propershape);
	}
	// we should also pad 0s for well defined shapes
	size_t n = this->data_->n_elems();
	if (n > rawn)
	{
		size_t deficiency = n - rawn;
		raw.insert(raw.end(), deficiency, 0);
	}
	assigner(this->data_, raw);
	this->is_init_ = true;
}

template <typename T>
void constant<T>::commit_sudoku_sub (void)
{
	if (false == is_managed_ && this->no_audience())
	{
		delete this;
	}
}

template <typename T>
inode<T>* constant<T>::clone_impl (void) const
{
	return nullptr;
}

template <typename T>
inode<T>* constant<T>::move_impl (void)
{
	return nullptr;
}

}

#endif