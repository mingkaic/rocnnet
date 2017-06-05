//
//  leaf.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_VARIABLE_HPP

namespace nnet
{

template <typename T>
variable<T>::variable (T scalar, std::string name) :
	ivariable<T>(std::vector<size_t>{1},
		new const_init<T>(scalar), name)
{
	initialize();
}

template <typename T>
variable<T>::variable (const tensorshape& shape, std::string name) :
	ivariable<T>(shape, nullptr, name) {}

template <typename T>
variable<T>::variable (const tensorshape& shape,
	const initializer<T>& init, std::string name) :
ivariable<T>(shape, init.clone(), name) {}

template <typename T>
variable<T>* variable<T>::clone (void) const
{
	return static_cast<variable<T>*>(clone_impl());
}

template <typename T>
variable<T>* variable<T>::move (void)
{
	return static_cast<variable<T>*>(move_impl());
}

template <typename T>
void variable<T>::set_initializer (const initializer<T>& init)
{
	if (this->init_)
	{
		delete this->init_;
	}
	this->init_ = init.clone();
}

template <typename T>
tensor<T>& variable<T>::initialize (void)
{
	assert(nullptr != this->init_);
	// if not alloc, attempt to allocate, throw if fail
	if (false == this->data_->is_alloc() &&
		false == this->data_->allocate())
	{
		throw std::runtime_error(this->get_label() + " data is not allocated");
	}
	(*this->init_)(this->data_);
	this->is_init_ = true;
	this->notify(UPDATE);
	return *this->data_;
}

template <typename T>
tensor<T>& variable<T>::initialize (tensorshape shape)
{
	assert(this->init_ != nullptr);
	if (false == this->data_->allocate(shape))
	{
		std::stringstream ss;
		ss << "shape ";
		print_shape(shape, ss);
		ss << " failed to allocate " << this->get_label();
		throw std::runtime_error(ss.str());
	}
	(*this->init_)(this->data_);
	this->is_init_ = true;
	this->notify(UPDATE);
	return *this->data_;
}

template <typename T>
void variable<T>::get_leaf (inode<T>*& out, variable<T>* leaf)
{
	if (this == leaf)
	{
		out = constant<T>::get_shared_one();
	}
	else
	{
		out = constant<T>::get_shared_zero();
	}
}

template <typename T>
void variable<T>::get_leaves (
	typename inode<T>::GRAD_CACHE& leaves) const
{
	leaves.emplace(const_cast<variable<T>*>(this), nullptr);
}

template <typename T>
variable_updater<T> variable<T>::assign (inode<T>* input) const
{
	return [this, input]()
	{
		tensor<T>* outputt = this->data_;
		transfer_func<T> assign(
		[outputt](std::vector<tensorshape>)
		{
			return outputt->get_shape();
		},
		[](T* dest, const tensorshape& shape, std::vector<const T*>& srcs, std::vector<tensorshape>& inshapes)
		{
			size_t ns = shape.n_elems();
			tensorshape& ins = inshapes.at(0);
			assert((ins.is_fully_defined() && ns == ins.n_elems()) ||
				   (!ins.is_fully_defined() && 0 == ns % ins.n_known()));
			assert(nullptr != srcs[0]);
			for (size_t i = 0; i < ns; i++)
			{
				dest[i] = srcs[0][i];
			}
		});
		const tensor<T>* inputt = input->get_eval();
		assign(outputt, {inputt});
	};
}

template <typename T>
variable_updater<T> variable<T>::assign_add (inode<T>* input) const
{
	return [this, input]()
	{
		tensor<T>* outputt = this->data_;
		transfer_func<T> assign(
		[outputt](std::vector<tensorshape>)
		{
			return outputt->get_shape();
		},
		[](T* dest, const tensorshape& shape, std::vector<const T*>& srcs, std::vector<tensorshape>& inshapes)
		{
			size_t ns = shape.n_elems();
			tensorshape& ins = inshapes.at(0);
			assert((ins.is_fully_defined() && ns == ins.n_elems()) ||
				   (!ins.is_fully_defined() && 0 == ns % ins.n_known()));
			assert(nullptr != srcs[0]);
			for (size_t i = 0; i < ns; i++)
			{
				dest[i] += srcs[0][i];
			}
		});
		const tensor<T>* inputt = input->get_eval();
		assign(outputt, {inputt});
	};
}

template <typename T>
variable_updater<T> variable<T>::assign_sub (inode<T>* input) const
{
	return [this, input]()
	{
		tensor<T>* outputt = this->data_;
		transfer_func<T> assign(
		[outputt](std::vector<tensorshape>)
		{
			return outputt->get_shape();
		},
		[](T* dest, const tensorshape& shape, std::vector<const T*>& srcs, std::vector<tensorshape>& inshapes)
		{
			size_t ns = shape.n_elems();
			tensorshape& ins = inshapes.at(0);
			assert((ins.is_fully_defined() && ns == ins.n_elems()) ||
				   (!ins.is_fully_defined() && 0 == ns % ins.n_known()));
			assert(nullptr != srcs[0]);
			for (size_t i = 0; i < ns; i++)
			{
				dest[i] -= srcs[0][i];
			}
		});
		const tensor<T>* inputt = input->get_eval();
		assign(outputt, {inputt});
	};
}

template <typename T>
inode<T>* variable<T>::clone_impl (void) const
{
	return new variable(*this);
}

template <typename T>
inode<T>* variable<T>::move_impl (void)
{
	return new variable(std::move(*this));
}

}

#endif
