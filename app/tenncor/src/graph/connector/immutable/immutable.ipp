//
//  immutable.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2017-02-28.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_IMMUTABLE_HPP

namespace nnet
{

template <typename T>
immutable<T>::~immutable (void) {}

template <typename T>
immutable<T>* immutable<T>::get (std::vector<inode<T>*> args,
	SHAPER shaper, transfer_func<T>* Nf,
	BACK_MAP<T> ginit, std::string name,
	inode<T>* ignore_jacobian)
{
	assert(false == args.empty());
	immutable<T>* imm = new immutable<T>(args, shaper, Nf, ginit, name);
	if (nullptr != ignore_jacobian)
	{
		std::unordered_set<ileaf<T>*> leaves = ignore_jacobian->get_leaves();
		for (ileaf<T>* leaf : leaves)
		{
			if (variable<T>* var = dynamic_cast<variable<T>*>(leaf))
			{
				imm->jacobians_.erase(var);
			}
		}
	}
	return imm;
}

template <typename T>
immutable<T>* immutable<T>::clone (void) const
{
	return static_cast<immutable<T>*>(this->clone_impl());
}

template <typename T>
immutable<T>* immutable<T>::move (void)
{
	return static_cast<immutable<T>*>(this->move_impl());
}

template <typename T>
immutable<T>& immutable<T>::operator = (const immutable<T>& other)
{
	if (this != &other)
	{
		base_immutable<T>::operator = (other);
		copy_helper(other);
	}
	return *this;
}

template <typename T>
immutable<T>& immutable<T>::operator = (immutable<T>&& other)
{
	if (this != &other)
	{
		base_immutable<T>::operator = (std::move(other));
		move_helper(std::move(other));
	}
	return *this;
}

template <typename T>
typename iconnector<T>::summary_series immutable<T>::summarize (void) const
{
	typename iconnector<T>::conn_summary summ(this->get_name(), shaper_, Nf_, ginit_);
	for (subject* sb : this->dependencies_)
	{
		summ.arg_ids_.push_back(static_cast<inode<T>*>(sb)->get_summaryid());
	}
	return { summ };
}

template <typename T>
immutable<T>::immutable (
	std::vector<inode<T>*> args,
	SHAPER shaper,
	transfer_func<T>* Nf,
	BACK_MAP<T> ginit, std::string label) :
base_immutable<T>(args, label),
shaper_(shaper),
Nf_(Nf),
ginit_(ginit) { this->update({}); }

template <typename T>
immutable<T>::immutable (const immutable<T>& other) :
	base_immutable<T>(other)
{
	copy_helper(other);
}

template <typename T>
immutable<T>::immutable (immutable<T>&& other) :
	base_immutable<T>(std::move(other))
{
	move_helper(std::move(other));
}

template <typename T>
inode<T>* immutable<T>::clone_impl (void) const
{
	return new immutable<T>(*this);
}

template <typename T>
inode<T>* immutable<T>::move_impl (void)
{
	return new immutable<T>(std::move(*this));
}

template <typename T>
void immutable<T>::forward_pass (void)
{
	std::vector<tensorshape> ts;
	std::vector<const tensor<T>*> tens;
	for (subject* sub : this->dependencies_)
	{
		const tensor<T>* arg = this->take_eval(static_cast<inode<T>*>(sub));
		if (arg)
		{
			assert(arg->is_alloc());
			ts.push_back(arg->get_shape());
		}
		else
		{
			ts.push_back(tensorshape());
		}
		tens.push_back(arg);
	}
	tensorshape s = shaper_(ts);
	if (nullptr == this->data_)
	{
		s.assert_is_fully_defined();
		this->data_ = new tensor<T>(s);
	}
	else if (s.is_fully_defined())
	{
		// if data_ is allocated, verify shape with data_
		if (this->data_->is_alloc())
		{
			tensorshape oshape = this->data_->get_shape();
			if (false == s.is_compatible_with(oshape))
			{
				std::stringstream ss;
				print_shape(s, ss);
				ss << " is incompatible with output shape ";
				print_shape(oshape, ss);
				throw std::runtime_error(ss.str());
			}
		}
		// otherwise allocate data_
		else
		{
			this->data_->allocate(s);
		}
	}
	if (temp_in_.empty())
	{
		temp_in_ = Nf_->prepare_args(s, tens);
	}
	(*Nf_)(this->data_, temp_in_);
}

template <typename T>
void immutable<T>::backward_pass (variable<T>* leaf)
{
	std::vector<std::pair<inode<T>*,inode<T>*> > deps;
	for (subject* s : this->dependencies_)
	{
		inode<T>* fn = static_cast<inode<T>*>(s);
		inode<T>* bn = this->take_gradient(fn, leaf);
		deps.push_back({fn, bn});
	}
	this->gcache_[leaf] = ginit_(deps);
}

template <typename T>
void immutable<T>::copy_helper (const immutable& other)
{
	ginit_ = other.ginit_;
	Nf_ = std::shared_ptr<transfer_func<T> >(other.Nf_->clone());
	temp_in_.clear();
}

template <typename T>
void immutable<T>::move_helper (immutable&& other)
{
	ginit_ = std::move(other.ginit_);
	Nf_ = std::move(other.Nf_);
	other.Nf_ = nullptr;
	temp_in_.clear();
	other.temp_in_.clear();
}

}

#endif
