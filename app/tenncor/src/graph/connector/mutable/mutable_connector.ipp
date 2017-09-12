//
//  mutable_connector.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-27.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef mutable_connect_hpp

namespace nnet
{

template <typename T>
void mutable_connector<T>::connect (void)
{
	if (valid_args() && nullptr == ic_)
	{
		iconnector<T>* con = dynamic_cast<iconnector<T>*>(op_maker_(arg_buffers_));
		delete ic_;
		ic_ = con;
		this->add_dependency(con);
		ic_->set_death((void**) &ic_); // ic_ resets to nullptr when deleted
	}
}

template <typename T>
void mutable_connector<T>::disconnect (void)
{
	if (nullptr != ic_)
	{
		// severe our dependency on ic_
		// to prevent this from getting destroyed
		this->kill_dependencies();
		delete ic_;
	}
}

template <typename T>
mutable_connector<T>::mutable_connector (MAKE_CONNECT<T> maker, size_t nargs) :
	iconnector<T>(std::vector<inode<T>*>{}, "mutable_connector"),
	op_maker_(maker), arg_buffers_(nargs, nullptr) {}

template <typename T>
mutable_connector<T>::mutable_connector (const mutable_connector<T>& other) :
	iconnector<T>(other), op_maker_(other.op_maker_),
	arg_buffers_(other.arg_buffers_) {}

template <typename T>
mutable_connector<T>* mutable_connector<T>::get (MAKE_CONNECT<T> maker, size_t nargs)
{
	return new mutable_connector<T>(maker, nargs);
}

template <typename T>
mutable_connector<T>::~mutable_connector (void)
{
	if (nullptr != ic_)
	{
		delete ic_;
	}
}

template <typename T>
mutable_connector<T>* mutable_connector<T>::clone (void)
{
	return new mutable_connector<T>(*this);
}

template <typename T>
mutable_connector<T>& mutable_connector<T>::operator = (const mutable_connector<T>& other)
{
	if (&other != this)
	{
		iconnector<T>::operator = (other);
		op_maker_ = other.op_maker_;
		arg_buffers_ = other.arg_buffers_;
	}
	return *this;
}

template <typename T>
tensorshape mutable_connector<T>::get_shape(void)
{
	if (nullptr == ic_)
	{
		return tensorshape();
	}
	return ic_->get_shape();
}

template <typename T>
tensor<T>* mutable_connector<T>::get_eval(void)
{
	if (nullptr == ic_)
	{
		return nullptr;
	}
	return ic_->get_eval();
}

template <typename T>
bindable_toggle<T>* mutable_connector<T>::derive(void)
{
	if (nullptr == ic_)
	{
		return nullptr;
	}
	return ic_->derive();
}

template <typename T>
functor<T>* mutable_connector<T>::get_jacobian (void)
{
	if (nullptr == ic_)
	{
		return nullptr;
	}
	return ic_->get_jacobian();
}

template <typename T>
void mutable_connector<T>::update (caller_info info, update_message msg)
{
	if (update_message::REMOVE_ARG == msg.cmd_)
	{
		disconnect();
	}
	else
	{
		this->notify(msg);
	}
}

template <typename T>
bool mutable_connector<T>::add_arg (inode<T>* var, size_t idx)
{
	bool replace = nullptr != arg_buffers_[idx];
	arg_buffers_[idx] = var;
	if (replace)
	{
		disconnect();
	}
	connect();
	return replace;
}

template <typename T>
bool mutable_connector<T>::remove_arg (size_t idx)
{
	if (nullptr != arg_buffers_[idx])
	{
		disconnect();
		arg_buffers_[idx] = nullptr;
		return true;
	}
	return false;
}

template <typename T>
bool mutable_connector<T>::valid_args (void)
{
	for (varptr<T> arg : arg_buffers_)
	{
		if (nullptr == arg.get())
		{
			return false;
		}
	}
	return true;
}

template <typename T>
size_t mutable_connector<T>::nargs (void) const
{
	return arg_buffers_.size();
}

template <typename T>
void mutable_connector<T>::get_args (std::vector<inode<T>*>& args) const
{
	args.clear();
	for (varptr<T> a : arg_buffers_)
	{
		args.push_back(a.get());
	}
}

}

#endif
