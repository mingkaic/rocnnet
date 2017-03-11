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
immutable<T>* immutable<T>::clone (void) const
{
	return static_cast<immutable<T>*>(clone_impl());
}

template <typename T>
immutable<T>::immutable (immutable<T>&& other) :
	iconnector<T>(other),
	ginit_(std::move(other.ginit_)) {}

template <typename T>
immutable<T>& immutable<T>::operator = (const immutable<T>& other)
{
	if (this != &other)
	{
		iconnector<T>::operator = (other);
		ginit_ = other.ginit_;
	}
	return *this;
}

template <typename T>
immutable<T>& immutable<T>::operator = (immutable<T>&& other)
{
	if (this != &other)
	{
		iconnector<T>::operator = (other);
		ginit_ = std::move(other.ginit_);
	}
	return *this;
}

template <typename T>
const tensor<T>* immutable<T>::get_gradient (inode<T>* wrt) const
{
	inode<T>* out;
	bool outtemp = false;
	iconnector<T>* conn = dynamic_cast<iconnector<T>*>(wrt);
	// check self
	if (wrt == this)
	{
		out = &this->one;
	}
		// check cache
	else if (variable<T>* leaf = dynamic_cast<variable<T>*>(wrt))
	{
		auto it = gcache_.find(leaf);
		if (gcache_.end() == it)
		{
			throw std::exception();
		}
		out = it->second;
	}
		// check graph
	else if (conn && is_same_graph(conn))
	{
		// WARNING: this is one of the more expensive operations
		// evoke temporary call
		outtemp = true;

		tensor<T>* res;
		this->temporary_eval(conn, *res);

		out = new placeholder<T>(this->get_shape());
		*out = *res; // move tensor value to placeholder temporary
		delete res;
	}
		// is zero
	else
	{
		out = &this->zero;
	}
	// modify res with jacobian
	for (JTRANSFER<T> js : jacobians_)
	{
		out = js(out, wrt);
	}
	const tensor<T>* res = out->get_eval();
	if (outtemp)
	{
		res = res->clone();
		delete out;
	}
	return res;
}

template <typename T>
bool immutable<T>::good_status (void) const
{
	return true;
}

template <typename T>
immutable<T>::immutable (std::vector<inode<T>*> args,
	BACK_MAP<T> F, std::string name) :
iconnector<T>(args, name),
ginit_(F)
{
	for (inode<T>* arg : this->dependencies_)
	{
		if (immutable<T>* a = static_cast<immutable<T>*>(arg))
		{
			if (false == a->jacobians_.empty())
			{
				assert(jacobians_.empty()); // jacobian conflict across branches
				jacobians_ = a->jacobians_; // copy over
			}
		}
		arg->get_leaves(gcache_);
	}
}

template <typename T>
immutable<T>::immutable (const immutable<T>& other) :
	iconnector<T>(other),
	ginit_(other.ginit_) {}

template <typename T>
inode<T>* immutable<T>::clone_impl (void) const
{
	return new immutable(*this);
}

template <typename T>
void immutable<T>::get_leaves (
	typename inode<T>::GRAD_CACHE& leaves) const
{
	for (auto leaf : gcache_)
	{
		leaves.emplace({leaf->first(), nullptr});
	}
}

template <typename T>
inode<T>* immutable<T>::get_leaf (variable<T>* leaf) const
{
	if (nullptr == gcache_[leaf])
	{
		// initiate grad_
		std::vector<inode<T>*> deps;
		this->access_dependency(
		[&deps](const react::subject* sub)
		{
			deps.push_back(static_cast<const inode<T>*>(sub));
		});
		gcache_[leaf] = ginit_(deps);
	}
	return gcache_[leaf];
}

}

#endif
