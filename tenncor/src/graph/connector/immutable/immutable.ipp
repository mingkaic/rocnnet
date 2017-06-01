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
immutable<T>* immutable<T>::get (std::vector<inode<T>*> args,
	SHAPER shaper, FORWARD_OP<T> Nf, BACK_MAP<T> F,
	std::string name, inode<T>* ignore_jacobian)
{
	immutable<T>* imm = new immutable<T>(args, shaper, Nf, F, name);
	if (nullptr != ignore_jacobian)
	{
		typename inode<T>::GRAD_CACHE leaves;
		ignore_jacobian->get_leaves(leaves);
		for (auto leafpair : leaves)
		{
			imm->jacobians_.erase(leafpair.first);
		}
	}
	return imm;
}

template <typename T>
immutable<T>::~immutable (void)
{
	delete data_;
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
		iconnector<T>::operator = (other);
		copy_helper(other);
		Nf_ = other.Nf_;
	}
	return *this;
}

template <typename T>
immutable<T>& immutable<T>::operator = (immutable<T>&& other)
{
	if (this != &other)
	{
		iconnector<T>::operator = (std::move(other));
		move_helper(std::move(other));
		Nf_ = std::move(other.Nf_);
	}
	return *this;
}

template <typename T>
tensorshape immutable<T>::get_shape (void) const
{
	if (nullptr == data_)
	{
		return tensorshape();
	}
	return data_->get_shape();
}

template <typename T>
const tensor<T>* immutable<T>::get_eval (void) const
{
	if (nullptr == data_)
	{
		return nullptr;
	}
	return data_;
}

template <typename T>
void immutable<T>::temporary_eval (const iconnector<T>* target, inode<T>*& out) const
{
	// base case
	if (this == target)
	{
		// return 1
		out = constant<T>::get(1);
		return;
	}
	// traverse towards target by comparing leaf sets
	std::vector<inode<T>*> args;
	for (subject* sub : this->dependencies_)
	{
		iconnector<T>* con = dynamic_cast<iconnector<T>*>(sub);
		if (nullptr != con && con->potential_descendent(target))
		{
			inode<T>* tempout;
			con->temporary_eval(target, tempout);
			args.push_back(tempout);
		}
		else
		{
			args.push_back(static_cast<inode<T>*>(sub));
		}
	};

	out = new immutable<T>(args, *this);
}

template <typename T>
bool immutable<T>::good_status (void) const
{
	return data_ != nullptr && data_->is_alloc();
}

template <typename T>
void immutable<T>::get_leaves (
	typename inode<T>::GRAD_CACHE& leaves) const
{
	for (auto leaf : gcache_)
	{
		leaves[leaf.first] = nullptr;
	}
}

template <typename T>
inode<T>* immutable<T>::get_leaf (variable<T>* leaf)
{
	auto it = gcache_.find(leaf);
	if (gcache_.end() == it)
	{
		return get_shared_zero<T>();
	}
	if (nullptr == it->second)
	{
		std::vector<inode<T>*> deps;
		for (subject* sub : this->dependencies_)
		{
			deps.push_back(static_cast<inode<T>*>(sub));
		};
		backward_pass(deps, leaf);
	}
	return gcache_[leaf];
}

template <typename T>
inode<T>* immutable<T>::get_gradient (inode<T>* wrt)
{
	inode<T>* out = nullptr;
	iconnector<T>* conn = dynamic_cast<iconnector<T>*>(wrt);
	// check self
	if (wrt == this)
	{
		out = get_shared_one<T>();
	}
	// check cache
	else if (variable<T>* leaf = dynamic_cast<variable<T>*>(wrt))
	{
		out = get_leaf(leaf);
		// modify res with jacobian
		auto& j = this->jacobians_[leaf];
		for (auto it = j.list_.rbegin(), et = j.list_.rend(); it != et; it++)
		{
			out = (*it)(out, leaf);
		}
	}
	// check graph
	else if (conn && this->is_same_graph(conn))
	{
		// WARNING: this is one of the more expensive operations
		// evoke temporary call, out pollutes memory, but it will be removed eventually...
		// todo: implement top-down garabage cleanup
		this->temporary_eval(conn, out);
		// todo: apply jacobian (and test)

	}
	// is zero
	else
	{
		out = get_shared_zero<T>();
	}
	return out;
}

template <typename T>
void immutable<T>::update (subject* /*arg*/)
{
	{
		if (this->gid_->freeze_)
		{
			this->gid_->jobs_.push(this);
			return;
		}
	}
	bool allgood = true;
	bool damaged = false;
	std::vector<const tensor<T>*> tens;
	for (auto it = this->dependencies_.begin(), et = this->dependencies_.end();
		it != et && allgood && !damaged; it++)
	{
		inode<T>* a = dynamic_cast<inode<T>*>(*it);
		damaged = nullptr == a;
		if (!damaged)
		{
			allgood = allgood && a->good_status();
			if (allgood)
			{
				tens.push_back(a->get_eval());
			}
		}
	};

	if (damaged)
	{
		// self destroy
		this->notify(UNSUBSCRIBE);
	}
	else if (allgood)
	{
		// forward pass
		forward_pass(tens);
		this->notify(UPDATE);
	}
}

template <typename T>
bool immutable<T>::read_proto (const tenncor::tensor_proto&)
{
	// it doesn't really make sense to deserialize data_ here, since data serves as a cache...
	// todo: have an option to disable caching for performance boost
	return false;
}

template <typename T>
void immutable<T>::summarize (std::vector<typename iconnector<T>::conn_summary>& conn_list) const
{
	conn_list.push_back(typename iconnector<T>::conn_summary(this->get_name(), Nf_, ginit_, this->dependencies_.size()));
}

template <typename T>
immutable<T>::immutable (
	std::vector<inode<T>*> args,
	SHAPER shaper, FORWARD_OP<T> forward,
	BACK_MAP<T> F, std::string label) :
iconnector<T>(args, label),
Nf_(shaper, forward),
ginit_(F)
{
	for (subject* sub : this->dependencies_)
	{
		static_cast<inode<T>*>(sub)->get_leaves(gcache_);
	}
	update(nullptr); // update data_ initially
}

template <typename T>
immutable<T>::immutable (std::vector<inode<T>*> args, const immutable<T>& other) :
	immutable<T>(other)
{
	for (size_t i = 0, n = args.size(); i < n; i++)
	{
		this->replace_dependency(args[i], i);
	}
	update(nullptr);
}

template <typename T>
void immutable<T>::commit_sudoku (void)
{
	delete this;
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
immutable<T>::immutable (const immutable<T>& other) :
	iconnector<T>(other),
	Nf_(other.Nf_)
{
	copy_helper(other);
}

template <typename T>
immutable<T>::immutable (immutable<T>&& other) :
	iconnector<T>(std::move(other)),
	Nf_(std::move(other.Nf_))
{
	move_helper(std::move(other));
}

template <typename T>
void immutable<T>::forward_pass (std::vector<const tensor<T>*> tens)
{
	Nf_(data_, tens);
}

template <typename T>
void immutable<T>::backward_pass (std::vector<inode<T>*> deps, variable<T>* leaf)
{
	gcache_[leaf] = ginit_(deps, leaf);
}

template <typename T>
void immutable<T>::copy_helper (const immutable& other)
{
	if (data_)
	{
		delete data_;
		data_ = nullptr;
	}
	if (other.data_)
	{
		data_ = other.data_->clone();
	}
	ginit_ = other.ginit_;
	gcache_ = other.gcache_;
}

template <typename T>
void immutable<T>::move_helper (immutable&& other)
{
	if (data_)
	{
		delete data_;
	}
	data_ = std::move(other.data_);
	other.data_ = nullptr;
	ginit_ = std::move(other.ginit_);
	gcache_ = std::move(other.gcache_);
}

}

#endif
