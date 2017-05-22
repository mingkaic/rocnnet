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
immutable<T>::~immutable (void) {}

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
	return data_.get();
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
		// initiate grad_
		std::vector<inode<T>*> deps;
		for (subject* sub : this->dependencies_)
		{
			deps.push_back(static_cast<inode<T>*>(sub));
		};
		gcache_[leaf] = ginit_(deps, leaf);
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
		JList& j = jacobians_[leaf];
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
		typename iconnector<T>::graph_node* info = nullptr;
		this->gid_->get_master(info);
		if (info->freeze_)
		{
			info->jobs_.push(this);
			return;
		}
	}
	bool allgood = true;
	bool damaged = false;
	std::vector<const tensor<T>*> tens;
	std::vector<tensorshape> ts;
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
				ts.push_back(a->get_shape());
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
		if (data_ == nullptr)
		{
			// initialize data in the expected resulting shape
			data_ = std::make_unique<tensor<T> >(Nf_.calc_shape(ts));
		}
		Nf_(*data_, tens);
		this->notify(UPDATE);
	}

	// graph id optimization (update master and remove unnecessary graph ids)
	typename iconnector<T>::graph_node* master = nullptr;
	this->gid_->get_master(master);
	if (this->gid_ != master)
	{
		master->replace(this->gid_); // gid_ = lhs
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
immutable<T>::immutable (
	std::vector<inode<T>*> args,
	SHAPER shaper, FORWARD_OP<T> forward,
	BACK_MAP<T> F, std::string label) :
iconnector<T>(args, label),
Nf_(shaper, forward),
ginit_(F)
{
	std::unordered_set<inode<T>*> deps;
	// todo: test for jacobian, and leaf transfer
	// if we have more than 1 jacobian, separate the operators for each branch
	for (subject* sub : this->dependencies_)
	{
		inode<T>* arg = static_cast<inode<T>*>(sub);
		// only perform following on unique dependent nodes:
		if (deps.end() == deps.find(arg))
		{
			if (immutable<T>* imm = dynamic_cast<immutable<T>*>(arg))
			{
				for (auto jpair : imm->jacobians_)
				{
					variable<T>* leaf = jpair.first;
					auto jit = jacobians_.find(leaf);
					// different jacobians originating from the same leaf cannot overlap
					JList& j = jpair.second;
					if (!j.list_.empty())
					{
						assert (jacobians_.end() == jit || jit->second.uid_ == j.uid_);
						jacobians_[leaf] = j;
					}
				}
			}
			arg->get_leaves(gcache_);
			deps.emplace(arg);
		}
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
void immutable<T>::copy_helper (const immutable& other)
{
	data_ = nullptr;
	if (other.data_)
	{
		data_ = std::unique_ptr<tensor<T> >(other.data_->clone());
	}
	ginit_ = other.ginit_;
}

template <typename T>
void immutable<T>::move_helper (immutable&& other)
{
	data_ = std::move(other.data_);
	other.data_ = nullptr;
	ginit_ = std::move(other.ginit_);
}

}

#endif
