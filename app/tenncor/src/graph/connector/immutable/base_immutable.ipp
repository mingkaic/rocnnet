//
//  base_immutable.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2017-06-26.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_BASE_IMMUTABLE_HPP

namespace nnet
{

template <typename T>
base_immutable<T>::~base_immutable (void) { if (data_) delete data_; }

template <typename T>
base_immutable<T>* base_immutable<T>::clone (void) const
{
	return static_cast<base_immutable<T>*>(this->clone_impl());
}

template <typename T>
base_immutable<T>* base_immutable<T>::move (void)
{
	return static_cast<base_immutable<T>*>(this->move_impl());
}

template <typename T>
base_immutable<T>& base_immutable<T>::operator = (const base_immutable<T>& other)
{
	if (this != &other)
	{
		iconnector<T>::operator = (other);
		copy_helper(other);
	}
	return *this;
}

template <typename T>
base_immutable<T>& base_immutable<T>::operator = (base_immutable<T>&& other)
{
	if (this != &other)
	{
		iconnector<T>::operator = (std::move(other));
		move_helper(std::move(other));
	}
	return *this;
}

template <typename T>
varptr<T> base_immutable<T>::derive (inode<T>* wrt)
{
	varptr<T> out;
	iconnector<T>* conn = dynamic_cast<iconnector<T>*>(wrt);
	// check self
	if (wrt == this)
	{
		out = constant<T>::get_shared_one();
	}
	// check cache
	else if (variable<T>* leaf = dynamic_cast<variable<T>*>(wrt))
	{
		out = this->get_gradient(leaf);
		// modify res with jacobians
		out = this->jacobian_call(out, leaf);
	}
	// check graph
	else if (conn && this->is_same_graph(conn))
	{
		// WARNING: this is one of the more expensive operations
		inode<T>* temp_out = nullptr;
		this->temporary_eval(conn, temp_out);
		out = temp_out;
		// todo: apply jacobian (and test)

	}
	// is zero
	else
	{
		out = constant<T>::get_shared_zero();
	}
	return out;
}

template <typename T>
void base_immutable<T>::temporary_eval (const iconnector<T>* target, inode<T>*& out) const
{
	constant<T>* base = nullptr;
	out = temp_eval_helper(target, base);
	if (iconnector<T>* outcon = dynamic_cast<iconnector<T>*>(out))
	{
		outcon->update(std::unordered_set<size_t>{});
	}
}

template <typename T>
tensorshape base_immutable<T>::get_shape (void) const
{
	if (this->g_man_) this->g_man_->update();
	if (nullptr == data_)
	{
		return tensorshape();
	}
	return get_eval()->get_shape();
}

template <typename T>
std::unordered_set<ileaf<T>*> base_immutable<T>::get_leaves (void) const
{
	std::unordered_set<ileaf<T>*> leaves;
	for (auto leaf : gcache_)
	{
		leaves.emplace(leaf.first);
	}
	return leaves;
}

template <typename T>
bool base_immutable<T>::good_status (void) const
{
	return data_ != nullptr && data_->is_alloc();
}

template <typename T>
bool base_immutable<T>::read_proto (const tenncor::tensor_proto&)
{
	// it doesn't really make sense to deserialize data_ here, since data serves as a cache...
	return false;
}

template <typename T>
void base_immutable<T>::update (std::unordered_set<size_t>)
{
	bool allgood = true;
	bool damaged = false;
	for (size_t i = 0, n_subs = this->dependencies_.size();
		i < n_subs && allgood && !damaged; i++)
	{
		if (inode<T>* a = dynamic_cast<inode<T>*>(this->dependencies_[i]))
		{
			allgood = a->good_status() && allgood;
		}
		else
		{
			damaged = true;
		}
	}

	if (damaged)
	{
		// self destroy
		this->notify(UNSUBSCRIBE);
	}
	else if (allgood)
	{
		assert(this->g_man_);
		if (this->g_man_->freeze_ || 1 < this->dependencies_.size())
		// n-aries are pull update
		{
			this->g_man_->add_update(this,
			[this]
			{
				forward_pass();
			});
		}
		else
		// unaries are push update
		{
			// forward pass
			forward_pass();
			this->notify(UPDATE);
		}
	}
}

template <typename T>
base_immutable<T>::base_immutable (std::vector<inode<T>*> args, std::string label) :
	iconnector<T>(args, label)
{
	std::unordered_set<ileaf<T>*> leafset;
	for (subject* sub : this->dependencies_)
	{
		std::unordered_set<ileaf<T>*> leef = static_cast<inode<T>*>(sub)->get_leaves();
		leafset.insert(leef.begin(), leef.end());
	}
	for (ileaf<T>* l : leafset)
	{
		gcache_[l] = nullptr;
	}
}

template <typename T>
base_immutable<T>::base_immutable (const base_immutable<T>& other) :
	iconnector<T>(other)
{
	copy_helper(other);
}

template <typename T>
base_immutable<T>::base_immutable (base_immutable<T>&& other) :
	iconnector<T>(std::move(other))
{
	move_helper(std::move(other));
}

template <typename T>
void base_immutable<T>::death_on_broken (void)
{
	delete this;
}

template <typename T>
const tensor<T>* base_immutable<T>::get_eval (void) const
{
	return data_;
}

template <typename T>
inode<T>* base_immutable<T>::get_gradient (variable<T>* leaf)
{
	auto it = gcache_.find(leaf);
	if (gcache_.end() == it)
	{
		return constant<T>::get_shared_zero();
	}
	if (nullptr == it->second)
	{
		backward_pass(leaf);
	}
	return gcache_[leaf];
}

template <typename T>
void base_immutable<T>::copy_helper (const base_immutable& other)
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
	gcache_ = other.gcache_;
}

template <typename T>
void base_immutable<T>::move_helper (base_immutable&& other)
{
	if (data_)
	{
		delete data_;
	}
	data_ = std::move(other.data_);
	other.data_ = nullptr;
	gcache_ = std::move(other.gcache_);
}

template <typename T>
inode<T>* base_immutable<T>::temp_eval_helper (const iconnector<T>* target, constant<T>*& base) const
{
	// base case
	if (this == target)
	{
		// return 1
		if (!base)
		{
			base = constant<T>::get(1);
		}
		return base;
	}
	// traverse towards target by comparing leaf sets
	std::vector<inode<T>*> args;
	for (subject* sub : this->dependencies_)
	{
		inode<T>* arg = static_cast<inode<T>*>(sub);
		base_immutable<T>* con = dynamic_cast<base_immutable<T>*>(arg);
		if (nullptr != con && con->potential_descendent(target))
		{
			args.push_back(con->temp_eval_helper(target, base));
		}
		else
		{
			args.push_back(arg);
		}
	}
	// create a new copy of this with out sharing base's life cycle
	base_immutable<T>* base_imm = this->arg_clone(args);
	base_imm->add_ondeath_dependent(base);
	return base_imm;
}

}

#endif