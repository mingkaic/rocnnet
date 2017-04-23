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
	std::string name, bool ignore_jacobian)
{
	immutable<T>* imm = new immutable<T>(args, shaper, Nf, F, name);
	if (ignore_jacobian)
	{
		imm->jacobians_.list_.clear();
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
void immutable<T>::temporary_eval (const iconnector<T>* target, tensor<T>*& out) const
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
	for (subject* sub : this->dependencies_)
	{
		inode<T>* a = static_cast<inode<T>*>(sub);
		iconnector<T>* con = dynamic_cast<iconnector<T>*>(a);
		if (nullptr != con && con->potential_descendent(target))
		{
			tensor<T>* t = nullptr;
			con->temporary_eval(target, t);

			allocated.push_back(t);
			tens.push_back(t);
		}
		else
		{
			tens.push_back(a->get_eval());
		}
	};

	out = new tensor<T>(this->get_shape());
	// out is the shape of the resulting shape
	Nf_(*out, tens);
	for (tensor<T>* at : allocated)
	{
		delete at;
	}
}

template <typename T>
bool immutable<T>::good_status (void) const
{
	return data_ != nullptr;
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
		return zero.get();
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
const tensor<T>* immutable<T>::get_gradient (inode<T>* wrt)
{
	inode<T>* out;
	bool outtemp = false;
	iconnector<T>* conn = dynamic_cast<iconnector<T>*>(wrt);
	// check self
	if (wrt == this)
	{
		out = one.get();
	}
	// check cache
	else if (variable<T>* leaf = dynamic_cast<variable<T>*>(wrt))
	{
		out = get_leaf(leaf);
		// todo: move this down to accommodate non-leaf gradients
		// modify res with jacobian
		for (JTRANSFER<T> js : jacobians_.list_)
		{
			out = js(out, leaf);
		}
	}
	// check graph
	else if (conn && this->is_same_graph(conn))
	{
		// WARNING: this is one of the more expensive operations
		// evoke temporary call
		outtemp = true;

		tensor<T>* res;
		this->temporary_eval(conn, res);

		placeholder<T>* pl = new placeholder<T>(this->get_shape());
		*pl = *res; // move tensor value to placeholder temporary
		out = pl;
		delete res;
	}
	// is zero
	else
	{
		out = zero.get();
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
void immutable<T>::update (subject* /*arg*/)
{
	bool allgood = true;
	bool damaged = false;
	std::vector<const tensor<T>*> tens;
	std::vector<tensorshape> ts;
	for (auto it = this->dependencies_.begin(), et = this->dependencies_.end();
		it != et && allgood && !damaged; it++)
	{
		if (inode<T>* a = dynamic_cast<inode<T>*>(*it))
		{
			allgood = allgood && a->good_status();
			if (allgood)
			{
				tens.push_back(a->get_eval());
				ts.push_back(a->get_shape());
			}
		}
		else
		{
			damaged = true;
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
			data_ = std::make_unique<tensor<T> >(Nf_.shaper_(ts));
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
immutable<T>::immutable (
	std::vector<inode<T>*> args,
	SHAPER shaper, FORWARD_OP<T> forward,
	BACK_MAP<T> F, std::string label) :
iconnector<T>(args, label),
Nf_(shaper, forward),
ginit_(F)
{
	std::unordered_set<inode<T>*> deps;
	std::string jlabel = "";
	// todo: test for jacobian, and leaf transfer
	for (subject* sub : this->dependencies_)
	{
		inode<T>* arg = dynamic_cast<inode<T>*>(sub);
		if (deps.end() == deps.find(arg))
		{
			immutable<T>* imm = dynamic_cast<immutable<T>*>(arg);
			if (nullptr != imm && false == imm->jacobians_.list_.empty())
			{
				jacobians_ = imm->jacobians_; // copy over
				// test for jacobian conflict across branches
				assert(jlabel.empty() || jlabel == jacobians_.uid_);
				jlabel = jacobians_.uid_;
			}
			arg->get_leaves(gcache_);
			deps.emplace(arg);
		}
	}
	common();
	update(nullptr); // update data_ initially
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
	common();
	copy_helper(other);
}

template <typename T>
immutable<T>::immutable (immutable<T>&& other) :
	iconnector<T>(std::move(other)),
	Nf_(std::move(other.Nf_))
{
	common();
	move_helper(std::move(other));
}

template <typename T>
void immutable<T>::common (void)
{
	zero = std::unique_ptr<constant<T> >(constant<T>::get(0));
	one = std::unique_ptr<constant<T> >(constant<T>::get(1));
	zero->is_managed_ = true;
	one->is_managed_ = true;
}

template <typename T>
void immutable<T>::copy_helper (const immutable& other)
{
	data_ = std::unique_ptr<tensor<T> >(other.data_->clone());
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
