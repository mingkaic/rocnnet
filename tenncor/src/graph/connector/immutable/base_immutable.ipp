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
const tensor<T>* base_immutable<T>::get_eval (void) const
{
	return data_;
}

template <typename T>
varptr<T> base_immutable<T>::get_gradient (inode<T>* wrt)
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
		varptr<T> leafout;
		get_leaf(leafout, leaf);
		// modify res with jacobian
		auto& j = this->jacobians_[leaf];
		for (auto it = j.list_.rbegin(), et = j.list_.rend(); it != et; it++)
		{
			leafout = (*it)(leafout, leaf);
		}
		out = leafout;
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
	eval_helper(target, out);
	if (iconnector<T>* outcon = dynamic_cast<iconnector<T>*>(out))
	{
		outcon->update({});
	}
}

template <typename T>
tensorshape base_immutable<T>::get_shape (void) const
{
	if (nullptr == data_)
	{
		return tensorshape();
	}
	return data_->get_shape();
}

template <typename T>
void base_immutable<T>::get_leaves (typename inode<T>::GRAD_CACHE& leaves) const
{
	for (auto leaf : gcache_)
	{
		leaves[leaf.first] = nullptr;
	}
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
void base_immutable<T>::update (std::vector<size_t> update_indices)
{
	if (this->gid_->freeze_)
	{
		for (size_t argidx : update_indices)
		{
			this->gid_->push(this, argidx);
		}
		return;
	}
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
		// forward pass
		forward_pass(update_indices);
		this->notify(UPDATE);
	}
}

template <typename T>
void base_immutable<T>::get_leaf (varptr<T>& out, variable<T>* leaf)
{
	auto it = gcache_.find(leaf);
	if (gcache_.end() == it)
	{
		out = constant<T>::get_shared_zero();
	}
	else
	{
		if (nullptr == it->second)
		{
			backward_pass(leaf);
		}
		out = gcache_[leaf];
	}
}

template <typename T>
base_immutable<T>::base_immutable (std::vector<inode<T>*> args, std::string label) :
	iconnector<T>(args, label)
{
	for (subject* sub : this->dependencies_)
	{
		static_cast<inode<T>*>(sub)->get_leaves(gcache_);
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
void base_immutable<T>::eval_helper (const iconnector<T>* target, inode<T>*& out) const
{
	// base case
	if (this == target)
	{
		// return 1
		out = constant<T>::get(1);
		return;
	}
	// traverse towards target by comparing leaf sets
	std::unordered_set<size_t> permidx;
	std::vector<subject*> args;
	for (size_t i = 0, n = this->dependencies_.size(); i < n; i++)
	{
		inode<T>* arg = static_cast<inode<T>*>(this->dependencies_[i]);
		base_immutable<T>* con = dynamic_cast<base_immutable<T>*>(arg);
		if (nullptr != con && con->potential_descendent(target))
		{
			inode<T>* tempout;
			con->eval_helper(target, tempout);
			args.push_back(tempout);
			if (nullptr == dynamic_cast<iconnector<T>*>(tempout)) permidx.emplace(i);
		}
		else
		{
			args.push_back(arg);
			permidx.emplace(i);
		}
	}
	out = merged_immutable<T>::get((base_immutable<T>*) this, args, permidx, true);
	for (size_t i = 0, n = args.size(); i < n; i++)
	{
		if (permidx.end() == permidx.find(i))
		{
			delete args[i];
		}
	}
}


template <typename T>
void solo_audience_merge (base_immutable<T>*& root)
{
	assert(root);
	// traverse from root to leaf:
	// merge observers with subjects with a sole observer
	// and delete merged observers and subjects
	std::unordered_set<base_immutable<T>*> visited = {root};
	std::list<base_immutable<T>*> node_q;
	node_q.push_back(root);
	while (false == node_q.empty())
	{
		base_immutable<T>* imm = node_q.front();
		assert(imm);
		node_q.pop_front();
		std::vector<inode<T>*> args = imm->get_arguments();
		std::unordered_set<size_t> ignore_args; // ignore leaves and nodes with multiple observers
		if (imm->mergible_)
		{
			for (size_t i = 0, n = args.size(); i < n; i++)
			{
				base_immutable<T>* a = dynamic_cast<base_immutable<T>*>(args[i]);
				if (nullptr == a || false == a->mergible_ || 1 != a->n_audience() || 0 == a->n_arguments()) // is leaf
				{
					ignore_args.emplace(i);
				}
			}
			if (ignore_args.size() < args.size()) // as long as there is at least one merged subject, merge
			{
				merged_immutable<T>* mnode = merged_immutable<T>::get(imm, ignore_args);
				node_q.push_front(mnode);
				// update root when necessary
				if (imm == root)
				{
					root = mnode;
				}
				// replace imm for parents
				mnode->steal_observers(imm);
				// delete merged observers
				delete imm;
			}
			// delete merged subjects, and enqueue non-merged nodes
			for (size_t i = 0, n = args.size(); i < n; i++)
			{
				base_immutable<T>* imarg = dynamic_cast<base_immutable<T>*>(args[i]);
				if (ignore_args.end() == ignore_args.find(i))
				{
					delete args[i];
				}
				else if (imarg && visited.end() == visited.find(imarg))
				{
					node_q.push_back(imarg);
					visited.emplace(imarg);
				}
			}
		}
	}
}


template <typename T>
struct merged_immutable<T>::temp_immutable : public base_immutable<T>
{
	temp_immutable (std::vector<inode<T>*> args,
		typename iconnector<T>::conn_summary s, varptr<T> gout) :
	base_immutable<T>(args, "temp_imm"),
	summ_(s), gout_(gout)
	{
		summ_.arg_ids_.clear();
		for (subject* s : this->dependencies_)
		{
			summ_.arg_ids_.push_back(static_cast<inode<T>*>(s)->get_summaryid());
		}
	}

	virtual ~temp_immutable (void) {}

	virtual std::string get_summaryid (void) const
	{
		return summ_.id_;
	}

	virtual void temporary_eval (const iconnector<T>*, inode<T>*& out) const
	{
		out = gout_;
	}

	virtual varptr<T> get_gradient (inode<T>*)
	{
		return gout_;
	}

	virtual typename iconnector<T>::summary_series summarize (void) const
	{
		return { summ_ };
	}

	virtual void get_leaf (varptr<T>& out, variable<T>*)
	{
		out = gout_;
	}

	virtual inode<T>* clone_impl (void) const
	{
		return new temp_immutable(*this);
	}

	virtual inode<T>* move_impl (void)
	{
		return new temp_immutable(std::move(*this));
	}

	virtual void forward_pass (std::vector<size_t>) {}

	virtual void backward_pass (variable<T>*) {}

	typename iconnector<T>::conn_summary summ_;

	varptr<T> gout_;
};


template <typename T>
merged_immutable<T>* merged_immutable<T>::get (base_immutable<T>* conn,
	std::unordered_set<size_t> ignore_indices, bool disabled_update)
{
	return new merged_immutable<T>(conn, ignore_indices, disabled_update);
}

template <typename T>
merged_immutable<T>* merged_immutable<T>::get (base_immutable<T>* conn,
	std::vector<subject*> args, std::unordered_set<size_t> ignore_indices, bool disabled_update)
{
	return new merged_immutable<T>(conn, args, ignore_indices, disabled_update);
}

template <typename T>
merged_immutable<T>* merged_immutable<T>::clone (void) const
{
	return static_cast<merged_immutable<T>*>(this->clone_impl());
}

template <typename T>
merged_immutable<T>* merged_immutable<T>::move (void)
{
	return static_cast<merged_immutable<T>*>(this->move_impl());
}

template <typename T>
std::string merged_immutable<T>::get_summaryid (void) const
{
	return summaries_.back().id_;
}

template <typename T>
typename iconnector<T>::summary_series merged_immutable<T>::summarize (void) const
{
	return summaries_;
}

template <typename T>
merged_immutable<T>::merged_immutable (base_immutable<T>* conn,
	std::unordered_set<size_t> ignore_indices, bool disabled_update) :
base_immutable<T>(*conn)
{
	assert(conn->mergible_);
	typename iconnector<T>::summary_series top_summary = conn->summarize();
	if (nullptr == dynamic_cast<merged_immutable<T>*>(conn))
	{
		this->set_label("merge_" + this->get_label());
	}

	init_helper(top_summary, this->dependencies_, ignore_indices);
	// no need to refresh gcache or jacobians since we've copied over conn's gcache and jacobian
	// which is the most correct with respect to the new arguments
	if (false == disabled_update) this->update({}); // update data_ initially
}

template <typename T>
merged_immutable<T>::merged_immutable (base_immutable<T>* conn,
	std::vector<subject*> args, std::unordered_set<size_t> ignore_indices, bool disabled_update) :
base_immutable<T>(*conn)
{
	assert(conn->mergible_);
	typename iconnector<T>::summary_series top_summary = conn->summarize();
	// dependencies change
	std::vector<inode<T>*> orig_dep = conn->get_arguments();
	size_t n_args = orig_dep.size();
	assert(n_args == args.size());
	std::unordered_map<std::string,std::string> id_map;
	for (size_t i = 0; i < n_args; i++)
	{
		id_map[orig_dep[i]->get_summaryid()] = static_cast<inode<T>*>(args[i])->get_summaryid();
	}
	// scan through top_summary to relink arg_ids to args
	for (auto& tsumm : top_summary)
	{
		for (std::string& aid : tsumm.arg_ids_)
		{
			auto idit = id_map.find(aid);
			if (id_map.end() != idit)
			{
				aid = idit->second;
			}
		}
	}

	if (nullptr == dynamic_cast<merged_immutable<T>*>(conn))
	{
		this->set_label("merge_" + this->get_label());
	}

	init_helper(top_summary, args, ignore_indices);
	// no need to refresh gcache or jacobians since we've copied over conn's gcache and jacobian
	// which is the most correct with respect to the new arguments
	if (false == disabled_update) this->update({}); // update data_ initially
}

template <typename T>
inode<T>* merged_immutable<T>::clone_impl (void) const
{
	return new merged_immutable<T>(*this);
}

template <typename T>
inode<T>* merged_immutable<T>::move_impl (void)
{
	return new merged_immutable<T>(std::move(*this));
}

template <typename T>
void merged_immutable<T>::forward_pass (std::vector<size_t>)
{
	std::unordered_map<std::string,tensorshape> shape_map;
	std::unordered_map<std::string,const tensor<T>*> dep_map;
	for (subject* s : this->dependencies_)
	{
		inode<T>* dep = static_cast<inode<T>*>(s);
		std::string dep_id = dep->get_summaryid();
		const tensor<T>* arg = dep->get_eval();
		if (arg)
		{
			assert(arg->is_alloc());
			shape_map[dep_id] = arg->get_shape();
		}
		else
		{
			shape_map[dep_id] = tensorshape();
		}
		dep_map[dep_id] = arg;
	}
	tensorshape final_shape = summary_traversal(shape_map,
	std::function<tensorshape(std::vector<tensorshape>,typename iconnector<T>::conn_summary&)>(
	[&shape_map](std::vector<tensorshape> args, typename iconnector<T>::conn_summary& s) -> tensorshape
	{
		tensorshape outs = s.Nf_->calc_shape(args);
		shape_map[s.id_] = outs;
		return outs;
	}));
	if (nullptr == this->data_)
	{
		final_shape.assert_is_fully_defined();
		this->data_ = new tensor<T>(final_shape);
	}
	else if (final_shape.is_fully_defined())
	{
		// if data_ is allocated, verify shape with data_
		if (this->data_->is_alloc())
		{
			tensorshape oshape = this->data_->get_shape();
			if (false == final_shape.is_compatible_with(oshape))
			{
				std::stringstream ss;
				print_shape(final_shape, ss);
				ss << " is incompatible with output shape ";
				print_shape(oshape, ss);
				throw std::runtime_error(ss.str());
			}
		}
		// otherwise allocate data_
		else
		{
			this->data_->allocate(final_shape);
		}
	}
	// populate temp_in_
	if (arg_ptrs_.empty())
	{
		for (auto summ : summaries_)
		{
			std::string summ_id = summ.id_;
			std::vector<const tensor<T>*> externals;
			std::vector<std::pair<T*,tensorshape> > internals;
			for (std::string a_id : summ.arg_ids_)
			{
				auto lit = dep_map.find(a_id);
				if (dep_map.end() != lit) // this summary consumes at least a leaf
				{
					externals.push_back(lit->second);
					internals.push_back({nullptr, tensorshape()});
				}
				else // this summary consumes a merged connector
				{
					externals.push_back(nullptr);
					auto raw_it = raw_intermediates_.find(a_id);
					assert(raw_intermediates_.end() != raw_it);
					internals.push_back({&(raw_it->second[0]), shape_map[a_id]});
				}
			}
			tensorshape summ_shape = shape_map[summ.id_];
			std::vector<const T*> from_externals = summ.Nf_->prepare_args(summ_shape, externals);
			std::vector<const T*> from_internals = summ.Nf_->prepare_args(summ_shape, internals);
			if (from_externals.size() == from_internals.size())
			{
				arg_ptrs_[summ_id] = std::vector<const T*>(std::max(from_internals.size(), from_externals.size()));
				// OR addresses (fill in nullptrs)
				std::transform(from_externals.begin(), from_externals.end(), from_internals.begin(), arg_ptrs_[summ_id].begin(),
				[](const T* lhs, const T* rhs)
				{
					return nullptr == lhs ? rhs : lhs;
				});
			}
			else if (0 == from_externals.size())
			{
				arg_ptrs_[summ_id] = from_internals;
			}
			else if (0 == from_internals.size())
			{
				arg_ptrs_[summ_id] = from_externals;
			}
			else
			{
				throw std::logic_error(nnutils::formatter() << "merged_immutable internal preparation has size "
					<< from_internals.size() << " while argument preparation has size " << from_externals.size());
			}
			raw_intermediates_[summ_id] = std::vector<T>(summ_shape.n_elems());
		}
	}
	// perform aggregation from arg_ptrs_ to raw_intermediates_
	for (size_t i = 0, n = summaries_.size()-1; i < n; i++)
	{
		auto& summ = summaries_[i];
		auto it = arg_ptrs_.find(summ.id_);
		assert(arg_ptrs_.end() != it);
		std::vector<const T*> s_input = it->second;
		// aggregate from s_input to raw_intermediate_
		(*summ.Nf_)(raw_intermediates_[summ.id_], s_input);
	}
	auto& finalsumm = summaries_.back();
	auto it = arg_ptrs_.find(finalsumm.id_);
	assert(arg_ptrs_.end() != it);
	std::vector<const T*> s_input = it->second;
	(*finalsumm.Nf_)(this->data_, s_input);
}

template <typename T>
void merged_immutable<T>::backward_pass (variable<T>* leaf)
{
	std::unordered_map<std::string,inode<T>*> dep_map;
	for (subject* sub : this->dependencies_)
	{
		inode<T>* d = static_cast<inode<T>*>(sub);
		dep_map[d->get_summaryid()] = d;
	}
	std::vector<temp_immutable*> temps;
	inode<T>* out = summary_traversal(dep_map,
	std::function<inode<T>*(std::vector<inode<T>*>,typename iconnector<T>::conn_summary&)>(
	[leaf,&temps](std::vector<inode<T>*> args, typename iconnector<T>::conn_summary& summ)
	{
		temp_immutable* temp_out = new temp_immutable(args, summ, summ.ginit_(args, leaf));
		temps.push_back(temp_out);
		return temp_out;
	}));
	base_immutable<T>* imm = static_cast<base_immutable<T>*>(out->get_gradient(leaf).get());
	for (temp_immutable* t : temps)
	{
		t->gout_ = nullptr;
	}
	delete out;
	solo_audience_merge(imm);
	this->gcache_[leaf] = imm;
}

template <typename T>
void merged_immutable<T>::init_helper (typename iconnector<T>::summary_series top_summary,
	std::vector<subject*> args, std::unordered_set<size_t> ignore_indices)
{
	// extract subject's arguments, and summaries
	std::vector<subject*> fresh_deps;
	for (size_t i = 0, n = args.size(); i < n; i++)
	{
		subject* sub = args[i];
		iconnector<T>* arg = dynamic_cast<iconnector<T>*>(sub);
		if (ignore_indices.end() == ignore_indices.find(i) && arg)
		{
			// update summaries
			typename iconnector<T>::summary_series argsums = arg->summarize();
			summaries_.insert(summaries_.end(), argsums.begin(), argsums.end());

			// add argument in order
			std::vector<inode<T>*> aa = arg->get_arguments();
			fresh_deps.insert(fresh_deps.end(), aa.begin(), aa.end());
		}
		else
		{
			// add argument in order
			fresh_deps.push_back(sub);
		}
	}
	summaries_.insert(summaries_.end(), top_summary.begin(), top_summary.end());

	// replace current dependencies
	{
		size_t oldndep = this->dependencies_.size();
		size_t freshndep = fresh_deps.size();
		assert(freshndep >= oldndep);
		for (size_t i = oldndep; i < freshndep; i++)
		{
			this->add_dependency(fresh_deps[i]);
		}
		for (size_t i = 0; i < oldndep; i++)
		{
			this->replace_dependency(fresh_deps[i], i);
		}
	}
}

template <typename T>
template <typename U>
U merged_immutable<T>::summary_traversal (std::unordered_map<std::string,U> arg_map,
	std::function<U(std::vector<U>,typename iconnector<T>::conn_summary&)> op)
{
	// summaries occur in their dependent order
	for (auto& s : summaries_)
	{
		// search for queued dependencies
		std::vector<U> s_input;
		for (std::string d_id : s.arg_ids_)
		{
			auto it = arg_map.find(d_id);
			if (arg_map.end() != it) s_input.push_back(it->second);
			else s_input.push_back(U());
		}
		// consume input
		U s_output = op(s_input, s);
		// record output
		arg_map[s.id_] = s_output;
	}
	return arg_map[summaries_.back().id_];
}

}

#endif