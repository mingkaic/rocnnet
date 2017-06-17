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
	transfer_func<T>* Nf,
	BACK_MAP<T> ginit,
	std::string name, inode<T>* ignore_jacobian)
{
	immutable<T>* imm = new immutable<T>(args, Nf, ginit, name);
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
	if (Nf_) delete Nf_;
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
	std::unordered_set<size_t> permidx;
	std::vector<inode<T>*> args;
	for (size_t i = 0, n = this->n_subjects(); i < n; i++)
	{
		inode<T>* arg = static_cast<inode<T>*>(this->dependencies_[i]);
		iconnector<T>* con = dynamic_cast<iconnector<T>*>(arg);
		if (nullptr != con && con->potential_descendent(target))
		{
			inode<T>* tempout;
			con->temporary_eval(target, tempout);
			args.push_back(tempout);
			if (nullptr == dynamic_cast<iconnector<T>*>(tempout)) permidx.emplace(i);
		}
		else
		{
			args.push_back(arg);
			permidx.emplace(i);
		}
	};

	immutable<T>* tempthis = new immutable(args, *this);
	out = merged_immutable<T>::get(tempthis, permidx);
	delete tempthis;
	for (size_t i = 0, n = args.size(); i < n; i++)
	{
		if (permidx.end() == permidx.find(i))
		{
			delete args[i];
		}
	}
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
void immutable<T>::get_leaf (varptr<T>& out, variable<T>* leaf)
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
			std::vector<inode<T>*> deps;
			for (subject* sub : this->dependencies_)
			{
				deps.push_back(static_cast<inode<T>*>(sub));
			};
			backward_pass(deps, leaf);
		}
		out = gcache_[leaf];
	}
}

template <typename T>
varptr<T> immutable<T>::get_gradient (inode<T>* wrt)
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
		// evoke temporary call, out pollutes memory, but it will be removed eventually...
		// todo: implement top-down garabage cleanup
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
void immutable<T>::update (std::vector<size_t> update_indices)
{
	if (this->gid_->freeze_)
	{
		for (size_t argidx : update_indices)
		{
			this->gid_->push(this, argidx);
		}
		return;
	}
	size_t n_subs = this->n_subjects();
	bool allgood = true;
	bool damaged = false;
	std::vector<const tensor<T>*> tens;
	for (size_t i = 0; i < n_subs && allgood && !damaged; i++)
	{
		if (inode<T>* a = dynamic_cast<inode<T>*>(this->dependencies_[i]))
		{
			if (a->good_status())
			{
				tens.push_back(a->get_eval());
			}
			else
			{
				allgood = false;
			}
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
		forward_pass(tens, update_indices);
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
std::vector<typename iconnector<T>::conn_summary> immutable<T>::summarize (void) const
{
	return { typename iconnector<T>::conn_summary(this->get_name(), Nf_, ginit_, this->dependencies_.size()) };
}

template <typename T>
immutable<T>::immutable (
	std::vector<inode<T>*> args,
	transfer_func<T>* Nf,
	BACK_MAP<T> ginit, std::string label) :
iconnector<T>(args, label),
Nf_(Nf),
ginit_(ginit)
{
	for (subject* sub : this->dependencies_)
	{
		static_cast<inode<T>*>(sub)->get_leaves(gcache_);
	}
	update({}); // update data_ initially
}

template <typename T>
immutable<T>::immutable (std::vector<inode<T>*> args, const immutable<T>& other) :
	immutable<T>(other)
{
	gcache_.clear();
	for (size_t i = 0, n = args.size(); i < n; i++)
	{
		args[i]->get_leaves(gcache_);
		this->replace_dependency(args[i], i);
	}
	update({});
}

template <typename T>
void immutable<T>::death_on_broken (void)
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
	iconnector<T>(other)
{
	copy_helper(other);
}

template <typename T>
immutable<T>::immutable (immutable<T>&& other) :
	iconnector<T>(std::move(other))
{
	move_helper(std::move(other));
}

template <typename T>
void immutable<T>::forward_pass (std::vector<const tensor<T>*> tens, std::vector<size_t> update_indices)
{
	std::vector<const tensor<T>*> intens;
	if (nullptr == data_ || update_indices.empty())
	{
		intens = tens;
	}
	else
	{
		intens.insert(intens.end(), tens.size(), nullptr);
		for (size_t uidx : update_indices)
		{
			intens[uidx] = tens[uidx];
		}
	}
	(*Nf_)(data_, intens);
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
	Nf_ = other.Nf_->clone();
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
	Nf_ = std::move(other.Nf_);
	other.Nf_ = nullptr;
}

// MERGE_IMMUTABLE

template <typename T>
void solo_merge (immutable<T>*& root)
{
	// traverse from root to leaf: merging nodes if it has one argument
	std::list<immutable<T>*> subs;
	std::unordered_set<immutable<T>*> sset;
	subs.push_back(root);
	while (false == subs.empty())
	{
		immutable<T>* ob = subs.front();
		subs.pop_front();
		sset.erase(ob);
		if (ob)
		{
			std::vector<subject*> args = ob->get_subjects();
			std::vector<immutable<T>*> imms(args.size());
			std::transform(args.begin(), args.end(), imms.begin(),
				[](subject* s) { return dynamic_cast<immutable<T>*>(s); });
			size_t nleafs = 0;
			// one to n between parent and child: merge sole parents
			if (ob->mergible_ &&
				std::all_of(imms.begin(), imms.end(),
				[&nleafs](immutable<T>* imm)
				{
					bool is_leaf = nullptr == imm;
					if (is_leaf) nleafs++;
					return is_leaf || 1 == imm->n_audience();
				}) && nleafs < args.size())
			// if the number of leaf nodes in arg == arg.size then arg can't be merged
			{
				merged_immutable<T>* mnode = merged_immutable<T>::get(ob);
				subs.push_front(mnode);
				sset.emplace(mnode);
				if (ob == root)
				{
					root = mnode;
				}
				std::vector<subject*> msubs = mnode->get_subjects();
				std::unordered_set<subject*> msubset(msubs.begin(), msubs.end());
				for (immutable<T>* im : imms)
				{
					if (im && msubset.end() == msubset.find(im) && im->mergible_)
					{
						msubset.insert(im);
						delete im;
					}
				}
			}
			else
			{
				for (immutable<T>* u : imms)
				{
					if (sset.end() == sset.find(u))
					{
						subs.push_back(u);
						sset.emplace(u);
					}
				}
			}
		}
	}
}

template <typename T>
merged_immutable<T>* merged_immutable<T>::get (immutable<T>* conn, bool destructive)
{
	if (destructive) return new merged_immutable<T>(conn);
	return new merged_immutable<T>(conn, {});
}

template <typename T>
merged_immutable<T>* merged_immutable<T>::get (immutable<T>* conn, std::unordered_set<size_t> ignore_indices)
{
	return new merged_immutable<T>(conn, ignore_indices);
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
std::vector<typename iconnector<T>::conn_summary> merged_immutable<T>::summarize (void) const
{
	std::vector<typename iconnector<T>::conn_summary> out = summaries_;
	std::string con_id = this->get_name();
	// label head dependencies
	size_t real_ndeps = 0;
	for (size_t i = 0, m = summaries_.size(); i < m; i++)
	{
		auto& conninfo = out[i];
		auto it = conninfo.dependents_.find("");
		if (conninfo.dependents_.end() != it)
		{
			conninfo.dependents_[con_id] = it->second;
			conninfo.dependents_.erase(it);
			real_ndeps++;
		}
	}
	// append head summary
	std::vector<typename iconnector<T>::conn_summary> sums = immutable<T>::summarize();
	assert(false == sums.empty());
	sums[0].ndeps_ = real_ndeps;
	out.push_back(sums[0]);
	return out;
}

template <typename T>
merged_immutable<T>::merged_immutable (immutable<T>* conn) :
	immutable<T>(std::move(*conn)) // start by copying head connector in its entirety
{
	assert(conn->mergible_);
	std::vector<subject*> fresh_deps;
	if (merged_immutable<T>* merg = dynamic_cast<merged_immutable<T>*>(conn))
	{
		fresh_deps = summary_merge(merg->sub_mapper_);
		summaries_.insert(summaries_.end(),
			merg->summaries_.begin(),
			merg->summaries_.end());
		merg->summaries_.clear();
		merg->sub_mapper_.clear();
	}
	else
	{
		this->set_label("merge_" + this->get_label());
		fresh_deps = summary_merge({});
	}

	// reset cache content
	for (auto& gpair : this->gcache_)
	{
		// todo: account for when conn already have backprop graphs. cascade destroy or integrate (move)
		gpair.second = nullptr;
	}

	// replace current dependencies
	this->dep_replace(fresh_deps);
	// no need to refresh gcache or jacobians since we've copied over conn's gcache and jacobian
	// which is the most correct with respect to the new arguments
	this->update({}); // update data_ initially
	
	delete conn;
}

template <typename T>
merged_immutable<T>::merged_immutable (immutable<T>* conn, std::unordered_set<size_t> ignore_indices) :
	immutable<T>(*conn)
{
	assert(conn->mergible_);
	std::vector<subject*> fresh_deps;
	if (merged_immutable<T>* merg = dynamic_cast<merged_immutable<T>*>(conn))
	{
		fresh_deps = summary_merge(merg->sub_mapper_, ignore_indices);
		summaries_.insert(summaries_.end(),
			merg->summaries_.begin(),
			merg->summaries_.end());
	}
	else
	{
		this->set_label("merge_" + this->get_label());
		fresh_deps = summary_merge({}, ignore_indices);
	}

	// reset cache content
	for (auto& gpair : this->gcache_)
	{
		gpair.second = nullptr;
	}

	// replace current dependencies
	this->dep_replace(fresh_deps);
	// no need to refresh gcache or jacobians since we've copied over conn's gcache and jacobian
	// which is the most correct with respect to the new arguments
	this->update({}); // update data_ initially
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
void  merged_immutable<T>::forward_pass (std::vector<const tensor<T>*> tens, std::vector<size_t> update_indices)
{
	std::unordered_map<std::string,std::vector<const tensor<T>*> > inputs;
	for (size_t i = 0, n = tens.size(); i < n; i++)
	{
		auto& subinfo = sub_mapper_[i];
		const tensor<T>* ten = tens[i];
		std::vector<const tensor<T>*>& invec = inputs[subinfo.first];
		if (invec.size() < subinfo.second + 1)
		{
			invec.insert(invec.end(), subinfo.second - invec.size() + 1, nullptr);
		}
		invec[subinfo.second] = ten;
	}

	std::vector<tensor<T>*> intermediate_tensors;
	// summaries occur in their dependent order
	for (auto& s : summaries_)
	{
		// search for queued dependencies
		auto it = inputs.find(s.id_);
		// dependencies should be filled in order, so inputs should contain the inputs for s
		assert(inputs.end() != it);
		std::vector<const tensor<T>*> input = it->second;
		// consume input
		tensor<T>* output = nullptr;
		(*s.Nf_)(output, input);
		intermediate_tensors.push_back(output);
		// cache output
		for (auto dep : s.dependents_)
		{
			std::vector<const tensor<T>*>& invec = inputs[dep.first];
			for (size_t argidx : dep.second)
			{
				if (invec.size() < argidx + 1)
				{
					invec.insert(invec.end(), argidx - invec.size() + 1, nullptr);
				}
				invec[argidx] = output;
			}
		}
	}
	immutable<T>::forward_pass(inputs[""], update_indices);
	for (tensor<T>* itens : intermediate_tensors)
		delete itens;
}

template <typename T>
void merged_immutable<T>::backward_pass (std::vector<inode<T>*> deps, variable<T>* leaf)
{
	std::unordered_map<std::string,std::vector<inode<T>*> > inputs;
	for (size_t i = 0, n = deps.size(); i < n; i++)
	{
		if (inode<T>* n = dynamic_cast<inode<T>*>(deps[i]))
		{
			auto& subinfo = sub_mapper_[i];
			std::vector<inode<T>*>& invec = inputs[subinfo.first];
			if (invec.size() < subinfo.second + 1)
			{
				invec.insert(invec.end(), subinfo.second - invec.size() + 1, nullptr);
			}
			invec[subinfo.second] = n;
		}
	}

	std::vector<temp_immutable*> intermediate_outputs;
	// summaries occur in their dependent order
	for (auto& s : summaries_)
	{
		// search for queued dependencies
		auto it = inputs.find(s.id_);
		// dependencies should be filled in order, so inputs should contain the inputs for s
		assert(inputs.end() != it);
		std::vector<inode<T>*> input = it->second;
		// consume input
		temp_immutable* temp_output = new temp_immutable(input, s, leaf, s.ginit_(input, leaf));
		intermediate_outputs.push_back(temp_output);
		// cache output
		for (auto dep : s.dependents_)
		{
			std::vector<inode<T>*>& invec = inputs[dep.first];
			for (size_t argidx : dep.second)
			{
				if (invec.size() < argidx + 1)
				{
					invec.insert(invec.end(), argidx - invec.size() + 1, nullptr);
				}
				invec[argidx] = temp_output;
			}
		}
	}
	immutable<T>::backward_pass(inputs[""], leaf);
	for (temp_immutable* temps : intermediate_outputs)
	{
		temps->clear(leaf);
	}
	intermediate_outputs.clear(); // clear to avoid dangling ptrs
	if (immutable<T>* imm = dynamic_cast<immutable<T>*>(this->gcache_[leaf].get()))
	{
		solo_merge(imm);
		this->gcache_[leaf] = imm;
	}
}

template <typename T>
std::vector<subject*> merged_immutable<T>::summary_merge (
	std::vector<std::pair<std::string,size_t> > othersubmapper, std::unordered_set<size_t> ignore_argidx)
{
	std::vector<subject*> new_args;
	using temp_updater = std::function<void(size_t)>;
	temp_updater merg_sum_updater;
	temp_updater sub_map_updater;
	if (othersubmapper.empty())
	{
		merg_sum_updater = [this](size_t i) { summaries_.back().dependents_[""].push_back(i); };
		sub_map_updater = [this](size_t i) { sub_mapper_.push_back({"", i}); };
	}
	else
	{
		merg_sum_updater = [this, &othersubmapper](size_t i)
		{
			auto subinfo = othersubmapper[i];
			summaries_.back().dependents_[subinfo.first].push_back(subinfo.second);
		};
		sub_map_updater = [this, &othersubmapper](size_t i)
		{ sub_mapper_.push_back(othersubmapper[i]); };
	}
	// extract new arguments, summaries, submapper from another merge_immutable
	for (size_t i = 0, n = this->n_subjects(); i < n; i++)
	{
		subject* sub = this->dependencies_[i];
		iconnector<T>* arg = dynamic_cast<iconnector<T>*>(sub);
		if (ignore_argidx.end() == ignore_argidx.find(i) && arg)
		{
			std::vector<typename iconnector<T>::conn_summary> argsums = arg->summarize();
			summaries_.insert(summaries_.end(), argsums.begin(), argsums.end());
			merg_sum_updater(i);

			std::string newid = summaries_.back().id_;
			std::vector<subject*> aa = arg->get_subjects();
			new_args.insert(new_args.end(), aa.begin(), aa.end());
			if (merged_immutable<T>* merg = dynamic_cast<merged_immutable<T>*>(arg))
			{
				for (size_t j = 0, m = aa.size(); j < m; j++)
				{
					auto& infopair = merg->sub_mapper_[j];
					std::string id = infopair.first;
					if (id.empty()) id = newid;
					sub_mapper_.push_back({id, infopair.second});
				}
			}
			else
			{
				for (size_t j = 0, m = aa.size(); j < m; j++)
				{
					sub_mapper_.push_back({newid, j});
				}
			}
		}
		else
		{
			new_args.push_back(sub);
			sub_map_updater(i);
		}
	}
	return new_args;
}

}

#endif
