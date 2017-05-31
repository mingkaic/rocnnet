//
// Created by Mingkai Chen on 2017-05-23.
//

#ifdef TENNCOR_MERGED_IMMUTABLE_HPP

namespace nnet
{

template <typename T>
void solo_merge (immutable<T>*& root)
{
	// traverse from root to leaf: merging nodes if it has one argument
	std::list<immutable<T>*> subs;
	subs.push_back(root);
	while (false == subs.empty())
	{
		immutable<T>* ob = subs.front();
		subs.pop_front();
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
			{
				merged_immutable<T>* mnode = merged_immutable<T>::get(ob);
				subs.push_front(mnode);
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
				std::unordered_set<immutable<T>*> uniqueset(imms.begin(), imms.end());
				subs.insert(subs.end(), uniqueset.begin(), uniqueset.end());
			}
		}
	}
}

template <typename T>
merged_immutable<T>* merged_immutable<T>::get (immutable<T>* conn)
{
	return new merged_immutable<T>(conn);
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
void merged_immutable<T>::summarize (std::vector<typename iconnector<T>::conn_summary>& conn_list) const
{
	std::string con_id = this->get_name();
	// append listed summaries
	size_t n = summaries_.size();
	conn_list.insert(conn_list.end(), summaries_.begin(), summaries_.end());
	// label head dependencies
	for (size_t i = n, m = summaries_.size(); i < m; i++)
	{
		auto& conninfo = conn_list[i];
		auto it = conninfo.dependents_.find("");
		if (conninfo.dependents_.end() == it)
		{
			conninfo.dependents_[con_id] = it->second;
			conninfo.dependents_.erase(it);
		}
	}
	// append head summary
	immutable<T>::summarize(conn_list);
}

template <typename T>
merged_immutable<T>::merged_immutable (immutable<T>* conn) :
	immutable<T>(std::move(*conn)) // start by copying head connector in its entirety
{
	assert(conn->mergible_);
	this->set_label("merge_" + this->get_label());
	std::vector<subject*> old_deps;

	if (merged_immutable<T>* imm = dynamic_cast<merged_immutable<T>*>(conn))
	{
		// connect new arguments with previous summaries
		for (size_t i = 0, n = this->n_subjects(); i < n; i++)
		{
			subject* sub = this->dependencies_[i];
			if (iconnector<T>* arg = dynamic_cast<iconnector<T>*>(sub))
			{
				arg->summarize(summaries_);
				std::vector<subject*> aa = arg->get_subjects();
				old_deps.insert(old_deps.end(), aa.begin(), aa.end());
				for (size_t j = 0, m = aa.size(); j < m; j++)
				{
					sub_mapper_.push_back({arg->get_name(), j});
				}
				auto subinfo = imm->sub_mapper_[i];
				summaries_.back().dependents_[subinfo.first].push_back(subinfo.second);
			}
			else
			{
				old_deps.push_back(sub);
				sub_mapper_.push_back(imm->sub_mapper_[i]);
			}
		}
		summaries_.insert(summaries_.end(),
			imm->summaries_.begin(),
			imm->summaries_.end());
		imm->summaries_.clear();
		imm->sub_mapper_.clear();
	}
	else
	{
		// create summaries
		for (size_t i = 0, n = this->n_subjects(); i < n; i++)
		{
			subject* sub = this->dependencies_[i];
			if (iconnector<T>* arg = dynamic_cast<iconnector<T>*>(sub))
			{
				arg->summarize(summaries_);
				std::vector<subject*> aa = arg->get_subjects();
				old_deps.insert(old_deps.end(), aa.begin(), aa.end());
				for (size_t j = 0, m = aa.size(); j < m; j++)
				{
					sub_mapper_.push_back({arg->get_name(), j});
				}
				summaries_.back().dependents_[""].push_back(i);
			}
			else
			{
				old_deps.push_back(sub);
				sub_mapper_.push_back({"", i});
			}
		}
	}

	// reset cache content
	for (auto& gpair : this->gcache_)
	{
//		if (immutable<T>* imm = dynamic_cast<immutable<T>*>(gpair.second))
//		{
//			solo_merge(imm);
//			gpair.second = imm;
//		}
		// todo: account for when conn already have backprop graphs. cascade destroy or integrate (move)
		gpair.second = nullptr;
	}

	// replace current dependencies
	this->dep_replace(old_deps);
	// no need to refresh gcache or jacobians since we've copied over conn's gcache and jacobian
	// which is the most correct with respect to the new arguments
	this->update(nullptr); // update data_ initially
	delete conn;
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
void  merged_immutable<T>::forward_pass (std::vector<const tensor<T>*> tens)
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
		s.Nf_(output, input);
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
	immutable<T>::forward_pass(inputs[""]);
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

	std::vector<inode<T>*> intermediate_outputs;
	// summaries occur in their dependent order
	for (auto& s : summaries_)
	{
		// search for queued dependencies
		auto it = inputs.find(s.id_);
		// dependencies should be filled in order, so inputs should contain the inputs for s
		assert(inputs.end() != it);
		std::vector<inode<T>*> input = it->second;
		// consume input
		inode<T>* goutput = s.ginit_(input, leaf);
		merged_immutable<T>* temp_output = new merged_immutable<T>(input, s, leaf, goutput);
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
	if (immutable<T>* imm = dynamic_cast<immutable<T>*>(this->gcache_[leaf]))
	{
		solo_merge(imm);
		this->gcache_[leaf] = imm;
	}
}

}

#endif
