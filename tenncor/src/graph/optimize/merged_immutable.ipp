//
// Created by Mingkai Chen on 2017-05-23.
//

#ifdef TENNCOR_MERGED_IMMUTABLE_HPP

namespace nnet
{

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
	size_t startidx = conn_list.size()-1;
	conn_list.insert(conn_list.end(), summaries_.begin(), summaries_.end());
	immutable<T>::summarize(conn_list);
	size_t arg_idx = 0;
	for (size_t top_didx : top_deps_)
	{
		conn_list[startidx + top_didx].dependents_.push_back({con_id, arg_idx++});
	}
}

template <typename T>
merged_immutable<T>::merged_immutable (immutable<T>* conn) :
	immutable<T>(*conn) // start by copying head connector in its entirety
{
	std::vector<subject*> temp_dep;
	std::string conn_id = conn->get_name();
	std::vector<subject*> args = conn->get_subjects();
	for (subject* sub : args)
	{
		if (iconnector<T>* arg = dynamic_cast<iconnector<T>*>(sub))
		{
			std::vector<subject*> aa = arg->get_subjects();
			temp_dep.insert(temp_dep.end(), aa.begin(), aa.end());
			arg->summarize(summaries_);
			top_deps_.push_back(summaries_.size()-1);
		}
		else
		{
			temp_dep.push_back(sub);
		}
	}

	// replace current dependencies with temp_dep
	this->dep_replace(temp_dep);
	// no need to refresh gcache or jacobians since we've copied over conn's gcache and jacobian
	// which is the most correct with respect to the new arguments
	this->update(nullptr); // update data_ initially
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
	std::unordered_map<std::string, std::vector<const tensor<T>*> > intermediates;
	size_t arg_idx = 0;
	std::vector<const tensor<T>*> headinput;
	std::vector<tensor<T>*> intermediate_tensors;
	// summaries occur in their dependent order
	for (auto& s : summaries_)
	{
		// search for queued dependencies
		auto it = intermediates.find(s.id_);
		std::vector<const tensor<T>*> input;
		if (intermediates.end() == it)
		{
			input.insert(input.end(), s.ndeps_, nullptr);
		}
		else
		{
			input = it->second;
		}
		// fill non-dependencies from arguments in order
		for (size_t i = 0, n = input.size(); i < n; i++)
		{
			if (nullptr == input[i])
			{
				input[i] = tens[arg_idx++];
			}
		}
		// consume input
		tensor<T>* output = nullptr;
		s.Nf_(output, input);
		intermediate_tensors.push_back(output);
		// cache output
		if (s.dependents_.empty())
		{
			headinput.push_back(output);
		}
		else
		{
			for (auto dep : s.dependents_)
			{
				intermediates[dep.first][dep.second] = output;
			}
		}
	}
	immutable<T>::forward_pass(headinput);
	for (tensor<T>* itens : intermediate_tensors)
		delete itens;
}

template <typename T>
void  merged_immutable<T>::backward_pass (std::vector<inode<T>*> deps, variable<T>* leaf)
{
	std::unordered_map<std::string, std::vector<inode<T>*> > intermediates;
	size_t arg_idx = 0;
	std::vector<inode<T>*> headinput;
	// summaries occur in their dependent order
	for (auto& s : summaries_)
	{
		// search for queued dependencies
		auto it = intermediates.find(s.id_);
		std::vector<inode<T>*> input;
		if (intermediates.end() == it)
		{
			input.insert(input.end(), s.ndeps_, nullptr);
		}
		else
		{
			input = it->second;
		}
		// fill non-dependencies from arguments in order
		for (size_t i = 0, n = input.size(); i < n; i++)
		{
			if (nullptr == input[i])
			{
				input[i] = deps[arg_idx++];
			}
		}
		// consume input
		inode<T>* output = s.ginit_(input, leaf);
		// cache output
		if (s.dependents_.empty())
		{
			headinput.push_back(output);
		}
		else
		{
			for (auto dep : s.dependents_)
			{
				intermediates[dep.first][dep.second] = output;
			}
		}
	}
	immutable<T>::backward_pass(headinput, leaf);
}

}

#endif
