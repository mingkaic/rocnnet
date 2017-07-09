//
//  iconnector.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-01.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <queue>

#ifdef TENNCOR_ICONNECTOR_HPP

namespace nnet
{

template <typename T, typename N>
inline std::vector<iconnector<T>*> to_con (std::vector<N*> args)
{
	std::vector<iconnector<T>*> conns;
	for (N* a : args)
	{
		if (iconnector<T>* con = dynamic_cast<iconnector<T>*>(a))
		{
			conns.push_back(con);
		}
	}
	return conns;
}

template <typename T>
iconnector<T>::~iconnector (void)
{
	if (g_man_) g_man_->suicide(this);
}

template <typename T>
iconnector<T>* iconnector<T>::clone (void) const
{
	return static_cast<iconnector<T>*>(this->clone_impl());
}

template <typename T>
iconnector<T>* iconnector<T>::move (void)
{
	return static_cast<iconnector<T>*>(this->move_impl());
}

template <typename T>
iconnector<T>& iconnector<T>::operator = (const iconnector<T>& other)
{
	if (this != &other)
	{
		iobserver::operator = (other);
		inode<T>::operator = (other);
		jacobians_ = other.jacobians_;
		// this copies other's dependencies so, this and other share a graph
		g_man_->suicide(this);
		g_man_ = graph_manager::get(const_cast<iconnector<T>*>(&other), this);
	}
	return *this;
}

template <typename T>
iconnector<T>& iconnector<T>::operator = (iconnector<T>&& other)
{
	if (this != &other)
	{
		iobserver::operator = (std::move(other));
		inode<T>::operator = (std::move(other));
		jacobians_ = std::move(other.jacobians_);
		// this copies other's dependencies so, this and other share a graph
		g_man_->suicide(this);
		g_man_ = graph_manager::get(&other, this);
		other.g_man_->suicide(&other);
		other.g_man_ = nullptr;
	}
	return *this;
}

template <typename T>
std::string iconnector<T>::get_name (void) const
{
	std::string args = "";
	auto it = this->dependencies_.begin();
	auto et = this->dependencies_.end();
	const inode <T>* arg = dynamic_cast<const inode<T>*>(*it);
	while (args.empty() && nullptr == arg)
	{
		arg = dynamic_cast<const inode<T>*>(*++it);
	}
	if (arg)
	{
		args = arg->get_label();
		++it;
	}
	while (it != et)
	{
		if (nullptr != (arg = dynamic_cast<const inode<T>*>(*it))) {
			args += "," + arg->get_label();
		}
		it++;
	}
	return inode<T>::get_name() + "(" + args + ")";
}

template <typename T>
std::vector<inode<T>*> iconnector<T>::get_arguments (void) const
{
	std::vector<inode<T>*> node_args(this->dependencies_.size());
	std::transform(this->dependencies_.begin(), this->dependencies_.end(), node_args.begin(),
		[](subject* s) { return static_cast<inode<T>*>(s); });
	return node_args;
}

template <typename T>
size_t iconnector<T>::n_arguments (void) const
{
	return this->dependencies_.size();
}

template <typename T>
const tensor<T>* iconnector<T>::eval (void)
{
	if (this->g_man_) this->g_man_->update();
	return this->get_eval();
}

template <typename T>
bool iconnector<T>::is_same_graph (const iconnector<T>* other) const
{
	return g_man_ == other->g_man_;
}

template <typename T>
bool iconnector<T>::potential_descendent (const iconnector<T>* n) const
{
	// A is a descendent of B iff A's leaf set is a subset of B's leaf set (or vice versa)
	std::unordered_set<ileaf<T>*> mine = this->get_leaves();
	std::unordered_set<ileaf<T>*> their = n->get_leaves();

	if (mine.size() < their.size()) return false;
	for (ileaf<T>* t : their)
	{
		if (mine.end() == mine.find(t))
		{
			return false;
		}
	}
	return true;
}

template <typename T>
void iconnector<T>::set_jacobian (JTRANSFER<T> jac, std::vector<variable<T>*> leaves)
{
	for (variable<T>* l : leaves)
	{
		jacobians_[l].list_.push_front(jac);
	}
}

template <typename T>
void iconnector<T>::freeze_status (bool freeze)
{
	freeze_ = freeze;
}

template <typename T>
iconnector<T>::iconnector (std::vector<inode<T>*> dependencies, std::string label) :
	inode<T>(label),
	iobserver(std::vector<subject*>(dependencies.begin(), dependencies.end()))
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
			if (iconnector<T>* imm = dynamic_cast<iconnector<T>*>(arg))
			{
				for (auto jpair : imm->jacobians_)
				{
					variable<T>* leaf = jpair.first;
					auto jit = this->jacobians_.find(leaf);
					// different jacobians originating from the same leaf cannot overlap
					auto& j = jpair.second;
					if (false == j.list_.empty())
					{
						assert (this->jacobians_.end() == jit || jit->second.uid_ == j.uid_);
						this->jacobians_[leaf] = j;
					}
				}
			}
			deps.emplace(arg);
		}
	}
	update_graph(to_con<T, inode<T> >(dependencies));
}

template <typename T>
iconnector<T>::iconnector (const iconnector<T>& other) :
	inode<T>(other),
	iobserver(other),
	jacobians_(other.jacobians_)
{
	if (g_man_) g_man_->suicide(this);
	g_man_ = graph_manager::get(const_cast<iconnector<T>*>(&other), this);
}

template <typename T>
iconnector<T>::iconnector (iconnector<T>&& other) :
	inode<T>(std::move(other)),
	iobserver(std::move(other)),
	jacobians_(std::move(other.jacobians_))
{
	if (g_man_) g_man_->suicide(this);
	g_man_ = graph_manager::get(&other, this);
	other.g_man_->suicide(&other);
	other.g_man_ = nullptr;
}

template <typename T>
void iconnector<T>::update_graph (std::vector<iconnector<T>*> args)
{
	if (args.empty())
	{
		if (nullptr == g_man_)
		{
			graph_manager::get(this);
		}
		return;
	}
	g_man_ = graph_manager::get(args[0], this);
	for (size_t i = 1, n = args.size(); i < n; i++)
	{
		g_man_->consume(args[i]->g_man_);
	}
}

template <typename T>
struct small_leafset
{
	bool operator() (const iconnector<T>* c1, const iconnector<T>* c2) const
	{
		return c1->get_leaves().size() > c2->get_leaves().size();
	}
};

template <typename T>
struct iconnector<T>::graph_manager
{
	static graph_manager* get (iconnector<T>* source, iconnector<T>* user = nullptr)
	{
		assert(source);
		graph_manager*& gn = source->g_man_;
		if (nullptr == gn) gn = new graph_manager();
		if (nullptr == user) user = source;
		gn->users_.emplace(user);
		return gn;
	}

	graph_manager (const graph_manager&) = delete;

	graph_manager (graph_manager&&) = delete;

	void suicide (iconnector<T>* user)
	{
		users_.erase(user);
		if (users_.empty()) delete this;
	}

	void consume (graph_manager* other)
	{
		if (this == other) return;
		while (false == other->updates_.empty())
		{
			updates_.push(other->updates_.top());
			other->updates_.pop();
		}
		update_map_.insert(other->update_map_.begin(), other->update_map_.end());
		std::unordered_set<iconnector<T>*> otherusers = other->users_;
		for (iconnector<T>* ouser : otherusers)
		{
			other->suicide(ouser);
			ouser->g_man_ = this;
		}
		users_.insert(otherusers.begin(), otherusers.end());
	}

	void add_update (iconnector<T>* dependent, std::function<void(void)> update)
	{
		// assert dependent is in users_
		if (update_map_.end() == update_map_.find(dependent))
		{
			updates_.push(dependent);
			update_map_[dependent] = update;
		}
	}

	void update (void)
	{
		while (false == updates_.empty())
		{
			iconnector<T>* iconn = updates_.top();
			auto updater = update_map_[iconn];
			updates_.pop();
			update_map_.erase(iconn);
			updater();
			iconn->notify(UPDATE);
		}
	}

private:
	std::priority_queue<iconnector<T>*, std::vector<iconnector<T>*>, small_leafset<T> > updates_;

	std::unordered_map<iconnector<T>*,std::function<void(void)> > update_map_;

	std::unordered_set<iconnector<T>*> users_;
	
	graph_manager (void) {}
	
	~graph_manager (void) {}
};

}

#endif