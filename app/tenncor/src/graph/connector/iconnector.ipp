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
		copy_helper(other);
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
		move_helper(std::move(other));
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
		if (nullptr != (arg = dynamic_cast<const inode<T>*>(*it)))
		{
			args += "," + arg->get_label();
		}
		it++;
	}
	return inode<T>::get_name() + "(" + args + ")";
}

template <typename T>
size_t iconnector<T>::get_depth (void) const
{
	return depth_;
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
	if (this->g_man_ && false == this->g_man_->freeze_)
	{
		this->g_man_->update();
	}
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
void iconnector<T>::set_jacobian_front (JTRANSFER<T> jac, std::vector<variable<T>*> leaves)
{
	for (variable<T>* l : leaves)
	{
		jacobians_[l].list_.push_front({jac, this});
	}
}

template <typename T>
void iconnector<T>::set_jacobian_back (JTRANSFER<T> jac, std::vector<variable<T>*> leaves)
{
	for (variable<T>* l : leaves)
	{
		jacobians_[l].list_.push_back({jac, this});
	}
}

template <typename T>
void iconnector<T>::freeze_status (bool freeze)
{
	assert(this->g_man_);
	if (freeze)
	{
		this->g_man_->update();
	}
	this->g_man_->freeze_ = freeze;
}

template <typename T>
iconnector<T>::iconnector (std::vector<inode<T>*> dependencies, std::string label) :
	inode<T>(label),
	iobserver(std::vector<subject*>(dependencies.begin(), dependencies.end()))
{
	size_t n = dependencies.size();
	if (n > 0)
	{
		std::vector<size_t> depths(n, 0);
		std::transform(dependencies.begin(), dependencies.end(), depths.begin(),
		[](inode<T>* n)
		{
			return n->get_depth();
		});
		depth_ = *(std::max_element(depths.begin(), depths.end())) + 1;
	}

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
					auto j = jpair.second;
					if (false == j.list_.empty())
					{
						auto jit = this->jacobians_.find(leaf);
						if (this->jacobians_.end() == jit)
						{
							this->jacobians_[leaf] = j;
							this->jacobians_[leaf].terminal_ = false;
						}
						else if (j.uid_ != jit->second.uid_)
						{
							this->jacobians_[leaf].terminal_ = true; // terminate
							this->jacobians_[leaf].list_.clear();
						}
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
	iobserver(other)
{
	copy_helper(other);
}

template <typename T>
iconnector<T>::iconnector (iconnector<T>&& other) :
	inode<T>(std::move(other)),
	iobserver(std::move(other))
{
	move_helper(std::move(other));
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
varptr<T> iconnector<T>::jacobian_call (varptr<T> out, variable<T>* leaf) const
{
	auto jpair = this->jacobians_.find(leaf);
	if (this->jacobians_.end() != jpair)
	{
		auto& jlist = jpair->second.list_;
		for (auto it = jlist.rbegin(), et = jlist.rend(); it != et; it++)
		{
			const JTRANSFER<T>& jt = it->first;
			// get the node where jacobian originate from
			const inode<T>* orig = it->second;
			// get origin arguments and its gradients
			std::vector<inode<T>*> args = orig->get_arguments();
			std::vector<inode<T>*> grads(args.size(), nullptr);
			std::transform(args.begin(), args.end(), grads.begin(),
			[this, leaf](inode<T>* arg)
			{
				return this->take_gradient(arg, leaf);
			});
			// operate on out using args and grad
			out = jt(out, args, grads);
		}
	}
	return out;
}

template <typename T>
void iconnector<T>::copy_helper (const iconnector<T>& other)
{
	jacobians_ = other.jacobians_;
	jacobian_correction(&other);
	// this copies other's dependencies so, this and other share a graph
	if (g_man_) g_man_->suicide(this);
	g_man_ = graph_manager::get(const_cast<iconnector<T>*>(&other), this);
}

template <typename T>
void iconnector<T>::move_helper (iconnector<T>&& other)
{
	jacobians_ = std::move(other.jacobians_);
	jacobian_correction(&other);
	// this copies other's dependencies so, this and other share a graph
	if (g_man_)
	{
		g_man_->suicide(this);
	}
	g_man_ = graph_manager::get(&other, this);
	if (other.g_man_)
	{
		other.g_man_->suicide(&other);
		other.g_man_ = nullptr;
	}
}

template <typename T>
void iconnector<T>::jacobian_correction (const inode<T>* other)
{
	// todo: move this down to immutable,
	// since if mutable, parent can have existing jacobian_ with references to other
	// assert this node has no parent (true when copying immutables)

	// check other's jacobians leafset for references to other and set to this
	for (auto& jpair : jacobians_)
	{
		std::list<std::pair<JTRANSFER<T>,inode<T>*> >& js = jpair.second.list_;
		if (js.back().second == other)
		{
			js.back().second = this;
		}
	}
}

template <typename T>
struct iconnector<T>::JList
{
	JList (void) : uid_(nnutils::uuid(this)) {}

	std::string uid_;
	std::list<std::pair<JTRANSFER<T>, inode<T>*> > list_;
	bool terminal_ = false;
};

template <typename T>
struct small_leafset
{
	bool operator() (const iconnector<T>* c1, const iconnector<T>* c2) const
	{
		return c1->get_depth() > c2->get_depth();
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
		// todo: add multithreading
		while (false == updates_.empty())
		{
			iconnector<T>* iconn = updates_.top();
			auto updater = update_map_[iconn];
			updates_.pop();
			updater();
			iconn->notify(UPDATE);
		}
		update_map_.clear();
	}

	bool freeze_ = false;

private:
	std::priority_queue<iconnector<T>*,std::vector<iconnector<T>*>,small_leafset<T> > updates_;

	std::unordered_map<iconnector<T>*,std::function<void(void)> > update_map_;

	std::unordered_set<iconnector<T>*> users_;
	
	graph_manager (void) {}
	
	~graph_manager (void) {}
};

}

#endif