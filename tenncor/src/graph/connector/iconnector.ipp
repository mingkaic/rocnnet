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
	if (gid_) gid_->suicide(this);
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
		gid_->suicide(this);
		gid_ = graph_node::get(const_cast<iconnector<T>*>(&other), this);
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
		gid_->suicide(this);
		gid_ = graph_node::get(&other, this);
		other.gid_->suicide(&other);
		other.gid_ = nullptr;
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
bool iconnector<T>::is_same_graph (const iconnector<T>* other) const
{
	return gid_ == other->gid_;
}

template <typename T>
bool iconnector<T>::potential_descendent (const iconnector<T>* n) const
{
	// is a descendent iff n's leaf set is a subset of this leaf set
	typename inode<T>::GRAD_CACHE mine;
	typename inode<T>::GRAD_CACHE their;
	this->get_leaves(mine);
	n->get_leaves(their);

	if (mine.size() < their.size()) return false;
	for (auto t : their)
	{
		variable<T>* leaf = t.first;
		if (mine.end() == mine.find(leaf))
		{
			return false;
		}
	}
	return true;
}

template <typename T>
void iconnector<T>::update_status (bool freeze)
{
	gid_->freeze_ = freeze;
	if (freeze) return;
	while (false == gid_->empty())
	{
		gid_->job_execute();
	}
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
	if (gid_) gid_->suicide(this);
	gid_ = graph_node::get(const_cast<iconnector<T>*>(&other), this);
}

template <typename T>
iconnector<T>::iconnector (iconnector<T>&& other) :
	inode<T>(std::move(other)),
	iobserver(std::move(other)),
	jacobians_(std::move(other.jacobians_))
{
	if (gid_) gid_->suicide(this);
	gid_ = graph_node::get(&other, this);
	other.gid_->suicide(&other);
	other.gid_ = nullptr;
}

template <typename T>
void iconnector<T>::update_graph (std::vector<iconnector<T>*> args)
{
	if (args.empty())
	{
		if (nullptr == gid_)
		{
			graph_node::get(this);
		}
		return;
	}
	gid_ = graph_node::get(args[0], this);
	for (size_t i = 1, n = args.size(); i < n; i++)
	{
		gid_->consume(args[i]->gid_);
	}
}

template <typename T>
struct bottom_first
{
	bool operator() (const iconnector<T>* c1, const iconnector<T>* c2) const
	{
		return c1->potential_descendent(c2);
	}
};

template <typename T>
struct top_first
{
	bool operator() (const iconnector<T>* c1, const iconnector<T>* c2) const
	{
		return c2->potential_descendent(c1);
	}
};

template <typename T>
using job_map_t = std::unordered_map<iconnector<T>*,std::vector<size_t> >;

template <typename T>
using job_queue_t = std::priority_queue<iconnector<T>*, std::vector<iconnector<T>*>, bottom_first<T> >;

template <typename T>
struct iconnector<T>::graph_node
{
	static graph_node* get (iconnector<T>* source, iconnector<T>* user = nullptr)
	{
		assert(source);
		graph_node*& gn = source->gid_;
		if (nullptr == gn) gn = new graph_node();
		if (nullptr == user) user = source;
		gn->users_.emplace(user);
		return gn;
	}

	graph_node (const graph_node&) = delete;

	graph_node (graph_node&&) = delete;

	void suicide (iconnector<T>* user)
	{
		users_.erase(user);
		if (users_.empty()) delete this;
	}

	void consume (graph_node* other)
	{
		if (this == other) return;
		while (false == other->jobs_.empty())
		{
			jobs_.push(other->jobs_.top());
			other->jobs_.pop();
		}
		freeze_ = freeze_ && other->freeze_;
		std::unordered_set<iconnector<T>*> otherusers = other->users_;
		for (iconnector<T>* ouser : otherusers)
		{
			other->suicide(ouser);
			ouser->gid_ = this;
		}
		users_.insert(otherusers.begin(), otherusers.end());
	}

	void push (iconnector<T>* icon, size_t argidx)
	{
		job_map_[icon].push_back(argidx);
		jobs_.push(icon);
	}

	bool empty (void) const { return jobs_.empty(); }

	void job_execute (void)
	{
		if (jobs_.empty()) return;
		iconnector<T>* headconn = jobs_.top();
		headconn->update(job_map_[headconn]);
		job_map_.erase(headconn);
		jobs_.pop();
	}

	bool freeze_ = false;

private:
	job_queue_t<T> jobs_;

	job_map_t<T> job_map_;

	std::unordered_set<iconnector<T>*> users_;
	
	graph_node (void) {}
	
	~graph_node (void) {}
};

}

#endif