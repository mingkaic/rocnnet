//
//  iconnector.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-01.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_ICONNECTOR_HPP

namespace nnet
{

template <typename T>
inline std::vector<subject*> to_sub (std::vector<inode<T>*> nodes)
{
	std::vector<subject*> subs;
	std::transform(nodes.begin(), nodes.end(), subs.begin(),
		[](inode<T>* n) -> subject*
		{
			return n;
		});
	return subs;
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
	}
	return *this;
}

template <typename T>
iconnector<T>& iconnector<T>::operator = (iconnector<T>&& other)
{
	if (this != &other)
	{
		iobserver::operator = (other);
		inode<T>::operator = (other);
	}
	return *this;
}

template <typename T>
std::string iconnector<T>::get_name (void) const
{
	std::string args;
	for (const subject* subs : this->dependencies_)
	{
		if (const inode<T>* arg = dynamic_cast<const inode<T>*>(subs))
		{
			args += arg->get_label() + ",";
		}
	};
	if (!args.empty()) args.pop_back();
	return inode<T>::get_name() + "(" + args + ")";
}

template <typename T>
bool iconnector<T>::is_same_graph (const iconnector<T>* other) const
{
	graph_node* lhs = nullptr;
	graph_node* rhs = nullptr;
	gid_->get_master(lhs);
	other->gid_->get_master(rhs);

	return lhs == rhs;
}

template <typename T>
bool iconnector<T>::potential_descendent (iconnector<T>* n) const
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
iconnector<T>::iconnector (std::vector<inode<T>*> dependencies, std::string name) :
	iobserver(to_sub<T>(dependencies)),
	inode<T>(name)
{
	update_graph(dependencies);
}

template <typename T>
iconnector<T>::iconnector (const iconnector<T>& other) :
	iobserver(other),
	inode<T>(other) {} // don't update graph for its new children

template <typename T>
iconnector<T>::iconnector (iconnector<T>&& other) :
	iobserver(other),
	inode<T>(other) {}

template <typename T>
void iconnector<T>::update_graph (std::vector<inode<T>*> args)
{
	if (args.size() == 1)
	{
		if (iconnector<T>* iconn = dynamic_cast<iconnector<T>*>(args[0]))
		{
			iconn->gid_->get_master(gid_);
		}
	}

	if (nullptr == gid_)
	{
		gid_ = new graph_node;
		for (inode<T>* arg : args)
		{
			if (iconnector<T>* iconn = dynamic_cast<iconnector<T>*>(arg))
			{
				iconn->gid_->update(gid_);
				gid_->replace(iconn->gid_); // iconn->gid_ = gid_
			}
		}
	}
	gid_->users_++; // add this as a user
}

template <typename T>
struct iconnector<T>::graph_node
{
	graph_node* top_ = nullptr;
	size_t users_ = 0;

	// traverse up to single master (worst case: O(n))
	void get_master (graph_node*& out) const
	{
		if (top_ == nullptr)
		{
			out = const_cast<graph_node*>(this);
		}
		else
		{
			top_->get_master(out);
		}
	}

	// add candidate to master then return new master
	void update (graph_node* master)
	{
		graph_node* old = nullptr;
		get_master(old);
		master->users_ += old->users_;
		old->top_ = master;
	}

	// replace old with this
	void replace (graph_node*& old)
	{
		if (0 == --old->users_)
		{
			delete old;
		}
		old = this;
		users_++;
	}
};

}

#endif