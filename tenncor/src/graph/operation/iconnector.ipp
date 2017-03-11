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
iconnector<T>* iconnector<T>::clone (void) const
{
	return static_cast<iconnector<T>*>(this->clone_impl());
}

template <typename T>
iconnector<T>& iconnector<T>::operator = (const iconnector<T>& other)
{
	if (this != &other)
	{
		react::iobserver::operator = (other);
		inode<T>::operator = (other);
	}
	return *this;
}

template <typename T>
iconnector<T>& iconnector<T>::operator = (iconnector<T>&& other)
{
	if (this != &other)
	{
		react::iobserver::operator = (other);
		inode<T>::operator = (other);
	}
	return *this;
}

template <typename T>
std::string iconnector<T>::get_name (void) const
{
	std::string args;
	access_dependencies(
	[&args](const react::subject* subs)
	{
		if (const inode<T>* arg = dynamic_cast<const inode<T>*>(subs))
		{
			args += arg->get_label() + ",";
		}
	});
	if (!args.empty()) args.pop_back();
	return inode<T>::get_name() + "(" + args + ")";
}

template <typename T>
bool iconnector<T>::is_same_graph (const iconnector<T>* other) const
{
	return **gid_ == **other->gid_;
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
		variable<T>* leaf = t->first;
		if (mine.end() == mine.find(leaf))
		{
			return false;
		}
	}
	return true;
}

template <typename T>
iconnector<T>::iconnector (std::vector<inode<T>*> dependencies, std::string name) :
	react::iobserver(dependencies),
	inode<T>(name), zero(0), one(1)
{
	update_graph(this->dependencies_);
}

template <typename T>
iconnector<T>::iconnector (const iconnector<T>& other) :
	react::iobserver(other),
	inode<T>(other), zero(0), one(1) {} // don't update graph for its new children

template <typename T>
iconnector<T>::iconnector (iconnector<T>&& other) :
	react::iobserver(other),
	inode<T>(other), zero(0), one(1) {}

template <typename T>
void iconnector<T>::update_graph (std::vector<inode<T>*> args)
{
	gid_ = &(&this->id_);
	for (inode<T>* arg : args)
	{
		if (iconnector<T>* iconn = dynamic_cast<iconnector<T>*>(arg))
		{
			*(iconn->gid_) = *gid_;
		}
	}
}

}

#endif