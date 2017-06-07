//
//  inode.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_INODE_HPP

namespace nnet
{

template <typename T>
inode<T>* inode<T>::clone (void) const
{
	return clone_impl();
}

template <typename T>
inode<T>* inode<T>::move (void)
{
	return move_impl();
}

template <typename T>
inode<T>& inode<T>::operator = (const inode<T>& other)
{
	if (this != &other)
	{
		subject::operator = (other);
		label_ = other.label_;
	}
	return *this;
}

template <typename T>
inode<T>& inode<T>::operator = (inode<T>&& other)
{
	if (this != &other)
	{
		subject::operator = (other);
		label_ = std::move(other.label_);
	}
	return *this;
}

template <typename T>
std::string inode<T>::get_uid (void) const
{
	return id_;
}

template <typename T>
std::string inode<T>::get_name (void) const
{
	return "<"+label_+":"+id_+">";
}

template <typename T>
inode<T>::inode (std::string label) :
	subject(),
	label_(label) {}

template <typename T>
inode<T>::inode (const inode<T>& other) :
	subject(other),
	label_(other.label_) {}

template <typename T>
inode<T>::inode (inode<T>&& other) :
	subject(std::move(other)),
	label_(std::move(other.label_)) {}

template <typename T>
std::vector<T> expose (const inode<T>* var)
{
	if (nullptr == var) return std::vector<T>{};
	const tensor<T>* ten = var->get_eval();
	if (nullptr == ten) return std::vector<T>{};
	return ten->expose();
}

template <typename T>
bool operator == (const inode<T>& c, T scalar)
{
	assert(c.good_status());
	std::vector<T> res = expose<T>(&c);
	return 1 == res.size() && scalar == res[0];
}

template <typename T>
bool operator != (const inode<T>& c, T scalar)
{
	assert(c.good_status());
	std::vector<T> res = expose<T>(&c);
	return 1 != res.size() || scalar != res[0];
}

}

#endif