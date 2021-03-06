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
inode<T>::~inode (void) {}

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
std::string inode<T>::get_label (void) const
{
	return label_;
}

template <typename T>
std::string inode<T>::get_name (void) const
{
	return "<" + label_ + ":" + this->get_uid() + ">";
}

template <typename T>
std::string inode<T>::get_summaryid (void) const
{
	return get_name();
}

template <typename T>
void inode<T>::set_label (std::string label)
{
	label_ = label;
}

template <typename T>
bool inode<T>::find_audience (std::string label, std::unordered_set<inode<T>*>& audience) const
{
	for (auto audpair : audience_)
	{
		iobserver* aud = audpair.first;
		if (inode<T>* anode = dynamic_cast<inode<T>*>(aud))
		{
			if (0 == anode->label_.compare(label))
			{
				audience.insert(anode);
			}
		}
	}
	return false == audience.empty();
}

template <typename T>
void inode<T>::set_metadata (std::string key, size_t value)
{
	metadata_[key] = value;
}

template <typename T>
void inode<T>::extract_metadata (inode<T>* n)
{
	for (auto npair : n->metadata_)
	{
		auto metait = metadata_.find(npair.first);
		if (metadata_.end() == metait)
		{
			metadata_[npair.first] = npair.second;
		}
		else if (npair.second != metait->second)
		{
			// warn
		}
	}
}

template <typename T>
optional<size_t> inode<T>::get_metadata (std::string key) const
{
	optional<size_t> out;
	auto it = metadata_.find(key);
	if (metadata_.end() != it)
	{
		out = it->second;
	}
	return out;
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
const tensor<T>* inode<T>::take_eval (inode<T>* source) const
{
	return source->get_eval();
}

template <typename T>
inode<T>* inode<T>::take_gradient (inode<T>* source, variable<T>* leaf) const
{
	return source->get_gradient(leaf);
}

template <typename T>
std::vector<T> expose (inode<T>* var)
{
	if (nullptr == var) return std::vector<T>{};
	const tensor<T>* ten = var->eval();
	if (nullptr == ten) return std::vector<T>{};
	return ten->expose();
}

}

#endif