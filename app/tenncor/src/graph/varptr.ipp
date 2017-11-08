//
//  varptr.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-13.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_VARPTR_HPP

namespace nnet {

template <typename T>
varptr<T>::varptr (void) : iobserver(false) {}

template <typename T>
varptr<T>::varptr (inode<T>* ptr) : iobserver({ptr}, false) {}

template <typename T>
varptr<T>::~varptr (void) {}

template <typename T>
varptr<T>& varptr<T>::operator = (inode<T>* other)
{
	if (this->dependencies_.empty()) this->add_dependency(other);
	else this->replace_dependency(other, 0);
	return *this;
}

template <typename T>
varptr<T>::operator inode<T>* (void) const { return get(); }

template <typename T>
inode<T>& varptr<T>::operator * (void) const { return *get(); }

template <typename T>
inode<T>* varptr<T>::operator -> (void) const { return get(); }

template <typename T>
inode<T>* varptr<T>::get (void) const
{
	if (this->dependencies_.empty()) return nullptr;
	return static_cast<inode<T>*>(this->dependencies_.at(0));
}

template <typename T>
void varptr<T>::update (std::unordered_set<size_t>) {}

template <typename T>
void varptr<T>::clear (void) { this->remove_dependency(0); }

template <typename T>
void varptr<T>::death_on_broken (void)
{
	if (false == this->dependencies_.empty())
		this->remove_dependency(0);
}

template <typename T>
placeptr<T>::placeptr (placeholder<T>* ptr) : varptr<T>(ptr) {}

template <typename T>
placeptr<T>& placeptr<T>::operator = (placeholder<T>* other)
{
	varptr<T>::operator = (other);
	return *this;
}

template <typename T>
placeptr<T>& placeptr<T>::operator = (std::vector<T> vec)
{
	*get() = vec;
	return *this;
}

template <typename T>
placeptr<T>& placeptr<T>::operator = (tensor<T>& ten)
{
	*get() = ten;
	return *this;
}

template <typename T>
placeptr<T>::operator placeholder<T>* (void) const
{
	return get();
}

template <typename T>
placeholder<T>& placeptr<T>::operator * (void)
{
	return *get();
}

template <typename T>
placeholder<T>* placeptr<T>::operator -> (void)
{
	return get();
}

template <typename T>
placeholder<T>* placeptr<T>::get (void) const
{
	return static_cast<placeholder<T>*>(varptr<T>::get());
}

}

#endif