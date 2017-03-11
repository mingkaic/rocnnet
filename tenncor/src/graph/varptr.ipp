//
//  varptr.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-13.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef varptr_hpp

namespace nnet {

template <typename T>
varptr<T>::varptr (inode<T>* ptr) : ptr_(ptr) {}

template <typename T>
varptr<T>& varptr<T>::operator = (inode<T>* other) { ptr_ = other; return *this; }

template <typename T>
varptr<T>& varptr<T>::operator = (const varptr<T>& other) { ptr_ = other.ptr_; return *this; }

template <typename T>
varptr<T>::operator inode<T>* (void) const { return ptr_; }

template <typename T>
inode<T>& varptr<T>::operator * (void) const { return *ptr_; }

template <typename T>
inode<T>* varptr<T>::operator -> (void) const { return ptr_; }

template <typename T>
inode<T>* varptr<T>::get (void) const { return ptr_; }

template <typename T>
placeptr<T>::placeptr (placeholder<T>* ptr) : varptr<T>(ptr) {}

template <typename T>
placeptr<T>& placeptr<T>::operator = (placeholder<T>* other) { this->ptr_ = other; }

template <typename T>
placeptr<T>& placeptr<T>::operator = (const placeptr<T>& other) { this->ptr_ = other.ptr_; }

template <typename T>
placeptr<T>& placeptr<T>::operator = (std::vector<T> vec)
{
    *(static_cast<placeholder<T>*>(this->ptr_)) = vec;
    return *this;
}

template <typename T>
placeptr<T>& placeptr<T>::operator = (tensor<T>& ten)
{
	*(static_cast<placeholder<T>*>(this->ptr_)) = ten;
	return *this;
}

template <typename T>
placeptr<T>::operator placeholder<T>* (void) const
{
    return static_cast<placeholder<T>*>(this->ptr_);
}

template <typename T>
placeholder<T>& placeptr<T>::operator * (void)
{
    return *(static_cast<placeholder<T>*>(this->ptr_));
}

template <typename T>
placeholder<T>* placeptr<T>::operator -> (void)
{
    return static_cast<placeholder<T>*>(this->ptr_);
}

template <typename T>
placeholder<T>* placeptr<T>::get (void) const
{
    return static_cast<placeholder<T>*>(this->ptr_);
}

}

#endif