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
varptr<T>::varptr (ivariable<T>* ptr) : ptr_(ptr) {}

template <typename T>
varptr<T>& varptr<T>::operator = (ivariable<T>* other) { ptr_ = other; }

template <typename T>
varptr<T>& varptr<T>::operator = (const varptr<T>& other) { ptr_ = other.ptr_; return *this; }

template <typename T>
varptr<T>::operator ivariable<T>* (void) const { return ptr_; }

template <typename T>
ivariable<T>& varptr<T>::operator * (void) const { return *ptr_; }

template <typename T>
ivariable<T>* varptr<T>::operator -> (void) const { return ptr_; }

template <typename T>
ivariable<T>* varptr<T>::get (void) const { return ptr_; }

template <typename T>
placeptr<T>::placeptr (placeholder<T>* ptr) : ptr_(ptr) {}

template <typename T>
placeptr<T>& placeptr<T>::operator = (placeholder<T>* other) { ptr_ = other; }

template <typename T>
placeptr<T>& placeptr<T>::operator = (const placeptr<T>& other) { ptr_ = other.ptr_; }

template <typename T>
placeptr<T>& placeptr<T>::operator = (std::vector<T> vec) { *ptr_ = vec; }

template <typename T>
placeptr<T>& placeptr<T>::operator = (const tensor<T>& ten) { *ptr_ = ten; }

template <typename T>
placeptr<T>::operator placeholder<T>* (void) const { return ptr_; }

template <typename T>
placeholder<T>& placeptr<T>::operator * (void) { return *ptr_; }

template <typename T>
placeholder<T>* placeptr<T>::operator -> (void) { return ptr_; }

template <typename T>
placeholder<T>* placeptr<T>::get (void) const { return ptr_; }

}

#endif