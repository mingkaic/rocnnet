//
//  varptr.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-13.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef varptr_hpp
#define varptr_hpp

#include "graph/variable/placeholder.hpp"

namespace nnet {

// tensor variable pointer wrapper

template <typename T>
class varptr {
	private:
		ivariable<T>* ptr_;

	public:
		varptr (ivariable<T>* ptr) : ptr_(ptr) {}
		varptr<T>& operator = (ivariable<T>* other) { ptr_ = other; }
		varptr<T>& operator = (const varptr<T>& other) { ptr_ = other.ptr_; }

		explicit operator ivariable<T>* (void) const { return ptr_; }
		ivariable<T>& operator * (void) { return *ptr_; }
		ivariable<T>* operator -> (void) { return ptr_; }
};

template <typename T>
class placeptr {
	private:
		placeholder<T>* ptr_;

	public:
		placeptr (placeholder<T>* ptr) : ptr_(ptr) {}
		placeholder<T>& operator = (std::vector<T> vec) { *ptr_ = vec; }
		placeholder<T>& operator = (const tensor<T>& ten) { *ptr_ = ten; }

		explicit operator ivariable<T>* (void) const { return ptr_; }
		ivariable<T>& operator * (void) { return *ptr_; }
		ivariable<T>* operator -> (void) { return ptr_; }
};

template <typename T>
varptr<T> make_var (T scalar) { return new variable<T>(scalar); }

template <typename T>
varptr<T> make_const (T scalar) { return new constant<T>(scalar); }

template <typename T>
varptr<T> make_var (const tensorshape& shape, std::string name = "") {
	return new variable<T>(shape, name);
}

template <typename T>
placeptr<T> make_place (const tensorshape& shape, std::string name = "") {
	return new placeholder<T>(shape, name);
}

template <typename T>
varptr<T> make_var (const tensorshape& shape, initializer<T>& init, std::string name = "") {
	return new variable<T>(shape, init, name);
}

template <typename T>
placeptr<T> make_place (const tensorshape& shape, initializer<T>& init, std::string name = "") {
	return new placeholder<T>(shape, init, name);
}

}

#endif /* varptr_hpp */
