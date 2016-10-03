//
//  variable.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef variable_hpp

namespace nnet {

// initializers

template <typename T>
void const_init<T>::operator () (tensor<T>& in) {
	this->delegate_task(in, [this](T* raw_data, size_t size) {
		std::fill(raw_data, raw_data+size, value);
	});
}

template <typename T>
void random_uniform<T>::operator () (tensor<T>& in) {
	this->delegate_task(in, [this](T* raw_data, size_t size) {
		for (size_t i = 0; i < size; i++) {
			raw_data[i] = distribution(session::get_generator());
		}
	});
}

// variable interface

template <typename T>
void ivariable<T>::copy (
	ivariable<T> const & other,
	std::string name) {
	out = other.out;
	if (0 == name.size()) {
		name = other.name+"_cpy";
	}
	this->name = name;
}

template <typename T>
ivariable<T>& ivariable<T>::operator = (const ivariable<T>& other) {
	if (this != &other) {
		copy(other);
	}
	return *this;
}

template <typename T>
tensor<T>* ivariable<T>::derive (ivariable<T>* over) const {
	if (over == this) {
		memory_alloc all;
		const_init<T> oneinit(1);
		tensor<T>* ones = new tensor<T>(this->out.get_shape());
		ones->allocate(all);
		oneinit(*ones);
		return ones;
	}
	return this->calc_derive(over);
}

// variable implementation

template <typename T>
void variable<T>::copy (
	variable<T> const & other,
	std::string name) {
	init = other.init->copy();
	ivariable<T>::copy(other, name);
}

template <typename T>
variable<T>::variable (const variable<T>& other, std::string name) {
	copy(other, name);
}

template <typename T>
variable<T>::variable (
	tensor_shape const & shape,
	std::string name) {
	this->out.set_shape(shape);
	this->name = name;
}

template <typename T>
variable<T>::variable (
	tensor_shape const & shape,
	initializer<T>& init,
	std::string name)
	: variable(shape, name) {
	this->init = init.copy();
}

template <typename T>
variable<T>* variable<T>::clone (std::string name) {
	return new variable(*this, name);
}

// template <typename T>
// variable<T>::variable (
// 	variable<T> const & other,
// 	std::string name) {
// 	copy(other);
// 	if (false == name.empty()) {
// 		this->name = name;
// 	}
// }

template <typename T>
variable<T>::~variable (void) {
	if (nullptr != this->init) {
		delete this->init;
	}
}

template <typename T>
variable<T>& variable<T>::operator = (const ivariable<T>& other) {
	if (this != &other) {
		if (nullptr != this->init) {
			delete this->init;
		}

		if (const variable<T>* vptr = dynamic_cast<const variable<T>*>(&other)) {
			copy(*vptr);
		} else {
			ivariable<T>::copy(other);
		}
	}
	return *this;
}

template <typename T>
tensor<T>& variable<T>::initialize (void) {
	memory_alloc all;
	this->out.allocate(all);
	assert(init != nullptr);
	(*init)(this->out);
	is_init = true;
	return this->out;
}

template <typename T>
tensor<T>& variable<T>::initialize (tensor_shape alloc_shape) {
	memory_alloc all;
	this->out.allocate(all, alloc_shape);
	(*init)(this->out);
	is_init = true;
	return this->out;
}

template <typename T>
void variable<T>::update (const tensor<T>& in, std::function<T(T,T)> op) {
	// TODO: use a better method of manipulating rawdata
	assert(this->out.is_same_size(in));
	T* old_data = this->out.raw_data;
	const T* new_data = in.raw_data;
	for (size_t i = 0; i < in.n_elems(); i++) {
		old_data[i] = op(old_data[i], new_data[i]);
	}
}

template <typename T>
const tensor<T>& variable<T>::eval (void) {
	assert(is_init);
	return this->out;
}

// placeholder implementation

template <typename T>
struct placeholder<T>::open_init : public initializer<T> {
	std::vector<T> prime;

	virtual void operator () (tensor<T>& in) {
		this->delegate_task(in, [this](T* raw_data, size_t size) {
			std::copy(prime.begin(), prime.end(), raw_data);
		});
	}

	virtual initializer<T>* copy (void) { return new open_init(); }
};

template <typename T>
placeholder<T>::placeholder (const tensor_shape& shape, std::string name)
	: variable<T>(shape, name) {
	memory_alloc all;
	this->out.allocate(all);
}

template <typename T>
variable<T>& placeholder<T>::operator = (std::vector<T> data) {
	this->init = new open_init();

	assert(data.size() <= this->out.n_elems());
	dynamic_cast<open_init*>(this->init)->prime = data;
	(*this->init)(this->out);

	delete this->init;
	this->init = nullptr;
	this->is_init = true;
	return *this;
}

template <typename T>
variable<T>& placeholder<T>::operator = (const tensor<T>& data) {
	this->out = data;
	return *this;
}

}

#endif
