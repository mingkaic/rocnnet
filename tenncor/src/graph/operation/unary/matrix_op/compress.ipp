//
//  compress.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-08.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef compress_hpp

namespace nnet {

// TENSOR COMPRESSION

template <typename T>
void compress<T>::make_gradient (VAR_PTR<T>& safety_ref) {
	VAR_PTR<T> g = this->var->get_gradient();
	this->set_gradient(std::shared_ptr<compress<T> >(new compress(g, index, collector)));
	safety_ref = this->grad;
}

template <typename T>
void compress<T>::shape_eval (void) {
	if (this->var) {
		tensor_shape ts = this->var->get_shape();
		if (ts.is_fully_defined()) {
			size_t below_dim;
			size_t cline;
			this->update(
					this->change_shape(ts, index, 0, below_dim, cline));
		}
	}
}

template <typename T>
void compress<T>::copy (const ivariable<T>& other, std::string name) {
	if (const compress<T>* cptr = dynamic_cast<const compress<T>*>(&other)) {
		index = cptr->index;
		collector = cptr->collector;
	}
	iunar_ops<T>::copy(other, name);
}

template <typename T>
compress<T>::compress (VAR_PTR<T> in, size_t index) : compress(in, index, [](const std::vector<T>& data) {
	T ans = 0;
	for (T raw : data) {
		ans += raw;
	}
	ans /= data.size();
	return ans;
}) {}

template <typename T>
compress<T>::compress (VAR_PTR<T> in, size_t index, std::function<T(const std::vector<T>&)> collector)
		: index(index), collector(collector) {
	this->init(in);
	shape_eval();
}

template <typename T>
EVOKER_PTR<T> compress<T>::clone_impl (std::string name) {
	return ivariable<T>::make_shared(new compress(*this, name));
}

template <typename T>
compress<T>& compress<T>::operator = (const ivariable<T>& other) {
	if (this != &other) {
		copy(other);
	}
	return *this;
}

template <typename T>
void compress<T>::set_cmpr_info (size_t index, std::function<T(const std::vector<T>&)> collector) {
	this->index = index;
	this->collector = collector;
	shape_eval(); // re-eval
}

template <typename T>
const tensor<T>& compress<T>::eval (void) {
	static tensor<T> one(1);
	if (this->derive_this) {
		return one;
	}
	assert(nullptr != this->var);
	const tensor<T>& in = this->var->eval();
	tensor<T>* ans = this->compress_op(in, index, collector);
	this-> out_ = *ans;
	delete ans;
	return this-> out_;
}

}

#endif