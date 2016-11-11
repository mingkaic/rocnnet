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
void compress<T>::setup_gradient (void) {
	std::vector<ivariable<T>*> args;
	for (ccoms::subject* child : this->dependencies_) {
		if (ivariable<T>* arg = dynamic_cast<ivariable<T>*>(child)) {
			this->grad_ = std::shared_ptr<compress<T> >(
				new compress(arg->get_gradient(), index, collector));
		}
	}
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
void compress<T>::copy (const compress<T>& other, std::string name) {
	index = other.index;
	collector = other.collector;
	ioperation<T>::copy(other, name);
}

template <typename T>
compress<T>::compress (ivariable<T>* in, size_t index) :
        compress(in, index, [](const std::vector<T>& data) {
	T ans = 0;
	for (T raw : data) {
		ans += raw;
	}
	ans /= data.size();
	return ans;
}) {}

template <typename T>
compress<T>::compress (ivariable<T>* in, size_t index, std::function<T(const std::vector<T>&)> collector) :
        iunar_ops(in), index(index), collector(collector) {
	if (session::pre_shape_eval()) {
	    shape_eval();
    }
}

template <typename T>
ievoker<T>* compress<T>::clone_impl (std::string name) {
	return new compress(*this, name);
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
	if (session::pre_shape_eval()) {
	    shape_eval();
    } // re-eval
}

template <typename T>
void compress<T>::update (void) {
    ivariable<T>* arg = dynamic_cast<ivariable<T>*>(this->dependencies_[0]);
	assert(arg);
	const tensor<T>& in = arg->get_eval();
	tensor<T>* ans = this->compress_op(in, index, collector);
	this->out_ = *ans;
	delete ans;
	return this->out_;
}

}

#endif