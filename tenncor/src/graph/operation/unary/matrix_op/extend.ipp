//
//  extend.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-08.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef extend_hpp

namespace nnet {

// TENSOR EXTENSION

template <typename T>
void extend<T>::setup_gradient (void) {
	std::vector<ivariable<T>*> args;
	for (ccoms::subject* child : this->dependencies_) {
		if (ivariable<T>* arg = dynamic_cast<ivariable<T>*>(child)) {
			ivariable<T>* g = arg->get_gradient();
			if (nullptr == watch.lock()) {
				this->grad = extend<T>::make(g, index, multiplier);
			} else {
				this->grad = extend<T>::make(g, watch);
			}
		}
	}
}

template <typename T>
void extend<T>::shape_eval (void) {
	if (this->var) {
		tensor_shape ts = this->var->get_shape();
		if (ts.is_fully_defined()) {
			size_t below_dim;
			size_t cline;
			this->update(
					this->change_shape(ts, index, multiplier, below_dim, cline));
		}
	}
}

template <typename T>
void extend<T>::copy (const ivariable<T>& other, std::string name) {
	if (const extend<T>* eptr = dynamic_cast<const extend<T>*>(&other)) {
		index = eptr->index;
		multiplier = eptr->multiplier;
	}
	iunar_ops<T>::copy(other, name);
}

template <typename T>
extend<T>::extend (ivariable<T>* in, ivariable<T>* watch) :
        iunar_ops(in), watch(watch) {
	if (session::pre_shape_eval()) {
	    shape_eval();
    }
}

template <typename T>
extend<T>::extend (ivariable<T>* in, size_t index, size_t multiplier) :
        iunar_ops(in), index(index), multiplier(multiplier) {
	if (session::pre_shape_eval()) {
	    shape_eval();
    }
}

template <typename T>
ievoker<T>* extend<T>::clone_impl (std::string name) {
	return new extend(*this, name);
}

template <typename T>
extend<T>& extend<T>::operator = (const ivariable<T>& other) {
	if (this != &other) {
		copy(other);
	}
	return *this;
}

template <typename T>
void extend<T>::set_ext_info (ivariable<T>* watch) {
	this->index = 0;
	this->multiplier = 0;
	this->consume(*(watch.get()));
	this->watch = watch;
}

template <typename T>
void extend<T>::set_ext_info (size_t index, size_t multiplier) {
	this->index = index;
	this->multiplier = multiplier;
	watch = nullptr;
//	std::unordered_set<ioperation<T>*> cons = watch->get_consumers();
//	if (cons.end() != cons.find(this)) {
//		cons.erase(this);
//	}
}

template <typename T>
void extend<T>::update (void) {
    ivariable<T>* arg = dynamic_cast<ivariable<T>*>(this->dependencies_[0]);
	assert(arg);
	const tensor<T>& in = arg->get_eval();
	if (nullptr == watch.lock()) {
		tensor<T> *ans = this->extend_op(in, index, multiplier);
		this->out_ = *ans;
		delete ans;
	} else {
		this->out_ = in;
		std::vector<size_t> target = watch.lock()->get_shape().as_list();
		std::vector<size_t> orig = in.get_shape().as_list();
		// this actually get expensive for big shapes, TODO: refactor/change anti-pattern
		for (size_t i = 0; i < target.size(); i++) {
			if (i < orig.size()) {
				if (target[i] > orig[i]) {
					tensor<T> *ans = this->extend_op(this->out_, i, target[i]/orig[i]);
					this->out_ = *ans;
					delete ans;
				}
			} else {
				tensor<T>* ans = this->extend_op(this->out_, i, target[i]);
				this->out_ = *ans;
				delete ans;
			}
		}
	}
	return this->out_;
}

}

#endif