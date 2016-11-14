//
//  operation.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef operation_hpp

namespace nnet {

// OPERATION INTERFACE UTILITY FUNCTIONS

template <typename T>
void ioperation<T>::copy (const ioperation<T>& other, std::string name) {
	// if grad is not being observed, then and only then delete
	if (nullptr != grad_ && grad_->no_audience()) {
		delete grad_;
	}
	// shallow copy
	grad_ = other.grad_;
	ivariable<T>::copy(other, name);
}

template <typename T>
ioperation<T>::ioperation (std::vector<ivariable<T>*> dependencies, std::string name) :
		ivariable<T>(std::vector<size_t>{}, name),
		iobserver(std::vector<ccoms::subject*>(dependencies_.begin(), dependencies_.end())) {
	if (session::pre_shape_eval()) {
		std::vector<tensorshape> s;
		for (ivariable<T>* a : dependencies) { s.push_back(a->get_shape()); }
		shape_eval(s);
	}
}

template <typename T>
ioperation<T>::~ioperation (void) {
	if (nullptr != grad_) {
		delete grad_;
	}
}

template <typename T>
ioperation<T>& ioperation<T>::operator = (const ioperation<T>& other) {
	if (this != &other) {
		this->copy(other);
	}
	return *this;
}

}

#endif
