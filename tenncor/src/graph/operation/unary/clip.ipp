//
//  clip.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-09-30.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef clip_hpp

namespace nnet {

// CLIP ELEMENT VALUES

template <typename T>
std::function<T(T)> clip_by_value<T>::get_op (void) {
	return [this](T in) {
		if (min > in) return min;
		else if (max < in) return max;
		return in;
	};
}

template <typename T>
EVOKER_PTR<T> clip_by_value<T>::clone_impl (std::string name) {
	return ivariable<T>::make_shared(new clip_by_value(*this, name));
}

template <typename T>
std::function<T(T)> clip_by_norm<T>::get_op (void) {
	T l2norm;
	this->template util_op<double>(l2norm, this->out_, [](T& out, T in) {
		out += sqrt(in);
	});
	l2norm = sqrt(l2norm);
	return [this, &l2norm](T in) {
		return in * cap / l2norm;
	};
}

template <typename T>
EVOKER_PTR<T> clip_by_norm<T>::clone_impl (std::string name) {
	return ivariable<T>::make_shared(new clip_by_norm(*this, name));
}

}

#endif