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
ievoker<T>* clip_by_norm<T>::clone_impl (std::string name) {
	return new clip_by_norm(*this, name);
}

}

#endif