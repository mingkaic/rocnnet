//
//  tensor_op.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-09.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef tensor_op_hpp

namespace nnet {

// Collect Operations

template <typename T>
void collect_op<T>::setup_gradient (void) {
	std::vector<ivariable<T>*> args;
	for (ccoms::subject* child : this->dependencies_) {
		if (ivariable<T>* arg = dynamic_cast<ivariable<T>*>(child)) {
			args.push_back(arg);
		}
	}
	this->grad = der_(args);
}

template <typename T>
ievoker<T>* collect_op<T>::clone_impl (std::string name) {
	return new elementary(args_, for_each_, der_, name);
}

template <typename T>
void collect_op<T>::shape_eval (void) {
	auto it = args_.begin();
	tensor_shape first = this->get_tensor_from(*it).get_shape();
	if (first.is_fully_defined()) {
		for (it++; args_.end() != it; it++) {
			tensor_shape ts = this->get_tensor_from(*it).get_shape();
			assert(first.is_compatible_with(ts) ||
				   1 == ts.n_dims() || 1 == first.n_dims());
			if (ts.n_dims() > first.n_dims()) first = ts;
		}
		this->update(first);
	}
}

template <typename T>
collect_op<T>::collect_op (std::vector<ivariable<T>*> args,
		std::function<void(T&, T, size_t)> op,
		BUILD_DERIVE<T> der,
		std::string name) :
			ioperation<T>(std::vector<ccoms::subject*>(args.begin(), args.end()), name),
			for_each_(op),
			der_(der) {
	if (session::pre_shape_eval()) {
		shape_eval();
	}
}

template <typename T>
void collect_op<T>::update (void) {
    this->out_ = tensor<T>(0);

	for (ccoms::subject* sub : this->dependencies_) {
	    if (ivariable<T>* arg = dynamic_cast<ivariable<T>*>(sub)) {
		    //this->elem_op(this->out_, arg->get_eval(), collect_);
		}
	}
	this->notify();
}

// template <typename T>
// ivariable<T>* clip_by_norm (const ivariable<T>* a, T cap) {
// 	if (nullptr == a) return nullptr;
// 	ivariable<T>* op = new elementary<T>(std::vector<ivariable<T>*>{a},
// 		[cap](T& collector, T other) {
// 			if (min > other) other = min;
// 			else if (max < other) other = max;
// 			collector = other;
// 		},
// 		[cap](std::vector<ivariable<T>*> args) {
// 			ivariable<T>* a = args.front();
// 			return clip_by_norm(a->get_gradient(), cap);
// 		}, 
// 	nnutils::formatter() << "clip_norm(" << a->get_name() << ")");
// 	return op;
// }

}

#endif