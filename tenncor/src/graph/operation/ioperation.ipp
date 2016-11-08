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
tensor_shape ioperation<T>::get_element_shape (
	const tensor<T>& a,
	const tensor<T>& b) const {
	tensor_shape ts_out;
	if (1 == a.n_elems()) {
		ts_out = b.get_shape();
	} else if (1 == b.n_elems() || a.is_same_size(b)){
		ts_out = a.get_shape();
	}
	return ts_out;
}

template <typename T>
tensor_shape ioperation<T>::transpose_shape (const tensor_shape& ins) const {
	// restrict shapes
	assert(ins.n_dims() == 2);
	std::vector<size_t> inl = ins.as_list();
	size_t dimX = inl[0];
	size_t dimY = inl[1];
	return tensor_shape(std::vector<size_t>{dimY, dimX});
}

template <typename T>
tensor_shape ioperation<T>::change_shape (
	const tensor_shape& ins,
	size_t index,
	double multiplier,
	size_t& below_dim,
	size_t& at_idx) const {
	ins.assert_is_fully_defined(); // can't change shape if shape info is not complete
	std::vector<size_t> tv = ins.as_list();
	// allocated additional space along index
	below_dim = 1;
	bool shape_changed = false;
	if (multiplier < 1) {
		at_idx = tv[index];
		if (0 == multiplier) {
			multiplier = 1/(double) at_idx;
		}
		// edge indices
		if (0 == index) {
			// pop front (tv is not empty, otherwise in.n_dims() > 0 would catch it)
			tv.front() = std::move(tv.back());
			tv.pop_back();
			shape_changed = true;
		} else if (tv.size()-1 == index) {
			below_dim = ins.n_elems() / at_idx; // above is 1
			tv.pop_back();
			shape_changed = true;
		}
	} else if (multiplier >= 1 && index >= tv.size()) {
		at_idx = index == tv.size() ? tv.back() : multiplier;
		// extending extra dimensions
		size_t extra_dims = index - tv.size();
		if (extra_dims) {
			tv.insert(tv.end(), extra_dims, 1);
		}
		tv.push_back(multiplier);
		below_dim = ins.n_elems() / at_idx;
		shape_changed = true;
	}
	if (0 == below_dim) below_dim = 1; // round up

	if (false == shape_changed) {
		at_idx = tv[index];
		for (size_t i = 0; i < index; i++) {
			below_dim *= tv[i];
		}
		tv[index] *= multiplier;
		if (0 == tv[index]) {
			tv[index] = 1; // ensure the shape is defined if it was previously defined
		}
	}

	return tensor_shape(tv);
}

template <typename T>
tensor_shape ioperation<T>::get_matrix_shape (
	const tensor<T>& t1, const tensor<T>& t2,
	bool transposeA, bool transposeB,
	size_t& common_dim) const {
	tensor_shape ts_out;
	tensor_shape t1s = t1.get_shape();
	tensor_shape t2s = t2.get_shape();

	if (2 >= t1s.n_dims() &&
		2 >= t2s.n_dims()) {
		std::vector<size_t> al = t1s.as_list();
		std::vector<size_t> bl = t2s.as_list();
		size_t ax = t1s.n_dims() ? al[0] : 0;
		size_t ay = t1s.n_dims() > 1 ? al[1] : 1;
		size_t bx = t2s.n_dims() ? bl[0] : 0;
		size_t by = t2s.n_dims() > 1 ? bl[1] : 1;
		size_t dimX, dimY, dimZ;
		if (transposeA && transposeB) {
			if (ay == bx) {
				common_dim = ay;
				ts_out = std::vector<size_t>{by, ax};
			}
		} else if (transposeA) {
			if (ay == by) {
				common_dim = ay;
				ts_out = std::vector<size_t>{bx, ax};
			}
		} else if (transposeB) {
			if (ax == bx) {
				common_dim = ax;
				ts_out = std::vector<size_t>{by, ay};
			}
		} else { // !(transposeA || transposeB)
			if (ax == by) {
				common_dim = ax;
				ts_out = std::vector<size_t>{bx, ay};
			}
		}
	}
	return ts_out;
}

template <typename T>
template <typename U>
void ioperation<T>::util_op (
	U& out, const tensor<T>& in,
	std::function<void(U&, T)> op) const {
	const T* inraw = this->get_raw(in);
	size_t limit = in.n_elems();
	for (size_t i = 0; i < limit; i++) {
		op(out, inraw[i]);
	}
}

template <typename T>
void ioperation<T>::elem_op (
		tensor<T>& out,
		const tensor<T>& in,
		std::function<void(T&, T)> op) const {
	size_t outs = out.n_elems();
	size_t ins = in.n_elems();
	const T* inraw = this->get_raw(in);

	if (outs < ins && 1 == outs) {
		T scalar = out.get({0});
		out = in;
		T* outraw = this->get_raw(out);
		for (size_t i = 0; i < ins; i++) {
			T buffer = scalar;
			op(buffer, inraw[i]);
			outraw[i] = buffer;
		}
	} else {
		T* outraw = this->get_raw(out);
		if (1 == ins) {
			for (size_t i = 0; i < outs; i++) {
				op(outraw[i], inraw[0]);
			}
		} else if (outs == ins) {
			for (size_t i = 0; i < outs; i++) {
				op(outraw[i], inraw[i]);
			}
		} else {
			throw std::invalid_argument(
			"cannot element-wise operate on tensors of vastly different shapes");
		}
	}
}

template <typename T>
tensor<T>* ioperation<T>::get_trace (const ivariable<T>& in) const {
	std::vector<size_t> tv = this->get_tensor_from(in).get_shape().as_list();
	size_t n_col = tv[0];
	size_t len_trace;
	if (tv[0] > tv[1]) {
		len_trace = tv[1];
		tv[0] = 1;
	} else {
		len_trace = tv[0];
		tv[1] = 1;
	}
	tensor<T>* ans = new tensor<T>(tv);
	T* inraw = this->get_raw(this->get_tensor_from(in));
	T* outraw = this->get_raw(*ans);
	for (size_t i = 0; i < len_trace; i++) {
		outraw[i] = inraw[i * (1 + n_col)];
	}
	return ans;
}

template <typename T>
tensor<T>* ioperation<T>::util_op (
	const tensor<T>& in,
	std::function<T(T)> op) const {
	tensor<T>* ans = new tensor<T>(in.get_shape());
	T* outraw = this->get_raw(*ans);
	const T* inraw = this->get_raw(in);
	for (size_t i = 0; i < ans->n_elems(); i++) {
		outraw[i] = op(inraw[i]);
	}
	return ans;
}

template <typename T>
tensor<T>* ioperation<T>::util_op (
	const tensor<T>& a,
	const tensor<T>& b,
	std::function<T(T, T)> op) const {
	tensor_shape ts = get_element_shape(a, b);
	assert(0 < ts.n_dims());
	tensor<T>* ans;
	ans = new tensor<T>(ts);

	if (1 == a.n_elems()) {
		T scalar = a.get({0});
		const T* in = this->get_raw(b);
		T* out = this->get_raw(ans);
		for (size_t i = 0; i < ans->n_elems(); i++) {
			out[i] = op(scalar, in[i]);
		}
	} else if (1 == b.n_elems()) {
		const T* in = this->get_raw(a);
		T scalar = b.get({0});
		T* out = ans->raw_data;
		for (size_t i = 0; i < ans->n_elems(); i++) {
			out[i] = op(in[i], scalar);
		}
	} else if (a.is_same_size(b)) {
		const T* ina = this->get_raw(a);
		const T* inb = this->get_raw(b);
		T* out = this->get_raw(*ans);
		for (size_t i = 0; i < ans->n_elems(); i++) {
			out[i] = op(ina[i], inb[i]);
		}
	} else {
		throw std::invalid_argument(
			"cannot element-wise operate on tensors of vastly different shapes");
	}
	return ans;
}

template <typename T>
tensor<T>* ioperation<T>::transpose_op (
	tensor<T> const & in) const {
	// restrict shapes
	tensor_shape news = transpose_shape(in.get_shape());
	std::vector<size_t> inl = news.as_list();
	size_t dimY = inl[0];
	size_t dimX = inl[1];
	tensor<T>* ans = new tensor<T>(news); // new shape
	const T* rawin = this->get_raw(in);
	T* rawout = this->get_raw(*ans);
	// not in place so x = y+1 doesn't work
	for (size_t y = 0; y < dimY; y++) {
		for (size_t x = 0; x < dimX; x++) {
			rawout[y+x*dimY] = rawin[x+y*dimX];
		}
	}
	return ans;
}

// restrict matrices shape at evaluation time
template <typename T>
tensor<T>* ioperation<T>::matmul_op (
	const tensor<T>& a,
	const tensor<T>& b,
	bool transposeA,
	bool transposeB) const {
	size_t dimZ;

	tensor_shape ts = get_matrix_shape(a, b, transposeA, transposeB, dimZ);
	ts.assert_is_fully_defined();
	std::vector<size_t> dims = ts.as_list();
	size_t dimX = dims[0];
	size_t dimY = dims[1];
	tensor<T>* ans = new tensor<T>(std::vector<size_t>{dimX, dimY});

	const T* rawa = this->get_raw(a);
	const T* rawb = this->get_raw(b);
	T* rawr = this->get_raw(*ans);
	for (size_t y = 0; y < dimY; y++) {
		for (size_t x = 0; x < dimX; x++) {
			rawr[x+y*dimX] = 0;
			for (size_t z = 0; z < dimZ; z++) {
				size_t aidx = transposeA ? y+z*dimY : z+y*dimZ;
				size_t bidx = transposeB ? z+x*dimZ : x+z*dimX;
				rawr[x+y*dimX] += rawa[aidx] * rawb[bidx];
			}
		}
	}

	return ans;
}

template <typename T>
tensor<T>* ioperation<T>::extend_op (const tensor<T>& in, size_t index, size_t multiplier) const {
	size_t cline;
	size_t below_dim = 1;
	tensor_shape ts = change_shape(in.get_shape(), index, multiplier, below_dim, cline); // new shape
	below_dim *= cline;
	tensor<T>* ans = new tensor<T>(ts);

	size_t above_dim = in.n_elems() / below_dim;
	if (0 == above_dim) above_dim = 1; // remember that extension can increase dimensionality
	// copy over data
	T* dest_data = this->get_raw(*ans);
	const T* src_data = this->get_raw(in);
	for (size_t i = 0; i < above_dim; i++) {
		// copy data multiplier times
		const T* src_addr = src_data + i * below_dim;
		for (size_t j = 0; j < multiplier; j++) {
			T* dest_addr = dest_data + below_dim * (multiplier * i + j);
			std::memcpy(dest_addr, src_addr, below_dim * sizeof(T));
		}
	}

	return ans;
}

template <typename T>
tensor<T>* ioperation<T>::compress_op (
	const tensor<T>& in,
	signed index,
	std::function<T(const std::vector<T>&)> collector) const {
	size_t total = in.n_elems();
	if (index < 0) { // if index is negative compress all values
		const T* src_data = this->get_raw(in);
		std::vector<T> gather;
		for (size_t i = 0; i < total; i++) {
			gather.push_back(src_data[i]);
		}
		return new tensor<T>(collector(gather));
	}
	if (index >= in.n_dims()) {
		// compressing an non-existent dimension...
		return new tensor<T>(in);
	}
	size_t cline;
	size_t below_dim;
	tensor_shape ts = change_shape(in.get_shape(), index, 0, below_dim, cline); // new shape

	tensor<T>* ans = new tensor<T>(ts);

	size_t above_dim = total / (below_dim*cline);
	// copy over data
	T* dest_data = this->get_raw(*ans);
	const T* src_data = this->get_raw(in);
	for (size_t i = 0; i < above_dim; i++) {
		for (size_t j = 0; j < below_dim; j++) {
			// apply compression to each element along cline dimension
			size_t dest_idx = j + i * below_dim;
			std::vector<T> gather;
			for (size_t k = 0; k < cline; k++) {
				size_t src_idx = j + k * below_dim + i * below_dim * cline;
				gather.push_back(src_data[src_idx]);
			}
			dest_data[dest_idx] = collector(gather);
		}
	}

	return ans;
}

template <typename T>
void ioperation<T>::update (tensor_shape candidate_shape) {
	// no point in propagating if the shape is undefined
	if (0 != candidate_shape.n_dims()) {
		this->out_.set_shape(candidate_shape);
		// propagate to consumers
		for (ioperation<T>* consumer : this->consumers) {
			consumer->shape_eval();
		}
	}
}

}

#endif
