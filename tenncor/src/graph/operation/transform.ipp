//
//  transform.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-09.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef transform_hpp

namespace nnet
{

// Collect Operations

template <typename T>
ivariable<T>* transform<T>::setup_gradient (void)
{
	std::vector<ivariable<T>*> args;
	for (ccoms::subject* child : this->dependencies_)
	{
		if (ivariable<T>* arg = sub_to_var<T>(child))
		{
			args.push_back(arg);
		}
	}
	return der_(args);
}

template <typename T>
transform<T>::transform (std::vector<ivariable<T>*> args,
	TEN_OP<T> op, SHAPE trans, BUILD_DERIVE<T> der, std::string name) :
	der_(der),
	ioperation<T>(args, name)
{
	this->shaper_ = trans; // used in shape_eval
	this->out_ = std::make_unique<tensor_op<T> >(op, trans);
	// try to update
	if (session::pre_shape_eval())
	{
		this->shape_eval().assert_is_fully_defined();
	}
	this->initialize();
}

template <typename T>
transform<T>* transform<T>::clone (void)
{
	return new transform<T>(*this);
}

template <typename T>
varptr<T> transpose (const varptr<T> a)
{
	if (nullptr == a) return nullptr;
	ivariable<T>* op = transform<T>::build(std::vector<ivariable<T>*>{a},
	[](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		const T* src = args[0];
		// we have the new shape
		std::vector<size_t> outs = info.res_shape_.as_list();
		// old dimensions
		size_t dimX = outs[1]; size_t dimY = outs[0];
		// not in place so x = y+1 doesn't work
		for (size_t y = 0; y < dimY; y++)
		{
			for (size_t x = 0; x < dimX; x++)
			{
				dest[y+x*dimY] = src[x+y*dimX];
			}
		}
	},
	[](std::vector<tensorshape> shapes)
	{
		tensorshape ts = shapes[0];
		if (ts.is_fully_defined())
		{
			// restrict shape to no greater than 2-D for now
			assert(ts.n_dims() <= 2);
			std::vector<size_t> inl = ts.as_list();
			if (ts.n_dims() == 1)
			{
				return std::vector<size_t>{1, inl[0]};
			}
			return std::vector<size_t>{inl[1], inl[0]};
		}
		return std::vector<size_t>{};
	},
	[](std::vector<ivariable<T>*> args)
	{
		ivariable<T>* a = args.front();
		return transpose(varptr<T>(a->get_gradient()));
	}, "transpose");
	return op;
}

// fit to watch
template <typename T>
varptr<T> fit (const varptr<T> a, const varptr<T> watch)
{
	if (nullptr == a && nullptr == watch) return nullptr;
	// additional constraint that watch shape must be have shape with
	// dimensions greater or equal to a's dimensional value (shape.as_list()[i])
	ivariable<T>* op = transform<T>::build(std::vector<ivariable<T>*>{a, watch},
	[](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		const T* src = args[0];
		tensorshape oshape = info.arg_shape_[0];
		std::vector<size_t> orig = oshape.as_list();
		std::vector<size_t> tv = info.res_shape_.as_list();
		size_t total = info.res_shape_.n_elems();

		T temp[total];
		T temp2[total];

		const T* super_src = src;
		T* super_dest = temp;
		size_t super_below = 1;
		size_t ototal = oshape.n_elems(); // old total

		for (size_t index = 0; index < tv.size(); index++)
		{
			size_t mult = 0;
			if (index < orig.size())
			{
				// original dimension must be equal or less than the result dimension
				assert(orig[index] <= tv[index]);
				if (0 == tv[index] % orig[index])
				{
					mult = tv[index] / orig[index];
					ototal *= mult;
				}
				else
				{
					// TODO: dimension expansion doesn't match nicely, implement later
					throw std::bad_function_call();
				}
			}
			else
			{
				mult = tv[index];
			}

			// below calculates all elements encompassed up to the index dimension
			// that is for a shape of <1, 2, 3, 4> and index 2
			// below = 1 * 2 * 3 = 6
			size_t below = super_below * tv[index] / mult;
			// above calculates the number of tensors (of index rank) within the original tensor
			// that is for a shape of <1, 2, 3, 4> and index 2
			// the tensors of index rank is represented by the first 3 dimensions <1, 2, 3>
			// the overall tensor is represented as a tensor of tensor < <1, 2, 3>, 4>
			// above is 4
			// above = original total / below
			// original total = resulting total / multiplier
			// expand original across resulting dimension
			size_t above = total / (mult * below);

			// copy over data
			size_t src_idx = 0;
			size_t dest_idx = 0;
			for (size_t i = 0; i < above; i++)
			{
				src_idx = i * below;
				// copy data mult times
				for (size_t j = 0; j < mult; j++)
				{
					dest_idx = below * (mult * i + j);
					std::memcpy(super_dest + dest_idx, super_src + src_idx, below * sizeof(T));
				}
			}
			// state update: below_dim, super_src, and super_dest
			super_below *= tv[index];

			// swap super buffers as long as it's not the last one
			if (index < tv.size()-1)
			{
				if (super_src == temp) {
					super_src = temp2;
					super_dest = temp;
				} else {
					super_src = temp;
					super_dest = temp2;
				}
			}
		}
		std::memcpy(dest, super_dest, total * sizeof(T));
	},
	[](std::vector<tensorshape> shapes)
	{
		tensorshape orig = shapes[0];
		tensorshape watchshape = shapes[1]; // watch is always argument 2
		if (watchshape.is_fully_defined()
			&& watchshape.n_elems() < orig.n_elems())
		{
			return tensorshape();
		}
		return watchshape;
	},
	[watch](std::vector<ivariable<T>*> args)
	{
		ivariable<T>* a = args.front();
		return fit(varptr<T>(a->get_gradient()), watch);
	}, "fit");
	return op;
}

template <typename T>
varptr<T> extend (const varptr<T> a, size_t index, size_t multiplier)
{
	if (nullptr == a && 1 >= multiplier) return nullptr;
	ivariable<T>* op = transform<T>::build(std::vector<ivariable<T>*>{a},
	[index, multiplier](shapeinfo info, T* dest, std::vector<const T*> args)
	{
		const T* src = args[0];
		// REMEMBER that ts is the resulting shape, not the original shape
		// both above and below values are calculations based on the original shape
		std::vector<size_t> tv = info.res_shape_.as_list();
		// below calculates all elements encompassed up to the index dimension
		// that is for a shape of <1, 2, 3, 4> and index 2
		// below = 1 * 2 * 3 = 6
		size_t below = 1;
		for (size_t i = 0; i < index; i++)
		{
			below *= tv[i];
		}
		// we know that for the resulting shape, the dimensional-value at index is multiplied by multiplier
		// so to obtain the original dimension, we divide by multiplier
		below *= tv[index] / multiplier;
		// above calculates the number of tensors (of index rank) within the original tensor
		// that is for a shape of <1, 2, 3, 4> and index 2
		// the tensors of index rank is represented by the first 3 dimensions <1, 2, 3>
		// the overall tensor is represented as a tensor of tensor < <1, 2, 3>, 4>
		// above is 4
		// above = original total / below
		// original total = resulting total / multiplier
		size_t above = info.res_shape_.n_elems() / (multiplier * below);

		// copy over data
		for (size_t i = 0; i < above; i++)
		{
			// copy data multiplier times
			const T* src_addr = src + i * below;
			for (size_t j = 0; j < multiplier; j++)
			{
				T* dest_addr = dest + below * (multiplier * i + j);
				std::memcpy(dest_addr, src_addr, below * sizeof(T));
			}
		}
	},
	[index, multiplier](std::vector<tensorshape> shapes)
	{
		tensorshape ts = shapes[0];
		ts.assert_is_fully_defined();
		std::vector<size_t> tv = ts.as_list();
		// allocated additional space along index
		size_t dims = ts.n_dims();
		if (index >= dims)
		{
			// extending extra dimensions
			size_t extra_dims = index - dims;
			if (extra_dims)
			{
				tv.insert(tv.end(), extra_dims, 1);
			}
			tv.push_back(multiplier);
		}
		else
		{
			tv[index] *= multiplier;
		}
		return tv;
	},
	[index, multiplier](std::vector<ivariable<T>*> args)
	{
		ivariable<T>* a = args.front();
		return extend(varptr<T>(a->get_gradient()), index, multiplier);
	}, "extend");
	return op;
}

template <typename T>
varptr<T> compress (const varptr<T> a, int index,
	std::function<T(const std::vector<T>&)> collector)
{
	if (nullptr == a) return nullptr;
	TEN_OP<T> gatherer;
	SHAPE shaper;
	if (index >= 0)
	{
		gatherer = TEN_OP<T>(
		[index, collector](shapeinfo info, T* dest, std::vector<const T*> args)
		{
			const T* src = args[0];
			// REMEMBER that ts is the resulting shape, not the original shape
			// both above and below values are calculations based on the original shape
			// original shape
			tensorshape orig = info.arg_shape_[0];
			assert((unsigned) index < orig.n_dims());
			std::vector<size_t> tv = orig.as_list();
			size_t idx_val = tv[index];
			// below for compression calculates all elements below the index dimension
			// that is for a shape of <1, 2, 3, 4> and index 2
			// below = 1 * 2 = 2
			size_t below = 1;
			for (int i = 0; i < index; i++)
			{
				below *= tv[i];
			}
			// above denotes the same above as the one in extend
			size_t above = orig.n_elems() / (below*idx_val);

			// copy over data
			for (size_t i = 0; i < above; i++)
			{
				for (size_t j = 0; j < below; j++)
				{
					// apply compression to each element along idx_val dimension
					size_t dest_idx = j + i * below;
					std::vector<T> gather;
					for (size_t k = 0; k < idx_val; k++)
					{
						gather.push_back(src[j + k * below + i * below * idx_val]);
					}
					dest[dest_idx] = collector(gather);
				}
			}
		});
		shaper = SHAPE(
		[index](std::vector<tensorshape> shapes)
		{
			tensorshape ts = shapes[0];
			ts.assert_is_fully_defined();
			assert((unsigned) index < ts.n_dims());
			std::vector<size_t> tv = ts.as_list();
			if (0 == index)
			{ // pop front
				tv.front() = std::move(tv.back());
				tv.pop_back();
			}
			else if (tv.size()-1 == (unsigned) index)
			{
				tv.pop_back();
			}
			else
			{
				tv[index] = 1;
			}
			return tv;
		});
	}
	else
	{
		gatherer = TEN_OP<T>(
		[collector, a](shapeinfo info, T* dest, std::vector<const T*> args)
		{
			std::vector<size_t> tv = info.res_shape_.as_list();
			size_t total = info.res_shape_.n_elems();
			dest[0] = collector(std::vector<T>(args[0], args[0]+total));
		});
		shaper = SHAPE(
		[index](std::vector<tensorshape> shapes)
		{
			return std::vector<size_t>{1};
		});
	}

	ivariable<T>* op = transform<T>::build(std::vector<ivariable<T>*>{a}, gatherer, shaper,
		[index, collector](std::vector<ivariable<T>*> args)
		{
			ivariable<T>* a = args.front();
			return compress(varptr<T>(a->get_gradient()), index, collector);
		}, "compress");
	return op;
}

}

#endif