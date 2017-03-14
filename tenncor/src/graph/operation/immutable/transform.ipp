//
//  transform.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-24.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_TRANSFORM_HPP

namespace nnet
{

template <typename T>
T mean (const std::vector<T>& data)
{
	T ans = 0;
	for (T raw : data)
	{
		ans += raw;
	}
	ans /= data.size();
	return ans;
}

template <typename T>
varptr<T> transpose (const varptr<T> a)
{
	if (nullptr == a) return nullptr;
	return operation<T>::get(std::vector<inode<T>*>{a},
		[](std::vector<tensorshape> shapes)
		{
			tensorshape ts = shapes[0];
			if (ts.is_fully_defined())
			{
				// restrict shape to no greater than 2-D for now
				assert(ts.rank() <= 2);
				std::vector<size_t> inl = ts.as_list();
				if (ts.rank() == 1)
				{
					return std::vector<size_t>{1, inl[0]};
				}
				return std::vector<size_t>{inl[1], inl[0]};
			}
			return std::vector<size_t>{};
		},
		[](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
		{
			const T* src = args[0];
			// we have the new shape
			std::vector<size_t> outs = shape.as_list();
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
		[](std::vector<inode<T>*> args, variable<T>* leaf)
		{
			return transpose(varptr<T>(args.front()->get_leaf(leaf)));
		}, "transpose");
}

// fit to watch
template <typename T>
varptr<T> fit (const varptr<T> a, const varptr<T> watch)
{
	if (nullptr == a && nullptr == watch) return nullptr;
	// additional constraint that watch shape must be have shape with
	// dimensions greater or equal to a's dimensional value (shape.as_list()[i])
	return operation<T>::get(std::vector<inode<T>*>{a, watch},
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
		[a](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
		{
			const T* src = args[0];
			tensorshape oshape = a->get_shape();
			std::vector<size_t> orig = oshape.as_list();
			std::vector<size_t> tv = shape.as_list();
			size_t total = shape.n_elems();

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
		[watch](std::vector<inode<T>*> args, variable<T>* leaf)
		{
			return fit(varptr<T>(args.front()->get_leaf(leaf)), watch);
		}, "fit");
}

template <typename T>
varptr<T> extend (const varptr<T> a, size_t index, size_t multiplier)
{
	if (nullptr == a && 1 >= multiplier) return nullptr;
	return operation<T>::get(std::vector<inode<T>*>{a},
		[index, multiplier](std::vector<tensorshape> shapes)
		{
			tensorshape ts = shapes[0];
			ts.assert_is_fully_defined();
			std::vector<size_t> tv = ts.as_list();
			// allocated additional space along index
			size_t dims = ts.rank();
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
		[index, multiplier](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
		{
			const T* src = args[0];
			// REMEMBER that ts is the resulting shape, not the original shape
			// both above and below values are calculations based on the original shape
			std::vector<size_t> tv = shape.as_list();
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
			size_t above = shape.n_elems() / (multiplier * below);

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
		[index, multiplier](std::vector<inode<T>*> args, variable<T>* leaf)
		{
			return extend(varptr<T>(args.front()->get_leaf(leaf)), index, multiplier);
		}, "extend");
}

template <typename T>
varptr<T> compress (const varptr<T> a, int index,
	std::function<T(const std::vector<T>&)> collector)
{
	if (nullptr == a) return nullptr;
	FORWARD_OP<T> gatherer;
	SHAPER shaper;
	if (index >= 0)
	{
		gatherer =
		[a, index, collector](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
		{
			const T* src = args[0];
			// REMEMBER that ts is the resulting shape, not the original shape
			// both above and below values are calculations based on the original shape
			// original shape
			tensorshape orig = a->get_shape();
			assert((unsigned) index < orig.rank());
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
		};
		shaper =
		[index](std::vector<tensorshape> shapes)
		{
			tensorshape ts = shapes[0];
			ts.assert_is_fully_defined();
			assert((unsigned) index < ts.rank());
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
		};
	}
	else
	{
		gatherer =
		[collector, a](T* dest, const tensorshape& shape, std::vector<const T*>& args, std::vector<tensorshape>&)
		{
			std::vector<size_t> tv = shape.as_list();
			size_t total = shape.n_elems();
			dest[0] = collector(std::vector<T>(args[0], args[0]+total));
		};
		shaper =
		[index](std::vector<tensorshape> shapes)
		{
			return std::vector<size_t>{1};
		};
	}

	return operation<T>::get(std::vector<inode<T>*>{a}, shaper, gatherer,
		[index, collector](std::vector<inode<T>*> args, variable<T>* leaf)
		{
			return compress(varptr<T>(args.front()->get_leaf(leaf)), index, collector);
		}, "compress");
}

}

#endif
