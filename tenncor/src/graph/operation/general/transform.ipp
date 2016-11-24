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
void transform<T>::setup_gradient (void)
{
	std::vector<ivariable<T>*> args;
	for (ccoms::subject* child : this->dependencies_)
	{
		if (ivariable<T>* arg = dynamic_cast<ivariable<T>*>(child))
		{
			args.push_back(arg);
		}
	}
	this->grad_ = der_(args);
}

template <typename T>
ivariable<T>* transform<T>::clone_impl (std::string name)
{
	return new transform<T>(*this, name);
}

template <typename T>
tensorshape transform<T>::shape_eval (void)
{
	tensorshape first;
	for (ccoms::subject* sub : this->dependencies_)
	{
		if (ivariable<T>* v = dynamic_cast<ivariable<T>*>(sub))
		{
			first = shape_(v->get_shape());
		}
	}
	return first;
}

template <typename T>
transform<T>::transform (const transform<T>& other, std::string name) :
	ccoms::iobserver(other),
	ivariable<T>(other, name),
	ioperation<T>(other, name),
	collect_(other.collect_),
	shape_(other.shape_),
	der_(other.der_) {}

template <typename T>
transform<T>::transform (ivariable<T>* arg,
	std::function<void(T*,const T*,tensorshape)> op,
	std::function<tensorshape(tensorshape)> trans,
	BUILD_DERIVE<T> der, std::string name) :
	ccoms::iobserver(std::vector<ccoms::subject*>{arg}),
	ivariable<T>(std::vector<size_t>{}, name),
	ioperation<T>(std::vector<ivariable<T>*>{arg}, name),
	collect_(op), shape_(trans), der_(der)
{
	this->out_ = std::make_unique<tensor_op<T> >(
	[this](T* dest, std::vector<const T*> srcs)
	{
		tensorshape ts = shape_eval();
		collect_(dest, srcs[0], ts);
	});
	// try to update
	update(nullptr);
	if (session::pre_shape_eval())
	{
		shape_eval();
	}
}

template <typename T>
transform<T>* transform<T>::clone (std::string name)
{
	return static_cast<transform<T>*>(clone_impl(name));
}

template <typename T>
transform<T>& transform<T>::operator = (const transform<T>& other)
{
	if (this != &other)
	{
		collect_ = other.collect_;
		shape_ = other.shape_;
		der_ = other.der_;
		this->copy(other);
	}
	return *this;
}

template <typename T>
void transform<T>::update (ccoms::subject* caller)
{
	ivariable<T>* arg = dynamic_cast<ivariable<T>*>(this->dependencies_[0]);
	tensor<T> one(1);
	tensor<T>* t;
	if (nullptr == caller)
	{
		t = arg->get_eval();
	}
	else
	{
		t = arg == caller ? &one : nullptr;
	}
	
	// t is var's eval if caller is nullptr, otherwise
	// t is one if var is the caller, nullptr otherwise
	this->valid_tensor_ = nullptr != t;
	if (!this->out_->is_alloc())
	{
		this->out_->set_shape(shape_eval());
	}
	if (this->valid_tensor_)
	{
		(*this->out_)(std::vector<tensor<T>*>{t});
	}

	this->notify();
}

template <typename T>
varptr<T> clip_norm (const varptr<T> a, T cap)
{
	if (nullptr == a) return nullptr;
 	ivariable<T>* op = transform<T>::build(a,
 		[cap](T* dest, const T* src, tensorshape ts)
 		{
 			T l2norm = 0;
 			size_t n = ts.n_elems();
 			for (size_t i = 0; i < n; i++)
 			{
 				l2norm += src[i]*src[i];
 			}
			l2norm = std::sqrt(l2norm);

 			for (size_t i = 0; i < n; i++)
 			{
				dest[i] = src[i] * cap / l2norm;
 			}
 		},
 		[](tensorshape ts)
 		{
			ts.assert_is_fully_defined();
			return ts;
		},
 		[cap](std::vector<ivariable<T>*> args)
 		{
 			ivariable<T>* a = args.front();
 			return clip_norm(varptr<double>(a->get_gradient()), cap);
 		},
 	"clip_norm(" + a->get_name() + ")");
 	return op;
}

template <typename T>
varptr<T> transpose (const varptr<T> a)
{
	if (nullptr == a) return nullptr;
 	ivariable<T>* op = transform<T>::build(a,
 		[](T* dest, const T* src, tensorshape ts)
 		{
			// we have the new shape
			std::vector<size_t> inl = ts.as_list();
			// old dimensions
			size_t dimX = inl[1]; size_t dimY = inl[0];
			// not in place so x = y+1 doesn't work
			for (size_t y = 0; y < dimY; y++)
			{
				for (size_t x = 0; x < dimX; x++)
				{
					dest[y+x*dimY] = src[x+y*dimX];
				}
			}
 		},
 		[](tensorshape ts)
 		{
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
 			return transpose(a->get_gradient());
 		},
 	"transpose(" + a->get_name() + ")");
 	return op;
}

// fit to watch
template <typename T>
varptr<T> fit (const varptr<T> a, const varptr<T> watch)
{
	if (nullptr == a && nullptr == watch) return nullptr;
	// additional constraint that watch shape must be have shape with
	// dimensions greater or equal to a's dimensional value (shape.as_list()[i])
 	ivariable<T>* op = transform<T>::build(a,
 		[watch, a](T* dest, const T* src, tensorshape ts)
 		{
			std::vector<size_t> orig = a->get_shape().as_list();
			std::vector<size_t> tv = ts.as_list();
			size_t total = ts.n_elems();

			T temp[ts.n_elems()];
			T temp2[ts.n_elems()];

			const T* super_src = src;
			T* super_dest = temp;
			size_t below_dim = 1;

			for (size_t index = 0; index < tv.size(); index++)
			{
				below_dim *= tv[index];
				size_t mult = 0;
				if (index < orig.size())
				{
					assert(orig[index] <= tv[index]);
					if (0 == orig[index] % tv[index])
					{
						mult = tv[index] / orig[index];
					}
				}
				else
				{
					mult = tv[index];
				}
				if (mult)
				{
					size_t below = below_dim * tv[index] / mult;
					size_t above = total / below;

					// copy over data
					const T* src_addr = super_src;
					for (size_t i = 0; i < above; i++)
					{
						// copy data mult times
						src_addr += i * below;
						for (size_t j = 0; j < mult; j++)
						{
							T* dest_addr = super_dest + below * (mult * i + j);
							std::memcpy(dest_addr, src_addr, below * sizeof(T));
						}
					}
					// state update: below_dim, super_src, and super_dest
					below_dim *= mult;
					if (super_src == temp)
					{
						super_src = temp2;
						super_dest = temp;
					}
					else
					{
						super_src = temp;
						super_dest = temp2;
					}
				}
			}
			std::memcpy(dest, super_dest, total * sizeof(T));
 		},
 		[watch](tensorshape ts)
 		{
			ts.assert_is_fully_defined();
			tensorshape s = watch->get_shape();
			assert(s.n_elems() >= ts.n_elems());
			return s;
		},
 		[watch](std::vector<ivariable<T>*> args)
 		{
 			ivariable<T>* a = args.front();
 			return fit(varptr<double>(a->get_gradient()), watch);
 		},
 	nnutils::formatter() << "fit[" << watch->get_name() <<  "](" << a->get_name() + ")");
 	return op;
}

template <typename T>
varptr<T> extend (const varptr<T> a, size_t index, size_t multiplier)
{
	if (nullptr == a && 1 >= multiplier) return nullptr;
 	ivariable<T>* op = transform<T>::build(a,
 		[index, multiplier](T* dest, const T* src, tensorshape ts)
 		{
			std::vector<size_t> tv = ts.as_list();
 			size_t below = 1;
 			for (size_t i = 0; i <= index; i++)
 			{
 				below *= tv[i];
 			}
 			below *= tv[index] / multiplier;

			size_t above = ts.n_elems() / below;
			// copy over data
			const T* src_addr = src;
			for (size_t i = 0; i < above; i++)
			{
				// copy data multiplier times
				src_addr += i * below;
				for (size_t j = 0; j < multiplier; j++)
				{
					const T* dest_addr = dest + below * (multiplier * i + j);
					std::memcpy(dest_addr, src_addr, below * sizeof(T));
				}
			}
 		},
 		[index, multiplier](tensorshape ts)
 		{
			ts.assert_is_fully_defined();
			std::vector<size_t> tv = ts.as_list();
			// allocated additional space along index
			bool shape_changed = false;
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
 			return extend(a->get_gradient(), index, multiplier);
 		},
 	nnutils::formatter() << "extend[" << index << "," <<
 		multiplier << "](" << a->get_name() + ")");
 	return op;
}

template <typename T>
varptr<T> compress (const varptr<T> a, int index,
	std::function<T(const std::vector<T>&)> collector)
{
	if (nullptr == a) return nullptr;
	std::function<void(T*,const T*,tensorshape)> gatherer;
	std::function<tensorshape(tensorshape)> shaper;
	if (index >= 0)
	{
		gatherer = std::function<void(T*,const T*,tensorshape)>(
		[index, collector, a](T* dest, const T* src, tensorshape ts)
		{
			assert(index < ts.n_dims());
			std::vector<size_t> tv = ts.as_list();
			tensorshape orig = a->get_shape();
			size_t idx_val = orig.as_list()[index];
			size_t below = 1;
			for (size_t i = 0; i <= index; i++)
			{
				below *= tv[i];
			}
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
		shaper = std::function<tensorshape(tensorshape)>(
		[index](tensorshape ts)
		{
			ts.assert_is_fully_defined();
			assert(index < ts.n_dims());
			std::vector<size_t> tv = ts.as_list();
			if (0 == index)
			{ // pop front
				tv.front() = std::move(tv.back());
				tv.pop_back();
			}
			else if (tv.size()-1 == index)
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
		gatherer = std::function<void(T*,const T*,tensorshape)>(
		[collector, a](T* dest, const T* src, tensorshape ts)
		{
			std::vector<size_t> tv = ts.as_list();
			size_t total = ts.n_elems();
			dest[0] = collector(std::vector<T>(dest, dest+total));
		});
		shaper = std::function<tensorshape(tensorshape)>(
		[index](tensorshape ts)
		{
			return std::vector<size_t>{1};
		});
	}

 	ivariable<T>* op = transform<T>::build(a, gatherer, shaper,
 		[index, collector](std::vector<ivariable<T>*> args)
 		{
 			ivariable<T>* a = args.front();
 			return compress(varptr<T>(a->get_gradient()), index, collector);
 		},
 	nnutils::formatter() << "compress[" << index << "](" << a->get_name() + ")");
 	return op;
}

}

#endif