//
//  matmul.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-08.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef matop_hpp

namespace nnet
{

// MATRIX MULTIPLICATION

template <typename T>
size_t matmul<T>::common_dim (void) const
{
	std::vector<size_t> t = dynamic_cast<ivariable<T>*>(
		this->dependencies_[0])->get_shape().as_list();
	if (transposeA_)
	{
		return 2 == t.size() ? t[1] : 1;
	}
	return t[0];
}

template <typename T>
void matmul<T>::setup_gradient (void)
{
	// matmul'(f, g) = jacobian(f, g)
	ivariable<T>* arga = dynamic_cast<ivariable<T>*>(this->dependencies_[0]);
	ivariable<T>* argb = dynamic_cast<ivariable<T>*>(this->dependencies_[1]);
	assert(arga && argb);
	this->grad_ = jacobian<T>::build(arga, argb, transposeA_, transposeB_, "J(" + this->get_name() + ")");
}

template <typename T>
tensorshape matmul<T>::shape_eval (void)
{
	tensorshape t1s = dynamic_cast<ivariable<T>*>(this->dependencies_[0])->get_shape();
	tensorshape t2s = dynamic_cast<ivariable<T>*>(this->dependencies_[1])->get_shape();

	if (5 > (t1s.n_dims() + t2s.n_dims()))
	{
		std::vector<size_t> al = t1s.as_list();
		std::vector<size_t> bl = t2s.as_list();

		size_t ax = t1s.n_dims() ? al[0] : 0;
		size_t ay = t1s.n_dims() > 1 ? al[1] : 1;
		size_t bx = t2s.n_dims() ? bl[0] : 0;
		size_t by = t2s.n_dims() > 1 ? bl[1] : 1;

		if (ay == bx && transposeA_ && transposeB_)
		{
			return std::vector<size_t>{by, ax};
		}
		else if (ay == by && transposeA_)
		{
			return std::vector<size_t>{bx, ax};
		}
		else if (ax == bx && transposeB_)
		{
			return std::vector<size_t>{by, ay};
		}
		else if (ax == by && !transposeA_ && !transposeB_)
		{
			return std::vector<size_t>{bx, ay};
		}
	}
	return std::vector<size_t>{};
}

template <typename T>
matmul<T>::matmul (const matmul<T>& other, std::string name) :
	ioperation<T>(other, name),
	transposeA_(other.transposeA_),
	transposeB_(other.transposeB_) {}

template <typename T>
ivariable<T>* matmul<T>::clone_impl (std::string name) {
	return new matmul<T>(*this, name);
}

template <typename T>
matmul<T>::matmul (ivariable<T>* a, ivariable<T>* b,
	bool transposeA, bool transposeB) :
	ioperation<T>((std::vector<ivariable<T>*>{a, b}),
	"(" + a->get_name() + "•" + b->get_name() + ")"),
	transposeA_(transposeA), transposeB_(transposeB)
{
	this->out_ = std::make_unique<tensor_op<T> >(
	[this](T*& dest, std::vector<const T*> srcs)
	{
		tensorshape ts = shape_eval();
		ts.assert_is_fully_defined();
		std::vector<size_t> dims = ts.as_list();
		size_t dimX = dims[0]; size_t dimY = dims[1];
		size_t dimZ = common_dim();

		const T* rawa = srcs[0];
		const T* rawb = srcs[1];
		for (size_t y = 0; y < dimY; y++)
		{
			for (size_t x = 0; x < dimX; x++)
			{
				dest[x+y*dimX] = 0;
				for (size_t z = 0; z < dimZ; z++)
				{
					size_t aidx = transposeA_ ? y+z*dimY : z+y*dimZ;
					size_t bidx = transposeB_ ? z+x*dimZ : x+z*dimX;
					dest[x+y*dimX] += rawa[aidx] * rawb[bidx];
				}
			}
		}
	});
}

template <typename T>
matmul<T>* matmul<T>::clone (std::string name)
{
	return static_cast<matmul<T>*>(clone_impl(name));
}

template <typename T>
matmul<T>& matmul<T>::operator = (const ivariable<T>& other)
{
	if (this != &other)
	{
		if (const matmul<T>* mptr = dynamic_cast<const matmul<T>*>(&other))
		{
			transposeA_ = mptr->transposeA_;
			transposeB_ = mptr->transposeB_;
		}
		this->copy(other);
	}
	return *this;
}

template <typename T>
void matmul<T>::update (ccoms::subject* caller)
{
	// caller is never used because we know matmul will never be the parent of a gradient leaf
	ivariable<T>* a = dynamic_cast<ivariable<T>*>(this->dependencies_[0]);
	ivariable<T>* b = dynamic_cast<ivariable<T>*>(this->dependencies_[1]);
	tensor<T>* at = a->get_eval();
	tensor<T>* bt = b->get_eval();

	this->valid_tensor_ = at && bt;
	if (this->valid_tensor_)
	{
		this->out_->set_shape(shape_eval());
		*(this->out_)(std::vector<tensor<T>*>{at, bt});
	}

	this->notify();
}

}

#endif