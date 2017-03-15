//
//  matmul.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-08.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_MATMUL_HPP

namespace nnet
{

template <typename T>
matmul<T>* matmul<T>::get (inode<T>* a, inode<T>* b,
	bool transposeA, bool transposeB)
{
	if (a && b)
	{
		return new matmul<T>(a, b, transposeA, transposeB);
	}
	return nullptr;
}

template <typename T>
matmul<T>* matmul<T>::get (matmul<T>&& other)
{
	return new matmul<T>(std::move(other));
}

template <typename T>
matmul<T>* matmul<T>::clone (void)
{
	return static_cast<matmul<T>*>(clone_impl());
}

template <typename T>
matmul<T>& matmul<T>::operator = (const matmul<T>& other)
{
	if (this == &other)
	{
		transposeA_ = other.transposeA_;
		transposeB_ = other.transposeB_;
		operation<T>::operator = (other);
	}
	return *this;
}

template <typename T>
matmul<T>& matmul<T>::operator = (matmul<T>&& other)
{
	if (this == &other)
	{
		transposeA_ = std::move(other.transposeA_);
		transposeB_ = std::move(other.transposeB_);
		operation<T>::operator = (other);
	}
	return *this;
}

template <typename T>
matmul<T>::matmul (inode<T>* a, inode<T>* b,
	   bool transposeA, bool transposeB) :
	operation<T>((std::vector<inode<T>*>{a, b}),
[this](std::vector<tensorshape> shapes) -> tensorshape
{
	tensorshape t1s = shapes[0];
	tensorshape t2s = shapes[1];

	if (5 > (t1s.rank() + t2s.rank()))
	{
		std::vector<size_t> al = t1s.as_list();
		std::vector<size_t> bl = t2s.as_list();

		size_t ax = t1s.rank() ? al[0] : 0;
		size_t ay = t1s.rank() > 1 ? al[1] : 1;
		size_t bx = t2s.rank() ? bl[0] : 0;
		size_t by = t2s.rank() > 1 ? bl[1] : 1;

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
	return tensorshape();
},
[this](T* out, const tensorshape& outs, std::vector<const T*>& in, std::vector<tensorshape>&)
{
	outs.assert_is_fully_defined();
	std::vector<size_t> dims = outs.as_list();
	size_t dimX = dims[0]; size_t dimY = dims[1];

	std::vector<size_t> t = static_cast<inode<T>*>(this->dependencies_[0])->get_shape().as_list();
	size_t dimZ = t[0];
	if (transposeA_)
	{
		dimZ = 2 == t.size() ? t[1] : 1;
	}

	const T* rawa = in[0];
	const T* rawb = in[1];
	for (size_t y = 0; y < dimY; y++)
	{
		for (size_t x = 0; x < dimX; x++)
		{
			out[x+y*dimX] = 0;
			for (size_t z = 0; z < dimZ; z++)
			{
				size_t aidx = transposeA_ ? y+z*dimY : z+y*dimZ;
				size_t bidx = transposeB_ ? z+x*dimZ : x+z*dimX;
				out[x+y*dimX] += rawa[aidx] * rawb[bidx];
			}
		}
	}
},
[this](std::vector<inode<T>*> args, variable<T>* leaf)
{
	return this;
}, "matmul"),
	transposeA_(transposeA), transposeB_(transposeB)
{
	this->jacobians_.push_back(
	[a, b, transposeA, transposeB](inode<T>* root, variable<T>* wrt)
	{
		varptr<T> grada = a->get_leaf(wrt);
		varptr<T> gradb = b->get_leaf(wrt);

		varptr<T> mA = new matmul<T>(root, b, transposeA, !transposeB);
		varptr<T> mB = new matmul<T>(a, root, !transposeA, transposeB);

		return mA * grada + mB * gradb;
	});
}

template <typename T>
matmul<T>::matmul (const matmul<T>& other) :
	transposeA_(other.transposeA_),
	transposeB_(other.transposeB_),
	operation<T>(other) {}

template <typename T>
matmul<T>::matmul (matmul<T>&& other) :
	operation<T>(std::move(other)),
	transposeA_(std::move(other.transposeA_)),
	transposeB_(std::move(other.transposeB_)) {}

template <typename T>
inode<T>* matmul<T>::clone_impl (void) const
{
	return new matmul<T>(*this);
}

}

#endif