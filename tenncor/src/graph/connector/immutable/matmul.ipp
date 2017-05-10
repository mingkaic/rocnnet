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
matmul<T>* matmul<T>::clone (void) const
{
	return static_cast<matmul<T>*>(clone_impl());
}

template <typename T>
matmul<T>* matmul<T>::move (void)
{
	return static_cast<matmul<T>*>(move_impl());
}

template <typename T>
matmul<T>::matmul (inode<T>* a, inode<T>* b,
	bool transposeA, bool transposeB) :
immutable<T>((std::vector<inode<T>*>{a, b}),
[transposeA, transposeB](std::vector<tensorshape> shapes) -> tensorshape
{
	tensorshape& t1s = shapes[0];
	tensorshape& t2s = shapes[1];

	if (5 > (t1s.rank() + t2s.rank()))
	{
		std::vector<size_t> al = t1s.as_list();
		std::vector<size_t> bl = t2s.as_list();

		size_t ax = t1s.rank() ? al[0] : 0;
		size_t ay = t1s.rank() > 1 ? al[1] : 1;
		size_t bx = t2s.rank() ? bl[0] : 0;
		size_t by = t2s.rank() > 1 ? bl[1] : 1;

		if (ay == bx && transposeA && transposeB)
		{
			return std::vector<size_t>{by, ax};
		}
		else if (ay == by && transposeA)
		{
			return std::vector<size_t>{bx, ax};
		}
		else if (ax == bx && transposeB)
		{
			return std::vector<size_t>{by, ay};
		}
		else if (ax == by && !transposeA && !transposeB)
		{
			return std::vector<size_t>{bx, ay};
		}
	}
	// warn user
	std::cout << "warning: matmul shapes do not match" << std::endl;
	return tensorshape();
},
[transposeA, transposeB](T* out, const tensorshape& outs,
	std::vector<const T*>& in, std::vector<tensorshape>& inshapes)
{
	outs.assert_is_fully_defined();
	std::vector<size_t> dims = outs.as_list();
	size_t dimX = dims[0]; size_t dimY = dims[1];

	std::vector<size_t> t = inshapes[0].as_list();
	size_t dimZ = t[0];
	if (transposeA)
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
				size_t aidx = transposeA ? y+z*dimY : z+y*dimZ;
				size_t bidx = transposeB ? z+x*dimZ : x+z*dimX;
				out[x+y*dimX] += rawa[aidx] * rawb[bidx];
			}
		}
	}
},
// todo: remove this from capture (if we copy this, this still refers to original node)
// meaning deleting original breaks the copy
[this](std::vector<inode<T>*>, variable<T>*)
{
	return this->one.get();
}, "matmul")
{
	auto jtrans = [a, b, transposeA, transposeB](
		inode<T>* root, variable<T>* wrt) -> inode<T>*
	{
		varptr<T> grada = a->get_leaf(wrt);
		varptr<T> gradb = b->get_leaf(wrt);

		if (grada->good_status() && *grada == (T)0 &&
			gradb->good_status() && *gradb == (T)0)
		{
			return root;
		}

		varptr<T> mA = new matmul<T>(root, b, transposeA, !transposeB);
		varptr<T> mB = new matmul<T>(a, root, !transposeA, transposeB);
		return mA * grada + mB * gradb;
	};

	typename inode<T>::GRAD_CACHE leaves;
	this->get_leaves(leaves);
	for (auto leaf_pair : leaves)
	{
		variable<T>* leaf = leaf_pair.first;
		this->jacobians_[leaf].list_.push_back(jtrans);
	}
}

template <typename T>
inode<T>* matmul<T>::clone_impl (void) const
{
	return new matmul<T>(*this);
}

template <typename T>
inode<T>* matmul<T>::move_impl (void)
{
	return new matmul<T>(std::move(*this));
}

}

#endif