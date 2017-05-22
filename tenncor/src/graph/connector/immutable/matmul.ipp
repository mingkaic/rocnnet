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

	std::vector<size_t> al = t1s.as_list();
	std::vector<size_t> bl = t2s.as_list();
	size_t rank1 = t1s.rank();
	size_t rank2 = t2s.rank();

	size_t ax = rank1 ? al[0] : 0;
	size_t ay = rank1 > 1 ? al[1] : 1;
	size_t bx = rank1 ? bl[0] : 0;
	size_t by = rank1 > 1 ? bl[1] : 1;

	// ensure the dimensions beyond 2d are equal
	size_t minend = std::min(rank1, rank2);
	std::vector<size_t> beyond2d;
	if (minend > 2)
	{
		auto ait = al.begin()+2;
		auto aet = al.begin()+minend;
		if (std::equal(ait, aet, bl.begin()+2))
		{
			beyond2d.insert(beyond2d.end(), ait, aet);
		}
		else
		{
			std::stringstream ss;
			ss << "attempting to matrix multiple shapes ";
			print_shape(t1s, ss);
			ss << " and ";
			print_shape(t2s, ss);
			throw std::logic_error(ss.str());
		}
		// check that remaining shape values are ones,
		// otherwise one shape is larger than the other
		auto it = rank1 > rank2 ? al.begin() : bl.begin();
		auto et = rank1 > rank2 ? al.end() : bl.end();
		if (!std::all_of(it + minend, et, [](size_t e) { return e == 1; }))
		{
			std::stringstream ss;
			ss << "attempting to matrix multiple different sized shapes ";
			print_shape(t1s, ss);
			ss << " and ";
			print_shape(t2s, ss);
			throw std::logic_error(ss.str());
		}
	}

	std::vector<size_t> res_shape;
	if (ay == bx && transposeA && transposeB)
	{
		res_shape = {by, ax};
	}
	else if (ay == by && transposeA)
	{
		res_shape = {bx, ax};
	}
	else if (ax == bx && transposeB)
	{
		res_shape = {by, ay};
	}
	else if (ax == by && !transposeA && !transposeB)
	{
		res_shape = {bx, ay};
	}
	else
	{
		// warn user (do not throw, this may be fixable)
		std::cout << "warning: matmul shapes do not match" << std::endl;
		return tensorshape();
	}

	res_shape.insert(res_shape.end(), beyond2d.begin(), beyond2d.end());
	return res_shape;
},
[transposeA, transposeB](T* out, const tensorshape& outs,
	std::vector<const T*>& in, std::vector<tensorshape>& inshapes)
{
	outs.assert_is_fully_defined();
	std::vector<size_t> dims = outs.as_list();
	size_t dimX = dims[0]; size_t dimY = dims[1];

	std::vector<size_t> t = inshapes[0].as_list();
	size_t shared_dim = t[0];
	if (transposeA)
	{
		shared_dim = t.size() > 1 ? t[1] : 1;
	}

	size_t nbeyond2d = std::accumulate(dims.begin()+2, dims.end(),
		(size_t) 1, std::multiplies<size_t>());
	size_t nmat = dimX * dimY;
	for (size_t i = 0; i < nbeyond2d; i++)
	{
		const T* rawa = in[0] + i * nmat;
		const T* rawb = in[1] + i * nmat;
		for (size_t y = 0; y < dimY; y++)
		{
			for (size_t x = 0; x < dimX; x++)
			{
				out[x+y*dimX] = 0;
				for (size_t z = 0; z < shared_dim; z++)
				{
					size_t aidx = transposeA ? y+z*dimY : z+y*shared_dim;
					size_t bidx = transposeB ? z+x*shared_dim : x+z*dimX;
					out[x+y*dimX] += rawa[aidx] * rawb[bidx];
				}
			}
		}
	}
},
[](std::vector<inode<T>*>, variable<T>*)
{
	return get_shared_one<T>();
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