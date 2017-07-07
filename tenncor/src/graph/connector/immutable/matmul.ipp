//
//  matmul.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-08.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef TENNCOR_MATMUL_HPP

#define STRASSEN_THRESHOLD 256

namespace nnet
{

inline size_t min_pad(size_t insize)
{
	size_t counter = 0;
	while (insize > STRASSEN_THRESHOLD)
	{
		insize++;
		insize >>= 1;
		counter ++;
	}
	return insize << counter;
}

template <typename T>
static void cubic_mul (T* c, const T* a, const T* b, size_t dimX, size_t dimY, size_t dimZ, size_t coord_map[8])
{
	for (size_t y = 0; y < dimY; y++)
	{
		for (size_t x = 0; x < dimX; x++)
		{
			c[x+y*dimX] = 0;
			for (size_t z = 0; z < dimZ; z++)
			{
				size_t aidx = coord_map[0] * y + coord_map[1] * z;
				size_t bidx = coord_map[2] * x + coord_map[3] * z;
				c[x + y * dimX] += a[aidx] * b[bidx];
			}
		}
	}
}

template <typename T>
static void strassen (T* c, const T* a, const T* b, size_t dimPad)
{
	if (dimPad <= STRASSEN_THRESHOLD)
	{
		size_t coord_map[4] = {dimPad, 1, 1, dimPad};
		return cubic_mul(c, a, b, dimPad, dimPad, dimPad, coord_map);
	}
	size_t quadRC = dimPad/2;
	size_t quadSize = quadRC*quadRC;
	// first 14 represent M1L to M7R
	// quadrant index 14 to 20 are represent M1 to M7
	T temp[quadSize * 21];
	memset(temp, 0, quadSize * 21 * sizeof(T));
	// M1L = A11 + A22	(0)
	// M1R = B11 + B22	(1)
	// M2L = A21 + A22	(2)
	// M2R = B11		(3)
	// M3L = A11		(4)
	// M3R = B12 - B22	(5)
	// M4L = A22		(6)
	// M4R = B21 - B11	(7)
	// M5L = A11 + A12	(8)
	// M5R = B22		(9)
	// M6L = A21 - A11	(10)
	// M6R = B11 + B12	(11)
	// M7L = A12 - A22	(12)
	// M7R = B21 + B22	(13)
	for (size_t x = 0; x < quadRC; x++)
	{
		for (size_t y = 0; y < quadRC; y++)
		{
			size_t quadidx = x + quadRC * y;

			// 11
			size_t idx11 = x + dimPad * y;

			// A11 used in M1L, M3L, M5L, M6L
			temp[quadidx] += a[idx11];
			temp[4 * quadSize + quadidx] += a[idx11];
			temp[8 * quadSize + quadidx] += a[idx11];
			temp[10 * quadSize + quadidx] -= a[idx11];

			// B11 used in M1R, M2R, M4R, M6R
			temp[quadSize + quadidx] += b[idx11];
			temp[3 * quadSize + quadidx] += b[idx11];
			temp[7 * quadSize + quadidx] -= b[idx11];
			temp[11 * quadSize + quadidx] += b[idx11];

			// 12
			size_t idx12 = x + quadRC + dimPad * y;

			// A12 used in M5L, M7L
			temp[8 * quadSize + quadidx] += a[idx12];
			temp[12 * quadSize + quadidx] += a[idx12];

			// B12 used in M3R, M6R
			temp[5 * quadSize + quadidx] += b[idx12];
			temp[11 * quadSize + quadidx] += b[idx12];

			// 21
			size_t idx21 = x + dimPad * (y + quadRC);

			// A21 used in M2L, M6L
			temp[2 * quadSize + quadidx] += a[idx21];
			temp[10 * quadSize + quadidx] += a[idx21];

			// B21 used in M4R, M7R
			temp[7 * quadSize + quadidx] += b[idx21];
			temp[13 * quadSize + quadidx] += b[idx21];

			// 22
			size_t idx22 = x + quadRC + dimPad * (y + quadRC);

			// A22 used in M1L, M2L, M4L, M7L
			temp[quadidx] += a[idx22];
			temp[2 * quadSize + quadidx] += a[idx22];
			temp[6 * quadSize + quadidx] += a[idx22];
			temp[12 * quadSize + quadidx] -= a[idx22];

			// B22 used in M1R, M3R, M5R, M7R
			temp[quadSize + quadidx] += b[idx22];
			temp[5 * quadSize + quadidx] -= b[idx22];
			temp[9 * quadSize + quadidx] += b[idx22];
			temp[13 * quadSize + quadidx] += b[idx22];
		}
	}

	// M1 = (A11 + A22) @ (B11 + B22) = M1L @ M1R	(14)
	// M2 = (A21 + A22) @ B11 = M2L @ B11			(15)
	// M3 = A11 @ (B12 - B22) = A11 @ M3R			(16)
	// M4 = A22 @ (B21 - B11) = A22 @ M4R			(17)
	// M5 = (A11 + A12) @ B22 = M5L @ B22			(18)
	// M6 = (A21 - A11) @ (B11 + B12) = M6L @ M6R	(19)
	// M7 = (A12 - A22) @ (B21 + B22) = M7L @ M7R	(20)
	strassen<T>(14 * quadSize + temp, temp, quadSize + temp, quadRC);
	strassen<T>(15 * quadSize + temp, 2 * quadSize + temp, 3 * quadSize + temp, quadRC);
	strassen<T>(16 * quadSize + temp, 4 * quadSize + temp, 5 * quadSize + temp, quadRC);
	strassen<T>(17 * quadSize + temp, 6 * quadSize + temp, 7 * quadSize + temp, quadRC);
	strassen<T>(18 * quadSize + temp, 8 * quadSize + temp, 9 * quadSize + temp, quadRC);
	strassen<T>(19 * quadSize + temp, 10 * quadSize + temp, 11 * quadSize + temp, quadRC);
	strassen<T>(20 * quadSize + temp, 12 * quadSize + temp, 13 * quadSize + temp, quadRC);

	// C11 = M1 + M4 - M5 + M7
	// C12 = M3 + M5
	// C21 = M2 + M4
	// C22 = M1 - M2 + M3 + M6
	for (size_t x = 0; x < quadRC; x++)
	{
		for (size_t y = 0; y < quadRC; y++)
		{
			size_t quadidx = x + quadRC * y;

			// C11
			size_t idx11 = x + dimPad * y;
			c[idx11] = temp[14 * quadSize + quadidx] + temp[17 * quadSize + quadidx] - temp[18 * quadSize + quadidx] + temp[20 * quadSize + quadidx];

			// C12
			size_t idx12 = x + quadRC + dimPad * y;
			c[idx12] = temp[16 * quadSize + quadidx] + temp[18 * quadSize + quadidx];

			// C21
			size_t idx21 = x + dimPad * (y + quadRC);
			c[idx21] = temp[15 * quadSize + quadidx] + temp[17 * quadSize + quadidx];

			// C22
			size_t idx22 = x + quadRC + dimPad * (y + quadRC);
			c[idx22] = temp[14 * quadSize + quadidx] - temp[15 * quadSize + quadidx] + temp[16 * quadSize + quadidx] + temp[19 * quadSize + quadidx];
		}
	}
}

template <typename T>
matmul<T>* matmul<T>::get (inode<T>* a, inode<T>* b,
	bool transposeA, bool transposeB)
{
	if (a && b)
	{
		std::unordered_set<inode<T>*> audience;
		if (a->find_audience(nnutils::formatter() << "matmul" << transposeA << transposeB, audience))
		{
			// share nodes when possible
			for (inode<T>* aud : audience)
			{
				if (matmul<T>* mul = dynamic_cast<matmul<T>*>(aud))
				{
					std::vector<inode<T>*> args = mul->get_arguments();
					if (args.size() == 2 && args[0] == a && args[1] == b)
						return mul;
				}
			}
		}
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
new transfer_func<T>(
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
		size_t bx = rank2 ? bl[0] : 0;
		size_t by = rank2 > 1 ? bl[1] : 1;

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
			std::stringstream ss;
			ss << "matmul shapes ";
			print_shape(t1s, ss);
			if (transposeA) ss << "transposed ";
			ss << "and ";
			print_shape(t2s, ss);
			if (transposeB) ss << "transposed ";
			ss << "do not match";
			throw std::logic_error(ss.str());
		}
		res_shape.insert(res_shape.end(), beyond2d.begin(), beyond2d.end());
		return res_shape;
	},
	{
		[transposeA](size_t i, tensorshape& ashape, const tensorshape& outshape) -> std::vector<size_t>
		{
			std::vector<size_t> coord = outshape.coordinate_from_idx(i);
			if (coord.size() < 2)
			{
				coord.push_back(0);
			}
			std::vector<size_t> alist = ashape.as_list();
			std::vector<size_t> indices;
			size_t idx = 0;
			size_t ncol = alist[0]; // traverse along the row
			if (transposeA)
			{
				idx = 1;
				ncol = alist.size() > 1 ? alist[1] : 1;
				std::swap(coord[0], coord[1]);
			}
			for (size_t j = 0; j < ncol; j++)
			{
				coord[idx] = j;
				indices.push_back(ashape.sequential_idx(coord));
			}
			return indices;
		},
		[transposeB](size_t i, tensorshape& bshape, const tensorshape& outshape) -> std::vector<size_t>
		{
			std::vector<size_t> coord = outshape.coordinate_from_idx(i);
			if (coord.size() < 2)
			{
				coord.push_back(0);
			}
			std::vector<size_t> blist = bshape.as_list();
			std::vector<size_t> indices;
			size_t idx = 1;
			size_t nrow = blist.size() > 1 ? blist[1] : 1; // traverse along the col
			if (transposeB)
			{
				idx = 0;
				nrow = blist[0];
				std::swap(coord[0], coord[1]);
			}
			for (size_t j = 0; j < nrow; j++)
			{
				coord[idx] = j;
				indices.push_back(bshape.sequential_idx(coord));
			}
			return indices;
		}
	},
	ELEM_FUNC<T>([](const T** group, size_t n) -> T
	{
		T accum = 0;
		for (size_t i = 0; i < n; i += 2)
		{
			accum += *(group[i]) * *(group[i+1]);
		}
		return accum;
	})),
[](std::vector<std::pair<inode<T>*,inode<T>*> >)
{
	return constant<T>::get_shared_one();
}, nnutils::formatter() << "matmul" << transposeA << transposeB)
{
	if (base_immutable<T>* imma = dynamic_cast<base_immutable<T>*>(a)) imma->mergible_ = false;
	if (base_immutable<T>* immb = dynamic_cast<base_immutable<T>*>(b)) immb->mergible_ = false;

	auto jtrans = [a, b, transposeA, transposeB](inode<T>* root, NODE_MAN<T> grad_getter) -> inode<T>*
	{
		varptr<T> grada = grad_getter(a);
		varptr<T> gradb = grad_getter(b);

		constant<T>* aconst = dynamic_cast<constant<T>*>(grada.get());
		constant<T>* bconst = dynamic_cast<constant<T>*>(gradb.get());
		if (aconst && *aconst == (T)0 && bconst && *bconst == (T)0)
		{
			return root;
		}
		varptr<T> mA;
		varptr<T> mB;
//		if (1 == root->get_shape().n_elems())
//		{
//			mA = root;
//			mB = root;
//		}
//		else
//		{
			mA = matmul<T>::get(root, b, false, !transposeB);
			mB = matmul<T>::get(a, root, !transposeA, false);
//		}
		if (transposeA)
		{
			mA = transpose<T>(mA);
		}
		if (transposeB)
		{
			mB = transpose<T>(mB);
		}
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