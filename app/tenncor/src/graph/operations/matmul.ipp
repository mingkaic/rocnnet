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

static inline size_t min_pad (size_t insize)
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
static void cubic_mul (T* c, const T* a, const T* b, size_t dimX, size_t dimY, size_t dimZ, size_t coord_map[4])
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
// strassen coordinates follow matrix convention (row, col)
static void strassen (T* c, T* a, T* b, size_t dimPad)
{
	if (dimPad <= STRASSEN_THRESHOLD)
	{
		size_t coord_map[4] = {dimPad, 1, 1, dimPad};
		return cubic_mul(c, a, b, dimPad, dimPad, dimPad, coord_map);
	}
	size_t quadRC = dimPad/2;
	size_t quadSize = quadRC*quadRC;

	// we are given 12 quadrants + 3 for 15 quadrant (14 required + 1 for recursion)
	T* temp = new T[quadSize];
	T* temp2 = new T[quadSize];
	// third buffer for recursive strassen call
	T* temp3 = new T[quadSize];

	// processed during first iteration
	// M3L = A11		(c0)
	// M2L = A21 + A22	(c1)
	// M4L = A22		(c2)
	// M5L = A11 + A12	(c3)
	// M6L = A21 - A11	(temp)
	// M7L = A12 - A22	(temp2)
	// processed during second iteration
	// M2R = B11		(a0)
	// M3R = B12 - B22	(a1) upon third iteration. during second it only B12
	// M4R = B21 - B11	(a2) upon third iteration. during second it only B21
	// M5R = B22		(a3)
	// processed during third iteration
	// M1L = A11 + A22	(b0) = c0 + c2
	// M1R = B11 + B22	(b1) = a0 + a3
	// M6R = B11 + B12	(b2) = a0 + a1 before M3R op
	// M7R = B21 + B22	(b3) = a2 + a3 before M4R op

	// SPACE SAVING METHOD
	// iteration 1: partition a to c such that
	// every element in quadrant (1, 1) ends up in first 1/4th of c, (1, 2) -> second 1/4, etc.
	for (size_t x = 0; x < quadRC; x++)
	{
		for (size_t y = 0; y < quadRC; y++)
		{
			size_t quadidx = x + quadRC * y;
			size_t idx11 = x + dimPad * y;
			size_t idx12 = x + quadRC + dimPad * y;
			size_t idx21 = x + dimPad * (y + quadRC);
			size_t idx22 = x + quadRC + dimPad * (y + quadRC);

			c[quadidx] = a[idx11]; 								// M3L
			c[quadSize + quadidx] = a[idx21] + a[idx22]; 		// M2L
			c[2 * quadSize + quadidx] = a[idx22]; 				// M4L
			c[3 * quadSize + quadidx] = a[idx11] + a[idx12]; 	// M5L
			temp[quadidx] = a[idx21] - a[idx11];				// M6L
			temp2[quadidx] = a[idx12] - a[idx22]; 				// M7L
		}
	}
	// iteration 2: partition b to a (same rule as iteration 1)
	for (size_t x = 0; x < quadRC; x++)
	{
		for (size_t y = 0; y < quadRC; y++)
		{
			size_t quadidx = x + quadRC * y;
			size_t idx11 = x + dimPad * y;
			size_t idx12 = x + quadRC + dimPad * y;
			size_t idx21 = x + dimPad * (y + quadRC);
			size_t idx22 = x + quadRC + dimPad * (y + quadRC);

			a[quadidx] = b[idx11]; 					// M2R
			a[quadSize + quadidx] = b[idx12]; 		// M3R (preliminary)
			a[2 * quadSize + quadidx] = b[idx21]; 	// M4R (preliminary)
			a[3 * quadSize + quadidx] = b[idx22]; 	// M5R
		}
	}
	// iteration 3: finalize calculations
	for (size_t quadidx = 0; quadidx < quadSize; quadidx++)
	{
		b[quadidx] = c[quadidx] + c[2 * quadSize + quadidx]; 								// M1L
		b[quadSize + quadidx] = a[quadidx] + a[3 * quadSize + quadidx]; 					// M1R
		b[2 * quadSize + quadidx] = a[quadidx] + a[quadSize + quadidx]; 					// M6R
		b[3 * quadSize + quadidx] = a[2 * quadSize + quadidx] + a[3 * quadSize + quadidx]; 	// M7R
		a[quadSize + quadidx] -= a[3 * quadSize + quadidx];									// M3R
		a[2 * quadSize + quadidx] -= a[quadidx];											// M4R
	}

	// goal: clear up c for additions in next stage
	// M6 = (A21 - A11) @ (B11 + B12) = M6L @ M6R	(temp3) = (temp) @ (b2)
	// M7 = (A12 - A22) @ (B21 + B22) = M7L @ M7R	(temp) = (temp2) @ (b3)
	// M1 = (A11 + A22) @ (B11 + B22) = M1L @ M1R	(temp2) = (b0) @ (b1)
	// M2 = (A21 + A22) @ B11 = M2L @ M2R			(b0) = (c1) @ (a0)
	// M3 = A11 @ (B12 - B22) = M3L @ M3R			(b1) = (c0) @ (a1)
	// M4 = A22 @ (B21 - B11) = M4L @ M4R			(b2) = (c2) @ (a2)
	// M5 = (A11 + A12) @ B22 = M5L @ M5R			(b3) = (c3) @ (a3)
	strassen<T>(temp3, temp, b + 2 * quadSize, quadRC);
	strassen<T>(temp, temp2, b + 3 * quadSize, quadRC);
	strassen<T>(temp2, b, b + quadSize, quadRC);
	strassen<T>(b, c + quadSize, a, quadRC);
	strassen<T>(b + quadSize, c, a + quadSize, quadRC);
	strassen<T>(b + 2 * quadSize, c + 2 * quadSize, a + 2 * quadSize, quadRC);
	strassen<T>(b + 3 * quadSize, c + 3 * quadSize, a + 3 * quadSize, quadRC);

	// C11 = M1 + M4 - M5 + M7	(temp2) + (b2) - (b3) + (temp)
	// C12 = M3 + M5			(b1) + (b3)
	// C21 = M2 + M4			(b0) + (b2)
	// C22 = M1 - M2 + M3 + M6	(temp2) - (b0) + (b1) + (temp3)
	for (size_t x = 0; x < quadRC; x++)
	{
		for (size_t y = 0; y < quadRC; y++)
		{
			size_t quadidx = x + quadRC * y;
			size_t idx11 = x + dimPad * y; // C11
			size_t idx12 = x + quadRC + dimPad * y; // C12
			size_t idx21 = x + dimPad * (y + quadRC); // C21
			size_t idx22 = x + quadRC + dimPad * (y + quadRC); // C22

			c[idx11] = temp2[quadidx] + b[2 * quadSize + quadidx] - b[3 * quadSize + quadidx] + temp[quadidx];
			c[idx12] = b[quadSize + quadidx] + b[3 * quadSize + quadidx];
			c[idx21] = b[quadidx] + b[2 * quadSize + quadidx];
			c[idx22] = temp2[quadidx] - b[quadidx] + b[quadSize + quadidx] + temp3[quadidx];
		}
	}
	delete [] temp;
	delete [] temp2;
	delete [] temp3;
}

static inline tensorshape matmul_shaper (std::vector<tensorshape> shapes)
{
	tensorshape& t1s = shapes[0];
	tensorshape& t2s = shapes[1];

	std::vector<size_t> al = t1s.as_list();
	std::vector<size_t> bl = t2s.as_list();
	size_t rank1 = t1s.rank();
	size_t rank2 = t2s.rank();

	// account for vectors
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

	// get resulting shape
	std::vector<size_t> res_shape;
	if (ax == by)
	{
		res_shape = {bx, ay};
	}
	else
	{
		std::stringstream ss;
		ss << "matmul shapes ";
		print_shape(t1s, ss);
		ss << "and ";
		print_shape(t2s, ss);
		ss << "do not match";
		throw std::logic_error(ss.str());
	}
	res_shape.insert(res_shape.end(), beyond2d.begin(), beyond2d.end());
	return res_shape;
}

template <typename T>
varptr<T> matmul (const varptr<T> a, const varptr<T> b, bool transposeA, bool transposeB)
{
	if (nullptr == a.get() || nullptr == b.get()) return nullptr;

	inode<T>* adata = a.get();
	inode<T>* bdata = b.get();
	if (transposeA)
	{
		if (inode<T>* parent = unary_parent_search(adata, "transpose_0_1"))
		{
			adata = parent;
		}
		else
		{
			adata = transpose(a, {0, 1});
		}
	}
	if (transposeB)
	{
		if (inode<T>* parent = unary_parent_search(bdata, "transpose_0_1"))
		{
			bdata = parent;
		}
		else
		{
			bdata = transpose(b, {0, 1});
		}
	}

	std::string opname = "matmul";
	if (inode<T>* parent = ordered_binary_parent_search(adata, bdata, opname))
	{
		return parent;
	}

	immutable<T>* mmul = immutable<T>::get(std::vector<inode<T>*>{adata, bdata},
	matmul_shaper,
	new transfer_func<T>(
	[](T* dest, std::vector<const T*> srcs, shape_io shapes)
	{
		assert(2 == srcs.size());
		std::vector<size_t> alist = shapes.ins_[0].as_list();
		std::vector<size_t> blist = shapes.ins_[1].as_list();
		size_t dim_z = alist[0];
		size_t dim_y;
		if (alist.size() < 2)
		{
			dim_y = 1;
		}
		else
		{
			dim_y = alist[1];
		}
		size_t dim_x = blist[0];

		// assert that beyond2d is same for A, B, and output C
		size_t beyond2d = shapes.ins_[0].n_elems() / (dim_z * dim_y);

#ifdef ENABLE_STRASSEN // strassen is very cumbersome in a lot of cases
		size_t dim_pad = min_pad(std::max(std::max(dim_x, dim_y), dim_z));
		if (dim_pad > STRASSEN_THRESHOLD)
		{
			for (size_t i = 0; i < beyond2d; i++)
			{
				const T* rawa = srcs[0] + i * (dim_z * dim_y);
				const T* rawb = srcs[1] + i * (dim_x * dim_z);
				T* rawc = dest + i * (dim_x * dim_y);

				size_t n_mat = dim_pad * dim_pad;
				T* out = new T[n_mat];
				T* a = new T[n_mat];
				T* b = new T[n_mat];
				std::memset(a, 0, n_mat * sizeof(T));
				std::memset(b, 0, n_mat * sizeof(T));
				for (size_t y = 0; y < dim_y; y++)
				{
					for (size_t z = 0; z < dim_z; z++)
					{
						size_t aidx = dim_z * y + z;
						a[z + dim_pad * y] = rawa[aidx];
					}
				}
				for (size_t z = 0; z < dim_z; z++)
				{
					for (size_t x = 0; x < dim_x; x++)
					{
						size_t bidx = x + dim_x * z;
						b[x + dim_pad * z] = rawb[bidx];
					}
				}
				strassen(out, a, b, dim_pad);
				for (size_t y = 0; y < dim_y; y++)
				{
					std::memcpy(rawc + y * dim_x, out + y * dim_pad, sizeof(T) * dim_x);
				}

				delete [] out;
				delete [] a;
				delete [] b;
			}
			return;
		}
#endif /* ENABLE_STRASSEN */
		for (size_t i = 0; i < beyond2d; i++)
		{
			const T* rawa = srcs[0] + i * (dim_z * dim_y);
			const T* rawb = srcs[1] + i * (dim_x * dim_z);
			T* rawc = dest + i * (dim_x * dim_y);

			size_t coord_map[4] = {dim_z, 1, 1, dim_x};
			cubic_mul<T>(rawc, rawa, rawb, dim_x, dim_y, dim_z, coord_map);
		}
	}),
	[](std::vector<std::pair<inode<T>*,inode<T>*> > args)
	{
		// todo: create alternative operation to eq (since eq prevents higher order derivatives)
		// desired behavior, create a matrix of 1 output in the same shape as the tforward operation
		// tforward shape may not be defined at this point,
		// so the shape instantiation must be performed during base_immutable<T>::update
		nnet::varptr<T> tforward = nnet::matmul<T>(args[0].first, args[1].first);
		return nnet::eq(tforward, tforward);
	}, opname);

	std::unordered_set<ileaf<T>*> temp = mmul->get_leaves();
	std::vector<variable<T>*> leef;
	for (ileaf<T>* ilef : temp)
	{
		if (variable<T>* var = dynamic_cast<variable<T>*>(ilef))
		{
			leef.push_back(var);
		}
	}

	mmul->set_jacobian_back(
	[transposeA, transposeB](inode<T>* root, std::vector<inode<T>*> args, std::vector<inode<T>*> grads) -> inode<T>*
	{
		varptr<T> arga = args[0];
		varptr<T> argb = args[1];
		varptr<T> grada = grads[0];
		varptr<T> gradb = grads[1];

		constant<T>* aconst = dynamic_cast<constant<T>*>(grada.get());
		constant<T>* bconst = dynamic_cast<constant<T>*>(gradb.get());
		if (aconst && *aconst == (T)0 && bconst && *bconst == (T)0)
		{
			return root;
		}
		varptr<T> mA = matmul<T>(root, argb, false, true);
		varptr<T> mB = matmul<T>(arga, root, true);
		if (transposeA)
		{
			mA = transpose<T>(mA);
		}
		if (transposeB)
		{
			mB = transpose<T>(mB);
		}
		return mA * grada + mB * gradb;
	}, leef);
	return mmul;
}

}

#endif