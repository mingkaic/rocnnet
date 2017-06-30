//
// Created by Mingkai Chen on 2017-03-10.
//

#include "util_test.h"
#include "fuzz.h"


#ifdef UTIL_TEST_H


bool tensorshape_equal (
	const tensorshape& ts1,
	const tensorshape& ts2)
{
	std::vector<size_t> vs = ts1.as_list();
	std::vector<size_t> vs2 = ts2.as_list();
	if (vs.size() != vs2.size()) return false;
	return std::equal(vs.begin(), vs.end(), vs2.begin());
}


bool tensorshape_equal (
	const tensorshape& ts1,
	std::vector<size_t>& ts2)
{
	std::vector<size_t> vs = ts1.as_list();
	if (vs.size() != ts2.size()) return false;
	return std::equal(vs.begin(), vs.end(), ts2.begin());
}


void print (std::vector<double> raw)
{
	for (double r : raw)
	{
		std::cout << r << " ";
	}
	std::cout << "\n";
}

tensorshape make_partial (std::vector<size_t> shapelist)
{
	size_t nzeros = 1;
	if (shapelist.size() > 2)
	{
		nzeros = FUZZ::getInt(1, "nzeros", {1, shapelist.size()-1})[0];
	}
	else if (shapelist.size() == 1)
	{
		shapelist.push_back(0);
		return shapelist;
	}
	std::vector<size_t> zeros = FUZZ::getInt(nzeros, "zeros", {0, shapelist.size()-1});
	for (size_t zidx : zeros)
	{
		shapelist[zidx] = 0;
	}
	return tensorshape(shapelist);
}

tensorshape make_incompatible (std::vector<size_t> shapelist)
{
	for (size_t i = 0; i < shapelist.size(); i++)
	{
		shapelist[i]++;
	}
	return tensorshape(shapelist);
}

// make partial full, but incompatible to comp
tensorshape make_full_incomp (std::vector<size_t> partial, std::vector<size_t> complete)
{
	assert(partial.size() == complete.size());
	for (size_t i = 0, n = partial.size(); i < n; i++)
	{
		if (partial[i] == 0)
		{
			partial[i] = complete[i]+1;
		}
	}
	return partial;
}

tensorshape padd(std::vector<size_t> shapelist, size_t nfront, size_t nback)
{
	std::vector<size_t> out(nfront, 1);
	out.insert(out.end(), shapelist.begin(), shapelist.end());
	out.insert(out.end(), nback, 1);
	return tensorshape(out);
}

std::vector<std::vector<double> > doubleDArr(std::vector<double> v, std::vector<size_t> dimensions)
{
	assert(dimensions.size() == 2);
	size_t cols = dimensions[0];
	size_t rows = dimensions[1];
	std::vector<std::vector<double> > mat(rows);
	auto it = v.begin();
	for (size_t i = 0; i < rows; i++)
	{
		mat[i].insert(mat[i].end(), it, it+cols);
		it+=cols;
	}
	return mat;
}

tensorshape random_shape (void)
{
	size_t scalar = FUZZ::getInt(1, "scalar", {2, 10})[0];
	std::vector<size_t> shape = FUZZ::getInt(scalar, "shape", {0, 21});
	return tensorshape(shape);
}

tensorshape random_def_shape (int lowerrank, int upperrank, size_t minn, size_t maxn)
{
	size_t rank = lowerrank;
	if (lowerrank != upperrank)
	{
		rank = FUZZ::getInt(1, "rank", {lowerrank, upperrank})[0];
	}
	assert(rank > 0);
	if (rank < 2)
	{
		return FUZZ::getInt(1, "shape", {minn, maxn});
	}
	// invariant: rank > 1
	size_t maxvalue = 0;
	size_t minvalue = 0;
	size_t lowercut = 0;
	if (rank > 5) lowercut = 1;
	for (size_t i = maxn; i > lowercut; i /= rank)
	{
		maxvalue++;
	}
	for (size_t i = minn; i > lowercut; i /= rank)
	{
		minvalue++;
	}

	// we don't care if minvalue overapproximates
	size_t ncorrection = 0;
	size_t realmaxn = std::pow((double)maxvalue, (double)rank);
	if (realmaxn > maxn)
	{
		for (size_t error = realmaxn - maxn; error > 0; error /= maxvalue)
		{
			ncorrection++;
		}
	}

	std::vector<size_t> shape;
	if (ncorrection == rank)
	// we would need too many corrections, make a shape
	// one dimension at a time (sacrificing rank if necessary)
	{
		if (minvalue < minn)
		// we don't want our output too small either (avoid bad inputs in tests)
		{
			maxvalue += minn - minvalue;
			minvalue = minn;
		}
		for (size_t i = 0; i < rank; i++)
		{
			std::stringstream ss;
			ss << "shape i=" << i;
			size_t shapei = maxvalue;
			if (minvalue != maxvalue)
			{
				shapei = FUZZ::getInt(1, ss.str(), {minvalue, maxvalue})[0];
			}
			shape.push_back(shapei);
			maxn /= shapei;
			if (maxn < maxvalue)
			{
				break; // stop early
			}
		}
	}
	else
	{
		std::vector<size_t> shape2;
		if (rank-ncorrection)
		{
			shape = FUZZ::getInt(rank-ncorrection, "shapepart1", {minvalue, maxvalue});
		}
		if (ncorrection)
		{
			shape2 = FUZZ::getInt(ncorrection, "shapepart2", {minvalue, maxvalue-1});
		}
		shape.insert(shape.end(), shape2.begin(), shape2.end());
	}
	tensorshape outshape(shape);
	if (outshape.n_elems() > 10000)
	{
		// warn
	}
	return outshape;
}


#endif
