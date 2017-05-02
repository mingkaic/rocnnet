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
		nzeros = FUZZ::getInt(1, {1, shapelist.size()-1})[0];
	}
	std::vector<size_t> zeros = FUZZ::getInt(nzeros, {0, shapelist.size()-1});
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
	size_t scalar = FUZZ::getInt(1, {2, 10})[0];
	std::vector<size_t> shape = FUZZ::getInt(scalar, {0, 21});
	return tensorshape(shape);
}

tensorshape random_def_shape (int lowerrank, int upperrank, size_t minn, size_t maxn)
{
	size_t rank;
	if (lowerrank == upperrank)
	{
		rank = lowerrank;
	}
	else
	{
		rank = FUZZ::getInt(1, {lowerrank, upperrank})[0];
	}
	size_t minvalue = std::pow((double)minn, 1 / (double)rank);
	size_t maxvalue = std::pow((double)maxn, 1 / (double)rank);
	if (minvalue > maxvalue) minvalue = maxvalue / 2;
	std::vector<size_t> shape = FUZZ::getInt(rank, {minvalue, maxvalue});
	return tensorshape(shape);
}


#endif
