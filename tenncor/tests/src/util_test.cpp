//
// Created by Mingkai Chen on 2017-03-10.
//

#include "util_test.h"
#include "fuzz.h"


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
	std::vector<size_t> zeros = FUZZ<size_t>::get(
		FUZZ<size_t>::get(1, {1, 5})[0], {0, shapelist.size()-1});
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
	auto it = v.begin(), et = v.end();
	for (size_t i = 0; i < rows; i++)
	{
		mat[i].insert(mat[i].end(), it, it+cols);
		it+=cols;
	}
	return mat;
}

tensorshape random_shape (void)
{
	size_t rank = FUZZ<size_t>::get(1, {2, 10})[0];
	std::vector<size_t> shape = FUZZ<size_t>::get(rank, {2, 21});
	return tensorshape(shape);
}
