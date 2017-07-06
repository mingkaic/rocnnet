//
// Created by Mingkai Chen on 2016-08-29.
//

#ifndef DISABLE_GRAPH_MODULE_TESTS

#include <algorithm>

#include "graph/leaf/variable.hpp"
#include "graph/operations/operations.hpp"
#include "graph/varptr.hpp"

#include "gtest/gtest.h"
#include "util_test.h"
#include "fuzz.h"


#ifndef DISABLE_TRANSFORM_TEST


using namespace nnet;


using SHAPE_CHANGE = std::function<tensorshape(tensorshape)>;
using DATA_CHANGE = std::function<std::vector<double>(std::vector<double>,tensorshape)>;

template <typename T>
using PARAM_EVAL = std::function<T(tensorshape)>;
template <typename T>
using UNARY_VAR = std::function<varptr<double>(varptr<double>,T)>;


static const double epi = std::numeric_limits<double>::epsilon();


template <typename T>
T no_param (tensorshape) { return (T)0; }


tensorshape as_is (tensorshape in) { return in; }


std::vector<double> onescalar (std::vector<double>, tensorshape)
{
	return std::vector<double>(1, 1);
}


template <typename T=double>
static void unaryTransTest (std::pair<int,int> ranklimit, UNARY_VAR<T> func,
	DATA_CHANGE expect_transfer, SHAPE_CHANGE expect_shape,
	optional<DATA_CHANGE> grad_transfer, optional<SHAPE_CHANGE> grad_shape,
	PARAM_EVAL<T> paramer = no_param<T>)
{
	tensorshape shape = random_def_shape(ranklimit.first, ranklimit.second);
	rand_uniform<double> rinit(2, 12);
	variable<double> var(shape, rinit, "unar_var");
	var.initialize();
	{
		const nnet::tensor<double>* vartens = var.get_eval();
		ASSERT_NE(nullptr, vartens);
		ASSERT_TRUE(vartens->is_alloc());
	}
	varptr<double> res = func(varptr<double>(&var), paramer(shape));
	{
		const nnet::tensor<double>* restens = res->get_eval();
		ASSERT_NE(nullptr, restens);
		ASSERT_TRUE(restens->is_alloc());
	}

	tensorshape expectoshape = expect_shape(shape);
	std::vector<double> varout = expose(&var);
	std::vector<double> expectout = expect_transfer(varout, shape);
	nnet::inode<double>* vgrad = var.get_gradient(&var);
	{
		const nnet::tensor<double>* vgradtens = vgrad->get_eval();
		ASSERT_NE(nullptr, vgradtens);
		ASSERT_TRUE(vgradtens->is_alloc());
	}

	// test forward
	tensorshape outshape = res->get_shape();
	std::vector<double> rout = expose<double>(res);
	EXPECT_TRUE(tensorshape_equal(expectoshape, outshape));
	ASSERT_EQ(expectout.size(), rout.size());
	for (size_t i = 0, n = rout.size(); i < n; i++)
	{
		EXPECT_EQ(expectout[i], rout[i]);
	}

	// test derivative
	if ((bool) grad_transfer && (bool) grad_shape)
	{
		tensorshape gradoshape = (*grad_shape)(var.get_gradient(&var)->get_shape());
		std::vector<double> gradout = (*grad_transfer)(expose(vgrad), vgrad->get_shape());
		const tensor<double>* backt = res->get_gradient(&var)->get_eval();
		tensorshape outgshape = backt->get_shape();
		std::vector<double> rgout = backt->expose();
		EXPECT_TRUE(tensorshape_equal(gradoshape, outgshape));
		ASSERT_EQ(gradout.size(), rgout.size());
		for (size_t i = 0, n = rgout.size(); i < n; i++)
		{
			EXPECT_EQ(gradout[i], rgout[i]);
		}
	}
	else
	{
		EXPECT_THROW(res->get_gradient(&var), std::exception);
	}

	// Behavior K000
	EXPECT_EQ(nullptr, func(varptr<double>(nullptr), paramer(shape)));
}


TEST(TRANSFORM, Transpose_K001)
{
	FUZZ::reset_logger();
	DATA_CHANGE transfer =
	[](std::vector<double> in, tensorshape inshape) -> std::vector<double>
	{
		if (1 == inshape.rank())
		{
			return in;
		}
		std::vector<size_t> slist = inshape.as_list();
		tensorshape outshape({slist[1], slist[0]});
		std::vector<double> out(in.size(), 0);
		for (size_t i = 0, n = in.size(); i < n; i++)
		{
			std::vector<size_t> incoord = inshape.coordinate_from_idx(i);
			size_t j = outshape.sequential_idx({incoord[1], incoord[0]});
			out[j] = in[i];
		}
		return out;
	};
	SHAPE_CHANGE shape =
	[](tensorshape in) -> tensorshape
	{
		if (1 == in.rank())
		{
			return std::vector<size_t>{1, in.as_list()[0]};
		}
		std::vector<size_t> slist = in.as_list();
		return std::vector<size_t>{slist[1], slist[0]};
	};
	unaryTransTest<double>({1, 2},
	[](varptr<double> in,double) { return nnet::transpose(in); },
	transfer, shape, transfer, shape);
}


TEST(TRANSFORM, Fit_K002)
{
	FUZZ::reset_logger();
	tensorshape realshape = random_def_shape();
	rand_uniform<double> rinit(2, 12);
	variable<double> shapeholder(realshape, rinit, "shapeholder");
	shapeholder.initialize();

	PARAM_EVAL<const varptr<double> > fitparam =
	[&shapeholder](tensorshape) -> const varptr<double>
	{
		return varptr<double>(&shapeholder);
	};
	DATA_CHANGE transfer =
	[&realshape](std::vector<double> in, tensorshape inshape) -> std::vector<double>
	{
		size_t n = realshape.n_elems();
		std::vector<double> out(n, 0);
		std::vector<size_t> outlist = realshape.as_list();
		for (size_t i = 0, m = in.size(); i < m; i++)
		{
			std::vector<size_t> incoord = inshape.coordinate_from_idx(i);
			bool b = true;
			for (size_t j = 0, o = incoord.size(); j < o && b; j++)
			{
				if (j >= outlist.size())
				{
					b = incoord[j] == 0;
				}
				else
				{
					b = incoord[j] < outlist[j];
				}
			}
			if (b)
			{
				size_t outidx = realshape.sequential_idx(incoord);
				out[outidx] = in[i];
			}
		}
		return out;
	};
	SHAPE_CHANGE shape = [&realshape](tensorshape) { return realshape; };

	optional<DATA_CHANGE> gradtransfer = (DATA_CHANGE) onescalar;
	optional<SHAPE_CHANGE> gradshape = (SHAPE_CHANGE) as_is;

	unaryTransTest<const varptr<double> >({2, 13},
	[](varptr<double> in, varptr<double> watch) { return nnet::fit(in, watch); },
	transfer, shape, gradtransfer, gradshape, fitparam);
}


TEST(TRANSFORM, Extend_K003To004)
{
	// K004
	FUZZ::reset_logger();
	size_t extend_index;
	size_t multiplier;
	PARAM_EVAL<std::pair<size_t,size_t> > extendparam =
	[&extend_index, &multiplier](tensorshape shape) -> std::pair<size_t,size_t>
	{
		size_t srank = shape.rank();
		extend_index = FUZZ::getInt(1, "extend_index", {0, srank-1})[0];
		multiplier = FUZZ::getInt(1, "multiplier", {2, 5})[0];
		return {extend_index, multiplier};
	};
	DATA_CHANGE transfer =
	[&extend_index, &multiplier](std::vector<double> in, tensorshape inshape) -> std::vector<double>
	{
		std::vector<size_t> invec = inshape.as_list();
		std::vector<double> out;
		size_t baselen = 1;
		for (size_t i = 0; i <= extend_index; i++)
		{
			baselen *= invec[i];
		}
		auto it = in.begin();
		auto et = in.end();
		while (it != et)
		{
			for (size_t i = 0; i < multiplier; i++)
			{
				out.insert(out.end(), it, it+baselen);
			}
			it += baselen;
		}
		return out;
	};
	SHAPE_CHANGE shape =
	[&extend_index, &multiplier](tensorshape inshape) -> tensorshape
	{
		std::vector<size_t> out = inshape.as_list();
		out[extend_index] *= multiplier;
		return out;
	};

	optional<DATA_CHANGE> gradtransfer = (DATA_CHANGE) onescalar;
	optional<SHAPE_CHANGE> gradshape = (SHAPE_CHANGE) as_is;

	unaryTransTest<std::pair<size_t,size_t> >({2, 13},
	[](varptr<double> in, std::pair<size_t,size_t> idxnmult)
	{
		size_t index = idxnmult.first;
		size_t multiplier = idxnmult.second;
		return nnet::extend(in, index, multiplier);
	},
	transfer, shape, gradtransfer, gradshape, extendparam);
	// K005
	tensorshape rshape = random_def_shape(2, 13);
	rand_uniform<double> rinit(2, 12);
	variable<double> var(rshape, rinit, "unar_var");
	varptr<double> zaro = extend(varptr<double>(&var), extend_index, 0);
	const tensor<double>* ztens = zaro->get_eval();
	EXPECT_EQ((size_t) 1, ztens->get_shape().n_elems());
	std::vector<double> zvec = ztens->expose();
	ASSERT_EQ((size_t) 1, zvec.size());
	EXPECT_EQ(0.0, zvec[0]);
	varptr<double> same = extend(varptr<double>(&var), extend_index, 1);
	EXPECT_EQ(&var, same.get());
}


TEST(TRANSFORM, Compress_K005)
{
	FUZZ::reset_logger();
	size_t compress_index;
	ELEM_FUNC<double> compression =
	[](const double** data, size_t n) -> double
	{
		assert(n > 0);
		return *data[0];
	};

	PARAM_EVAL<size_t> compressparam =
	[&compress_index, &compression](tensorshape shape) -> size_t
	{
		size_t srank = shape.rank();
		compress_index = FUZZ::getInt(1, "compress_index", {0, srank-1})[0];
		return compress_index;
	};
	DATA_CHANGE transfer =
	[&compress_index](std::vector<double> in, tensorshape inshape) -> std::vector<double>
	{
		std::vector<size_t> outlist = inshape.as_list();
		if (compress_index >= outlist.size()) return in;
		outlist[compress_index] = 1;
		tensorshape outshape(outlist);
		std::vector<double> out(outshape.n_elems(), 0);
		for (size_t i = 0, m = in.size(); i < m; i++)
		{
			std::vector<size_t> incoord = inshape.coordinate_from_idx(i);
			if (incoord[compress_index] == 0)
			{
				size_t outidx = outshape.sequential_idx(incoord);
				out[outidx] = in[i];
			}
		}
		return out;
	};
	SHAPE_CHANGE shape =
	[&compress_index](tensorshape inshape) -> tensorshape
	{
		std::vector<size_t> out = inshape.as_list();
		if (compress_index >= out.size()) return inshape;
		if (compress_index == 0)
		{
			out = std::vector<size_t>(out.begin()+1, out.end());
		}
		else if (out.size()-1 == (unsigned) compress_index)
		{
			out.pop_back();
		}
		else
		{
			out[compress_index] = 1;
		}
		return out;
	};

	optional<DATA_CHANGE> gradtransfer = (DATA_CHANGE) onescalar;
	optional<SHAPE_CHANGE> gradshape = (SHAPE_CHANGE) as_is;

	unaryTransTest<size_t>({2, 13},
	[&compression](varptr<double> in, size_t compidx) { return compress(in, compidx, compression); },
	transfer, shape, gradtransfer, gradshape, compressparam);
}


TEST(TRANSFORM, ArgCompress_K006To007)
{
	FUZZ::reset_logger();
	size_t arg_index;
	ELEM_FUNC<double> search =
	[](const double** data, size_t n) -> double
	{
		double mean = 0;
		for (size_t i = 0; i < n; i++)
		{
			mean += *(data[i]) / n;
		}
		double error = std::abs(*(data[0]) - mean);
		size_t idx = 0;
		for (size_t i = 1; i < n; i++)
		{
			double potent_error = std::abs(*(data[i]) - mean);
			if (potent_error < error)
			{
				error = potent_error;
				idx = i;
			}
		}
		return idx;
	};

	PARAM_EVAL<size_t> argcompressparam =
	[&arg_index, &search](tensorshape shape) -> size_t
	{
		size_t srank = shape.rank();
		arg_index = FUZZ::getInt(1, "arg_index", {0, srank-1})[0];
		return arg_index;
	};
	DATA_CHANGE transfer =
	[&arg_index, &search](std::vector<double> in, tensorshape inshape) -> std::vector<double>
	{
		std::vector<size_t> outlist = inshape.as_list();
		size_t n = outlist[arg_index];
		assert(arg_index < outlist.size());
		outlist[arg_index] = 1;
		tensorshape outshape(outlist);
		std::vector<double> out(outshape.n_elems(), 0);
		for (size_t i = 0, m = in.size(); i < m; i++)
		{
			std::vector<size_t> incoord = inshape.coordinate_from_idx(i);
			if (incoord[arg_index] == 0)
			{
				std::vector<const double*> vecs;
				size_t outidx = outshape.sequential_idx(incoord);
				for (size_t i = 0; i < n; i++)
				{
					incoord[arg_index] = i;
					size_t inidx = inshape.sequential_idx(incoord);
					vecs.push_back(&in[inidx]);
				}
				out[outidx] = search(&vecs[0], vecs.size());
			}
		}
		return out;
	};
	SHAPE_CHANGE shape =
	[&arg_index](tensorshape inshape) -> tensorshape
	{
		std::vector<size_t> out = inshape.as_list();
		assert(arg_index < out.size());
		if (arg_index == 0)
		{
			out = std::vector<size_t>(out.begin()+1, out.end());
		}
		else if (out.size()-1 == (unsigned) arg_index)
		{
			out.pop_back();
		}
		else
		{
			out[arg_index] = 1;
		}
		return out;
	};

	optional<DATA_CHANGE> gradtransfer;
	optional<SHAPE_CHANGE> gradshape;

	unaryTransTest<size_t>({3, 13},
	[&search](varptr<double> in, size_t arg_index) { return arg_compress(in, arg_index, search); },
	transfer, shape, gradtransfer, gradshape, argcompressparam);
}


#endif /* DISABLE_TRANSFORM_TEST */


#endif /* DISABLE_GRAPH_MODULE_TESTS */
