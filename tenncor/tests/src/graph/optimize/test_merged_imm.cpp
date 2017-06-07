//
// Created by Mingkai Chen on 2017-05-24.
//

#ifndef DISABLE_OPTIMIZE_MODULE_TESTS

#include <algorithm>

#include "graph/connector/immutable/matmul.hpp"
#include "graph/connector/immutable/elementary.hpp"
#include "graph/connector/immutable/transform.hpp"

#include "gtest/gtest.h"
#include "util_test.h"
#include "fuzz.h"


#ifndef DISABLE_MERGE_IMM_TEST


using namespace nnet;


using OP_ARG = std::vector<varptr<double> >;
using OP_FUNC = std::function<varptr<double>(OP_ARG)>;


struct OP_INFO
{
	OP_INFO (OP_FUNC func, size_t n_args) : func_(func), n_args_(n_args) {}
	OP_INFO (OP_FUNC func, size_t n_args,
		std::function<void(tensorshape,tensorshape&,tensorshape&)> shape) :
	func_(func), n_args_(n_args), second_shape_(shape) {}

	OP_FUNC func_;
	size_t n_args_;
	std::function<void(tensorshape,tensorshape&,tensorshape&)> second_shape_;
};


varptr<double> random_2layer_graph (tensorshape main_shape,
	std::vector<variable<double>*>& leaves)
{
	rand_uniform<double> rinit(0, 1);
	std::vector<double> scalars = FUZZ::getDouble(3, "scalar arguments", {0.2, 1000});
	size_t sidx = 0;
	const std::vector<OP_INFO> OPS =
	{
		{[](OP_ARG args) { return +args[0]; }, 1}, // ABS
		{[](OP_ARG args) { return -args[0]; }, 1}, // NEG
		{[](OP_ARG args) { return sin(args[0]); }, 1}, // SIN
		{[](OP_ARG args) { return cos(args[0]); }, 1}, // COS
		{[](OP_ARG args) { return tan(args[0]); }, 1}, // TAN
		{[](OP_ARG args) { return csc(args[0]); }, 1}, // CSC
		{[](OP_ARG args) { return sec(args[0]); }, 1}, // SEC
		{[](OP_ARG args) { return cot(args[0]); }, 1}, // COT
		{[](OP_ARG args) { return exp(args[0]); }, 1}, // EXP
		{[](OP_ARG args) { return sqrt(args[0]); }, 1}, // SQRT
		{[](OP_ARG args) { return pow(args[0], 3); }, 1}, // POW
		{[](OP_ARG args) { return clip_val(args[0], 0.0, 1.0); }, 1}, // CLIPVAL
		{[](OP_ARG args) { return clip_norm(args[0], 2.0); }, 1}, // CLIPNORM
		{[](OP_ARG args) { return args[0] + args[1]; }, 2}, // ADD
		{[](OP_ARG args) { return args[0] - args[1]; }, 2}, // SUB
		{[](OP_ARG args) { return args[0] * args[1]; }, 2}, // MUL
		{[](OP_ARG args) { return args[0] / args[1]; }, 2}, // DIV
		{[scalars, &sidx](OP_ARG args)
		{
			return args[0] + scalars[sidx++];
		}, 1}, // ADD
		{[scalars, &sidx](OP_ARG args)
		{
			return args[0] - scalars[sidx++];
		}, 1}, // SUB
		{[scalars, &sidx](OP_ARG args)
		{
			return args[0] * scalars[sidx++];
		}, 1}, // MUL
		{[scalars, &sidx](OP_ARG args)
		{
			if (scalars[sidx] == 0)
			{
				sidx++;
				return args[0] / 1.0;
			}
			return args[0] / scalars[sidx++];
		}, 1}, // DIV
		{[scalars, &sidx](OP_ARG args)
		{
			return scalars[sidx++] + args[0];
		}, 1}, // ADD
		{[scalars, &sidx](OP_ARG args)
		{
			return scalars[sidx++] - args[0];
		}, 1}, // SUB
		{[scalars, &sidx](OP_ARG args)
		{
			return scalars[sidx++] * args[0];
		}, 1}, // MUL
		{[scalars, &sidx](OP_ARG args)
		{
			return scalars[sidx++] / args[0];
		}, 1}, // DIV
		{[](OP_ARG args) { return matmul<double>::get(args[0], args[1]); }, 2,
		[](tensorshape expectshape, tensorshape& shapea, tensorshape& shapeb) // <m, n> <k, m> -> <k, n>
		{
			size_t alt = FUZZ::getInt(1, "ncol for alternate shape", {1, 17})[0];
			std::vector<size_t> exlist = expectshape.as_list();
			shapea = std::vector<size_t>{alt, exlist[1]};
			shapeb = std::vector<size_t>{exlist[0], alt};
		}}, // MATMUL
		{[](OP_ARG args) { return matmul<double>::get(args[0], args[1], true); }, 2,
		[](tensorshape expectshape, tensorshape& shapea, tensorshape& shapeb) // <n, m> <k, m> -> <k, n>
		{
			size_t alt = FUZZ::getInt(1, "ncol for alternate shape", {1, 17})[0];
			std::vector<size_t> exlist = expectshape.as_list();
			shapea = std::vector<size_t>{exlist[1], alt};
			shapeb = std::vector<size_t>{exlist[0], alt};
		}}, // MATMULT
		{[](OP_ARG args) { return matmul<double>::get(args[0], args[1], false, true); }, 2,
		[](tensorshape expectshape, tensorshape& shapea, tensorshape& shapeb) // <m, n> <m, k> -> <k, n>
		{
			size_t alt = FUZZ::getInt(1, "ncol for alternate shape", {1, 17})[0];
			std::vector<size_t> exlist = expectshape.as_list();
			shapea = std::vector<size_t>{alt, exlist[1]};
			shapeb = std::vector<size_t>{alt, exlist[0]};
		}}, // MATMULFT
		{[](OP_ARG args) { return matmul<double>::get(args[0], args[1], true, true); }, 2,
		[](tensorshape expectshape, tensorshape& shapea, tensorshape& shapeb) // <n, m> <m, k> -> <k, n>
		{
			size_t alt = FUZZ::getInt(1, "ncol for alternate shape", {1, 17})[0];
			std::vector<size_t> exlist = expectshape.as_list();
			shapea = std::vector<size_t>{exlist[1], alt};
			shapeb = std::vector<size_t>{alt, exlist[0]};
		}}, // MATMULTT
	};
	// select the 1st, 2nd, and potential operator,
//	std::vector<size_t> ops = FUZZ::getInt(3, "select three operators", {0, OPS.size()-1});
std::vector<size_t> ops = {2,8,4};
	auto heado = OPS[ops[2]];
	tensorshape a1shape = main_shape;
	tensorshape a2shape = main_shape;
	// heado.non_elem_ = we are constraint by a1shape[0] as shape[1]
	if (heado.second_shape_)
	{
		heado.second_shape_(main_shape, a1shape, a2shape);
	}
	varptr<double> headarg0;
	{
		auto o0 = OPS[ops[0]];
		variable<double>* leaf2 = nullptr;
		tensorshape shapea = a1shape;
		if (o0.n_args_ > 1)
		{
			tensorshape shapeb = a1shape;
			if (o0.second_shape_)
			{
				o0.second_shape_(a1shape, shapea, shapeb);
			}
			leaf2 = new variable<double>(shapeb, rinit, "leaf2");
		}
		variable<double>* leaf1 = new variable<double>(shapea, rinit, "leaf1");
		leaves.push_back(leaf1);
		if (leaf2) leaves.push_back(leaf2);
		headarg0 = o0.func_({varptr<double>(leaf1), varptr<double>(leaf2)});
	}
	varptr<double> headarg1;
	if (heado.n_args_ > 1)
	{
		auto o1 = OPS[ops[1]];
		tensorshape shapea = a2shape;
		tensorshape shapeb = a2shape;
		variable<double>* leaf2 = nullptr;
		if (o1.n_args_ > 1)
		{
			if (o1.second_shape_)
			{
				o1.second_shape_(a2shape, shapea, shapeb);
			}
			leaf2 = new variable<double>(shapeb, rinit, "leaf2");
		}
		variable<double>* leaf1 = new variable<double>(shapea, rinit, "leaf1");
		leaves.push_back(leaf1);
		if (leaf2) leaves.push_back(leaf2);
		headarg1 = o1.func_({varptr<double>(leaf1), varptr<double>(leaf2)});
	}
	return heado.func_({headarg0, headarg1});
}


#include "edgeinfo/comm_record.hpp"


TEST(MERGE_IMM, Constructor_M000)
{
	FUZZ::reset_logger();
	tensorshape shape = random_def_shape(2, 2);

	std::vector<variable<double>*> leaves;
	varptr<double> root = random_2layer_graph(shape, leaves);
	for (variable<double>* base : leaves)
	{
		base->initialize();
	}
	tensorshape origshape = root->get_shape();
	std::vector<double> origdata = expose(root.get());
	inode<double>* back = root->get_gradient(leaves[0]);
	tensorshape backshape = back->get_shape();
	std::vector<double> backdata = expose(back);
	std::vector<subject*> intermediate = dynamic_cast<iconnector<double>*>(root.get())->get_subjects();

	merged_immutable<double>* mroot = merged_immutable<double>::get(dynamic_cast<immutable<double>*>(root.get()));

	// delete intermediates, solo_merge usually takes care of all this
	for (subject* s : intermediate)
	{
		immutable<double>* imm = dynamic_cast<immutable<double>*>(s);
		if (imm && imm->mergible_)
			delete imm;
	}

	// verify if merged node is behaviorally identical to root
	// 1: same forward values
	ASSERT_TRUE(tensorshape_equal(origshape, mroot->get_shape()));
	std::vector<double> mergdata = expose(mroot);
	for (size_t i = 0, n = origdata.size(); i < n; i++)
	{
		EXPECT_EQ(origdata[i], mergdata[i]);
	}
	// 2: same backward node values
	inode<double>* mback = mroot->get_gradient(leaves[0]);
	ASSERT_TRUE(tensorshape_equal(backshape, mback->get_shape()));
	std::vector<double> mbackdata = expose(mback);
	for (size_t i = 0, n = backdata.size(); i < n; i++)
	{
		EXPECT_EQ(backdata[i], mbackdata[i]);
	}
	// verify if merged node is direct parent of leaves
	std::vector<subject*> margs = mroot->get_subjects();
	EXPECT_TRUE(std::equal(margs.begin(), margs.end(), leaves.begin()));

	for (variable<double>* base : leaves)
	{
		delete base;
	}
}


#endif /* DISABLE_MERGE_TEST */


#endif /* DISABLE_OPTIMIZE_MODULE_TESTS */