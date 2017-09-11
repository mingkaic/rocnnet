//
// Created by Mingkai Chen on 2017-03-26.
//

#ifndef TENNCOR_MOCK_IMMUTABLE_H
#define TENNCOR_MOCK_IMMUTABLE_H

#include "util_test.h"
#include "fuzz.h"

#include "graph/connector/immutable/immutable.hpp"

using namespace nnet;


SHAPER get_testshaper (void)
{
	tensorshape shape = random_def_shape();
	return [shape](std::vector<tensorshape>) { return shape; };
}


void testtrans (double* dest, std::vector<const double*> src, nnet::shape_io shape)
{
	size_t n_elems = shape.outs_.n_elems();
	std::uniform_real_distribution<double> dist(0, 13);

	auto gen = std::bind(dist, nnutils::get_generator());
	std::generate(dest, dest + n_elems, gen); // initialize to avoid errors
	if (src.size())
	{
		n_elems = std::min(n_elems, shape.ins_[0].n_elems());
		std::memcpy(dest, src[0], n_elems * sizeof(double));
	}
}


inode<double>* testback (std::vector<std::pair<inode<double>*,inode<double>*> >)
{
	return nullptr;
}


class mock_immutable : public immutable<double>
{
public:
	mock_immutable (std::vector<inode<double>*> args, std::string label,
		SHAPER shapes = get_testshaper(),
		TRANSFER_FUNC<double> tfunc = testtrans,
		BACK_MAP<double> back = testback) :
	immutable<double>(args, shapes,
	new transfer_func<double>(tfunc), back, label) {}

	std::function<void(mock_immutable*)> triggerOnDeath;

	virtual void death_on_broken (void)
	{
		if (triggerOnDeath)
		{
			triggerOnDeath(this);
		}
		immutable<double>::death_on_broken();
	}
};


#endif //TENNCOR_MOCK_IMMUTABLE_H
