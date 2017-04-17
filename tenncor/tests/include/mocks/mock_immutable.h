//
// Created by Mingkai Chen on 2017-03-26.
//

#ifndef TENNCOR_MOCK_IMMUTABLE_H
#define TENNCOR_MOCK_IMMUTABLE_H

#include "util_test.h"
#include "fuzz.h"

#include "graph/connector/immutable/immutable.hpp"

using namespace nnet;


tensorshape testshaper (std::vector<tensorshape>)
{
	return random_def_shape();
}


void testforward (double* out,const tensorshape& outs,std::vector<const double*>&,std::vector<tensorshape>&)
{
	size_t n = outs.n_elems();
	std::vector<double> data = FUZZ::getDouble(n);
	for (size_t i = 0; i < n; i++)
	{
		out[i] = data[i];
	}
}


inode<double>* testback (std::vector<inode<double>*>, variable<double>*)
{
	return nullptr;
}


class mock_immutable : public immutable<double>
{
public:
	mock_immutable (std::vector<inode<double>*> args, std::string label,
		SHAPER shapes = testshaper,
		FORWARD_OP<double> forward = testforward,
		BACK_MAP<double> back = testback) :
			immutable<double>(args, shapes, forward, back, label) {}

	std::function<void(mock_immutable*)> triggerOnDeath;

	virtual void commit_sudoku (void)
	{
		if (triggerOnDeath)
		{
			triggerOnDeath(this);
		}
		immutable<double>::commit_sudoku();
	}
};


#endif //TENNCOR_MOCK_IMMUTABLE_H
