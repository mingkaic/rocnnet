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

void testforward (double* out,const tensorshape& outs,std::vector<const double*>&,std::vector<tensorshape>&)
{
	size_t n = outs.n_elems();
	std::vector<double> data = FUZZ::getDouble(n, "data");
	std::memcpy(out, &data[0], sizeof(double) * n);
}


inode<double>* testback (std::vector<inode<double>*>, variable<double>*)
{
	return nullptr;
}


class mock_immutable : public immutable<double>
{
public:
	mock_immutable (std::vector<inode<double>*> args, std::string label,
		SHAPER shapes = get_testshaper(),
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
