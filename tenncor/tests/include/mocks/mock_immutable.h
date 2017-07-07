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

double testforward (const double** out, size_t n)
{
	if (n == 0 || nullptr == out[0]) return 0;
	return *out[0];
}


inode<double>* testback (std::vector<std::pair<inode<double>*,inode<double>*> >)
{
	return nullptr;
}


inline std::vector<OUT_MAPPER> nconfirm (std::vector<OUT_MAPPER> inmap, size_t nargs)
{
	if (inmap.size() < nargs)
	{
		for (size_t i = 0; i < nargs; i++)
		{
			inmap.push_back([](size_t i,tensorshape&,const tensorshape&){ return std::vector<size_t>{i}; });
		}
	}
	return inmap;
}


class mock_immutable : public immutable<double>
{
public:
	mock_immutable (std::vector<inode<double>*> args, std::string label,
		SHAPER shapes = get_testshaper(),
		ELEM_FUNC<double> elem = testforward,
		std::vector<OUT_MAPPER> omap = {},
		BACK_MAP<double> back = testback) :
	immutable<double>(args,
	new transfer_func<double>(shapes, nconfirm(omap, args.size()), elem),
	back,
	label) {}

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
