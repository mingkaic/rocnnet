//
// Created by Mingkai Chen on 2016-11-20.
//

#ifndef ROCNNET_MOCK_OPERATION_H
#define ROCNNET_MOCK_OPERATION_H

#include "gmock/gmock.h"
#include "tensor/tensorshape.hpp"
#include "graph/operation/ioperation.hpp"

class MockOperation : public virtual nnet::ioperation<double>
{
	private:
		MockOperation (ivariable<double>* in) :
			ivariable<double>(std::vector<size_t>{}, ""),
			ioperation<double>(std::vector<ivariable<double>*>{in}, ""),
			iobserver(std::vector<ccoms::subject*>{in})
		{ /* this->out remains null */ }


	public:
		static MockOperation* build (ivariable<double>* in) { return new MockOperation(in); }

		~MockOperation (void) {}
		// inherited from iobserver
		MOCK_METHOD1(update, void(ccoms::subject*));
		MOCK_METHOD1(clone_impl, ivariable<double>*(std::string));
		MOCK_METHOD0(shape_eval, nnet::tensorshape(void));
		MOCK_METHOD0(setup_gradient, void(void));
};


#endif //ROCNNET_MOCK_OPERATION_H
