//
// Created by Mingkai Chen on 2016-11-20.
//

#ifndef ROCNNET_MOCK_OPERATION_H
#define ROCNNET_MOCK_OPERATION_H

#include "gmock/gmock.h"
#include "tensor/tensorshape.hpp"
#include "graph/operation/ioperation.hpp"

class DummyOp : public nnet::ioperation<double>
{
	protected:
		DummyOp (ivariable<double>* in) :
			ioperation<double>(std::vector<ivariable<double>*>{in}, "")
		{ /* this->out remains null */ }

		DummyOp (ivariable<double>* a, ivariable<double>* b) :
			ioperation<double>(std::vector<ivariable<double>*>{a, b}, "")
		{ /* this->out remains null */ }


	public:
		~DummyOp (void) {}
		// inherited from iobserver
		virtual void update (ccoms::caller_info info, ccoms::update_message msg = ccoms::update_message())
		{
			mock_update(info, msg);
			nnet::ioperation<double>::update(info, msg);
		}

		virtual void mock_update (ccoms::caller_info info, ccoms::update_message msg = ccoms::update_message()) = 0;
};

class MockOperation : public DummyOp
{
	private:
		MockOperation (ivariable<double>* in) :
			DummyOp(in)
		{ /* this->out remains null */ }
		MockOperation (ivariable<double>* a, ivariable<double>* b) :
			DummyOp(a, b)
		{ /* this->out remains null */ }

		MockOperation (const MockOperation& other) : DummyOp(other) {}

	public:
		static MockOperation* build (ivariable<double>* in) { return new MockOperation(in); }
		static MockOperation* build (ivariable<double>* a, ivariable<double>* b) { return new MockOperation(a, b); }

		~MockOperation (void) {}
		virtual MockOperation* clone (void) { return new MockOperation(*this); }
		
		// inherited from iobserver
		MOCK_METHOD1(mock_update, void (ccoms::caller_info));
		MOCK_METHOD2(mock_update, void (ccoms::caller_info, ccoms::update_message));
		MOCK_METHOD1(clone_impl, ivariable<double>*(std::string));
		MOCK_METHOD0(shape_eval, nnet::tensorshape(void));
		MOCK_METHOD0(setup_gradient, void(void));
};


#endif //ROCNNET_MOCK_OPERATION_H
