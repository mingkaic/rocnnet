//
// Created by Mingkai Chen on 2016-11-20.
//

#ifndef ROCNNET_MOCK_OPERATION_H
#define ROCNNET_MOCK_OPERATION_H

#include "gmock/gmock.h"
#include "tensor/tensorshape.hpp"
#include "graph/operation/immutable/operation.hpp"

class DummyOp : public nnet::operation<double>
{
	protected:
		DummyOp (inode<double>* in) :
			operation<double>(std::vector<inode<double>*>{in}, "")
		{ /* this->out remains null */ }

		DummyOp (inode<double>* a, inode<double>* b) :
			operation<double>(std::vector<inode<double>*>{a, b}, "")
		{ /* this->out remains null */ }


	public:
		~DummyOp (void) {}
		// inherited from iobserver
		virtual void update (react::caller_info info, react::update_message msg = react::update_message())
		{
			mock_update(info, msg);
			nnet::operation<double>::update(info, msg);
		}

		virtual void mock_update (react::caller_info info, react::update_message msg = react::update_message()) = 0;
};

class MockOperation : public DummyOp
{
	private:
		MockOperation (inode<double>* in) :
			DummyOp(in)
		{ /* this->out remains null */ }
		MockOperation (inode<double>* a, inode<double>* b) :
			DummyOp(a, b)
		{ /* this->out remains null */ }

		MockOperation (const MockOperation& other) : DummyOp(other) {}

	public:
		static MockOperation* build (inode<double>* in) { return new MockOperation(in); }
		static MockOperation* build (inode<double>* a, inode<double>* b) { return new MockOperation(a, b); }

		~MockOperation (void) {}
		virtual MockOperation* clone (void) { return new MockOperation(*this); }
		
		// inherited from iobserver
		MOCK_METHOD1(mock_update, void (react::caller_info));
		MOCK_METHOD2(mock_update, void (react::caller_info, react::update_message));
		MOCK_METHOD1(clone_impl, inode<double>*(std::string));
		MOCK_METHOD0(shape_eval, nnet::tensorshape(void));
		MOCK_METHOD0(setup_gradient, inode<double>*(void));
};


#endif //ROCNNET_MOCK_OPERATION_H
