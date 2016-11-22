//
// Created by Mingkai Chen on 2016-11-15.
//

#ifndef ROCNNET_MOCK_CCOMS_H
#define ROCNNET_MOCK_CCOMS_H

#include "gmock/gmock.h"
#include "graph/ccoms/iobserver.hpp"

class MockSubject : public ccoms::subject
{
	public:
		MockSubject (void) {}
		MockSubject(const MockSubject& other) : ccoms::subject(other) {}
		~MockSubject (void) {}

		void mock_detach(ccoms::iobserver* obs)
		{
			ccoms::subject::detach(obs);
		}
		MOCK_METHOD1(detach, void(ccoms::iobserver*));
};

class MockObserver : public ccoms::iobserver
{
	protected:
		MockObserver (std::vector<ccoms::subject*> subs) :
			ccoms::iobserver(subs) {}

	public:
		static MockObserver* build (ccoms::subject* sub)
		{
			return new MockObserver(std::vector<ccoms::subject*>{sub});
		}
		static MockObserver* build (ccoms::subject* a, ccoms::subject* b)
		{
			return new MockObserver(std::vector<ccoms::subject*>{a, b});
		}
		MockObserver(const MockObserver& other) : ccoms::iobserver(other) {}
		~MockObserver (void) {}
		
		std::vector<ccoms::subject*> expose_dependencies (void)
		{
			return this->dependencies_;
		}
		MOCK_METHOD1(update, void(ccoms::subject*));
};

#endif //ROCNNET_MOCK_CCOMS_H
