//
// Created by Mingkai Chen on 2016-11-15.
//

#ifndef ROCNNET_MOCK_CCOMS_H
#define ROCNNET_MOCK_CCOMS_H

#include "gmock/gmock.h"
#include "react/iobserver.hpp"

class MockSubject : public react::subject
{
	public:
		MockSubject (void) : react::subject(nullptr) {}
		MockSubject(const MockSubject& other) : react::subject(other, nullptr) {}
		~MockSubject (void) {}

		void mock_detach(react::iobserver* obs)
		{
			react::subject::detach(obs);
		}
		MOCK_METHOD1(detach, void(react::iobserver*));
};

class MockObserver : public react::iobserver
{
	protected:
		MockObserver (std::vector<react::subject*> subs) :
			react::iobserver(subs) {}

	public:
		static MockObserver* build (react::subject* sub)
		{
			return new MockObserver(std::vector<react::subject*>{sub});
		}
		static MockObserver* build (react::subject* a, react::subject* b)
		{
			return new MockObserver(std::vector<react::subject*>{a, b});
		}
		MockObserver(const MockObserver& other) : react::iobserver(other) {}
		~MockObserver (void) {}
		
		std::vector<react::subject*> expose_dependencies (void)
		{
			return this->dependencies_;
		}
		MOCK_METHOD1(update, void(react::caller_info));
		MOCK_METHOD2(update, void(react::caller_info, react::update_message));
};

#endif //ROCNNET_MOCK_CCOMS_H
