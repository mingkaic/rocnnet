//
// Created by Mingkai Chen on 2016-11-20.
//

#include "mock_ccoms.h"
using ::testing::_;


// Behavior E500
TEST(CCOMS, DetachObserverOnDeath_E500)
{
//	MockSubject* subject = new MockSubject();
//
//	// dynamically allocate to detect leak when subject dies
//	MockObserver* o1 = MockObserver::build(subject);
//	MockObserver* o2 = MockObserver::build(subject);
//	MockObserver* o3 = MockObserver::build(subject);
//
//	// subject auto detach on destruction
//	EXPECT_CALL(*subject, detach(_)).Times(3);
//	// death
//	delete subject;
//	// REAL cleanup
//	// these should detach observers from subject
//	delete o1;
//	delete o2;
//	delete o3;
}


// Behavior E501
TEST(CCOMS, ObserverUpdate_E501)
{

}


// Behavior E502
TEST(CCOMS, NoCopy_E502)
{

}