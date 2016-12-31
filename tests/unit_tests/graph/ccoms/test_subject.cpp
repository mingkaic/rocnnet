//
// Created by Mingkai Chen on 2016-11-20.
//

#include "mock_ccoms.h"
using ::testing::_;


// Behavior E500
TEST(CCOMS, DetachObserverOnDeath_E500)
{
	MockSubject* subject = new MockSubject();

	// dynamically allocate to detect leak when subject dies
	MockObserver* o1 = MockObserver::build(subject);
	MockObserver* o2 = MockObserver::build(subject);
	MockObserver* o3 = MockObserver::build(subject);

	// can't expect subject detach, since inherited MockSubject dies
	// before base subject class detach is called
	// if o1, o2, and o3 memory leaks then error

	// death
	delete subject;
}


// Behavior E501
TEST(CCOMS, ObserverUpdate_E501)
{
    MockSubject subject;
	MockObserver* o1 = MockObserver::build(&subject);
	MockObserver* o2 = MockObserver::build(&subject);
	MockObserver* o3 = MockObserver::build(&subject);
	// subject is now observed 3 times
	// triggering subject notify will call each update 3 times
	EXPECT_CALL(*o1, update(_, _)).Times(1);
	EXPECT_CALL(*o2, update(_, _)).Times(1);
	EXPECT_CALL(*o3, update(_, _)).Times(1);

	subject.notify();
}


// Behavior E502
TEST(CCOMS, NoCopy_E502)
{
	MockSubject solo;
	MockSubject leaf1;
	MockSubject leaf2;

	MockObserver* branch1 = MockObserver::build(&solo);
	MockObserver* branch2 = MockObserver::build(&leaf1, &leaf2);

	EXPECT_CALL(solo, detach(_)).Times(1);
	EXPECT_CALL(leaf1, detach(_)).Times(1);
	EXPECT_CALL(leaf2, detach(_)).Times(1);

	MockSubject* solocpy = new MockSubject(solo);
	MockSubject* leaf1cpy = new MockSubject(leaf1);
	MockSubject* leaf2cpy = new MockSubject(leaf2);
	
	// subject copies never copy over audiences
	EXPECT_TRUE(solocpy->no_audience());
	EXPECT_TRUE(leaf1cpy->no_audience());
	EXPECT_TRUE(leaf2cpy->no_audience());
	
	delete branch1;
	delete branch2;
	delete solocpy;
	delete leaf1cpy;
	delete leaf2cpy;
	// real detach
	solo.mock_detach(branch1);
	leaf1.mock_detach(branch2);
	leaf2.mock_detach(branch2);
}
