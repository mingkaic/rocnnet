//
// Created by Mingkai Chen on 2017-03-10.
//

#define DISABLE_GRAPH_MODULE_TESTS
#ifndef DISABLE_GRAPH_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"
#include "fuzz.h"


//#define DISABLE_REACT_TEST
#ifndef DISABLE_REACT_TEST


// Behavior E000
TEST(REACT, DetachFromSubjectOnDeath_E000)
{
	MockSubject subject;

	MockObserver* o1 = MockObserver::build(&subject);
	MockObserver* o2 = MockObserver::build(&subject);
	MockObserver* o3 = MockObserver::build(&subject);

	EXPECT_CALL(subject, detach(o1));
	EXPECT_CALL(subject, detach(o2));
	EXPECT_CALL(subject, detach(o3));

	// these should detach observers from subject
	delete o1;
	delete o2;
	delete o3;
	// real detach
	subject.mock_detach(o1);
	subject.mock_detach(o2);
	subject.mock_detach(o3);
}


// Behavior E001
TEST(REACT, CopyDep_E001)
{
	MockSubject solo;
	MockSubject leaf1;
	MockSubject leaf2;

	MockObserver* branch1 = MockObserver::build(&solo);
	MockObserver* branch2 = MockObserver::build(&leaf1, &leaf2);

	EXPECT_CALL(solo, detach(_)).Times(2);
	EXPECT_CALL(leaf1, detach(_)).Times(2);
	EXPECT_CALL(leaf2, detach(_)).Times(2);

	MockObserver* branch1cpy = new MockObserver(*branch1);
	MockObserver* branch2cpy = new MockObserver(*branch2);

	// expect dependencies to equal
	std::vector<react::subject*> dep1 = branch1cpy->expose_dependencies();
	std::vector<react::subject*> dep2 = branch2cpy->expose_dependencies();
	EXPECT_EQ(1, dep1.size());
	EXPECT_EQ(&solo, dep1[0]);
	EXPECT_EQ(2, dep2.size());
	// order should matter
	EXPECT_EQ(&leaf1, dep2[0]);
	EXPECT_EQ(&leaf2, dep2[1]);

	// actual cleanup
	delete branch1;
	delete branch2;
	delete branch1cpy;
	delete branch2cpy;
	// real detach
	solo.mock_detach(branch1);
	solo.mock_detach(branch1cpy);
	leaf1.mock_detach(branch2);
	leaf1.mock_detach(branch2cpy);
	leaf2.mock_detach(branch2);
	leaf2.mock_detach(branch2cpy);
}


TEST(react, DetachObserverOnDeath_E500)
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
TEST(react, ObserverUpdate_E501)
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
TEST(react, NoCopy_E502)
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


#endif /* DISABLE_REACT_TEST */


#endif /* DISABLE_GRAPH_MODULE_TESTS */
