//
// Created by Mingkai Chen on 2017-03-10.
//

//#define DISABLE_GRAPH_MODULE_TESTS
#ifndef DISABLE_GRAPH_MODULE_TESTS

#include <algorithm>

#include "mocks/mock_subject.h"
#include "mocks/mock_observer.h"
#include "fuzz.h"


//#define DISABLE_REACT_TEST
#ifndef DISABLE_REACT_TEST


// covers subject
// copy constructor and assignment,
// no audience, attach, detach
TEST(REACT, CopySub_A000)
{
	mock_subject sassign1;
	mock_subject sassign2;

	mock_observer o1;
	mock_subject s1;
	mock_subject s2(&o1);

	EXPECT_TRUE(s1.no_audience());
	EXPECT_FALSE(s2.no_audience());

	mock_subject cpy1(s1);
	mock_subject cpy2(s2);

	EXPECT_TRUE(cpy1.no_audience());
	EXPECT_TRUE(cpy2.no_audience());

	sassign1 = s1;
	sassign2 = s2;

	EXPECT_TRUE(sassign1.no_audience());
	EXPECT_TRUE(sassign2.no_audience());

	// detach to avoid doing anything with o1
	s1.mock_detach(&o1);
	s2.mock_detach(&o1);
	cpy1.mock_detach(&o1);
	cpy2.mock_detach(&o1);
	sassign2.mock_detach(&o1);
	sassign2.mock_detach(&o1);
}


// covers subject
// move constructor and assignment
TEST(REACT, MoveSub_A000)
{
	mock_subject sassign1;
	mock_subject sassign2;

	mock_observer o1;
	mock_subject s1;
	mock_subject s2(&o1);

	EXPECT_TRUE(s1.no_audience());
	EXPECT_FALSE(s2.no_audience());

	mock_subject mv1(std::move(s1));
	mock_subject mv2(std::move(s2));

	EXPECT_TRUE(s1.no_audience());
	EXPECT_TRUE(s2.no_audience());
	EXPECT_TRUE(mv1.no_audience());
	ASSERT_FALSE(mv2.no_audience());

	sassign1 = std::move(mv1);
	sassign2 = std::move(mv2);

	EXPECT_TRUE(mv1.no_audience());
	EXPECT_TRUE(mv2.no_audience());
	EXPECT_TRUE(sassign1.no_audience());
	ASSERT_FALSE(sassign2.no_audience());

	// detach to avoid doing anything with o1
	s1.mock_detach(&o1);
	s2.mock_detach(&o1);
	mv1.mock_detach(&o1);
	mv2.mock_detach(&o1);
	sassign2.mock_detach(&o1);
	sassign2.mock_detach(&o1);
}


// covers subject, iobserver
// notify
TEST(REACT, Notify_A001)
{
	mock_observer o1;
	mock_observer o2;
	mock_subject s1(&o1);
	mock_subject s2(&o2, &o1);

	EXPECT_CALL(o1, update(0, UPDATE)).Times(1);
	EXPECT_CALL(o1, update(&s1)).Times(1);
	s1.notify(UPDATE); // o1 update gets s1 at idx 0

	EXPECT_CALL(o1, update(1, UPDATE)).Times(1);
	EXPECT_CALL(o2, update(0, UPDATE)).Times(1);
	EXPECT_CALL(o1, update(&s2)).Times(1);
	EXPECT_CALL(o2, update(&s2)).Times(1);
	s2.notify(UPDATE);
	// o2 update gets s2 at idx 0,
	// o1 update gets s2 at idx 1

	// suicide calls
	EXPECT_CALL(o1, update(0, UNSUBSCRIBE)).Times(2);
	EXPECT_CALL(o1, commit_sudoku()).Times(2);
	s1.notify(UNSUBSCRIBE);

	EXPECT_CALL(o1, update(1, UNSUBSCRIBE)).Times(1);
	EXPECT_CALL(o2, update(0, UNSUBSCRIBE)).Times(1);
	EXPECT_CALL(o1, commit_sudoku()).Times(1);
	EXPECT_CALL(o2, commit_sudoku()).Times(1);
	s2.notify(UNSUBSCRIBE);

	// detach to avoid doing anything with o1 and o2
	s1.mock_detach(&o1);
	s2.mock_detach(&o1);
	s2.mock_detach(&o2);
}


// covers subject
// destructor
TEST(REACT, SUBDEATH_A002)
{
//	MockSubject* subject = new MockSubject();
//
//	// dynamically allocate to detect leak when subject dies
//	MockObserver* o1 = MockObserver::build(subject);
//	MockObserver* o2 = MockObserver::build(subject);
//	MockObserver* o3 = MockObserver::build(subject);
//
//	// can't expect subject detach, since inherited MockSubject dies
//	// before base subject class detach is called
//	// if o1, o2, and o3 memory leaks then error
//
//	// death
//	delete subject;
}


// covers subject
// attach
TEST(REACT, Attach_A003)
{
}


// covers subject
// detach without index and with index
TEST(REACT, Detach_A004)
{
}


// covers iobserver
// default and dependency constructors
TEST(REACT, ObsConstruct_A005)
{
}


// covers iobserver
// copy constructor and assignment
TEST(REACT, CopyObs_A006)
{
//	MockSubject solo;
//	MockSubject leaf1;
//	MockSubject leaf2;
//
//	MockObserver* branch1 = MockObserver::build(&solo);
//	MockObserver* branch2 = MockObserver::build(&leaf1, &leaf2);
//
//	EXPECT_CALL(solo, detach(_)).Times(2);
//	EXPECT_CALL(leaf1, detach(_)).Times(2);
//	EXPECT_CALL(leaf2, detach(_)).Times(2);
//
//	MockObserver* branch1cpy = new MockObserver(*branch1);
//	MockObserver* branch2cpy = new MockObserver(*branch2);
//
//	// expect dependencies to equal
//	std::vector<subject*> dep1 = branch1cpy->expose_dependencies();
//	std::vector<subject*> dep2 = branch2cpy->expose_dependencies();
//	EXPECT_EQ(1, dep1.size());
//	EXPECT_EQ(&solo, dep1[0]);
//	EXPECT_EQ(2, dep2.size());
//	// order should matter
//	EXPECT_EQ(&leaf1, dep2[0]);
//	EXPECT_EQ(&leaf2, dep2[1]);
//
//	// actual cleanup
//	delete branch1;
//	delete branch2;
//	delete branch1cpy;
//	delete branch2cpy;
//	// real detach
//	solo.mock_detach(branch1);
//	solo.mock_detach(branch1cpy);
//	leaf1.mock_detach(branch2);
//	leaf1.mock_detach(branch2cpy);
//	leaf2.mock_detach(branch2);
//	leaf2.mock_detach(branch2cpy);
}


// covers iobserver
// move constructor and assignment
TEST(REACT, MoveObs_A006)
{
}


// covers iobserver
// add_dependency
TEST(REACT, AddDep_A007)
{
}


// covers iobserver
// remove_dependency
TEST(REACT, RemDep_A008)
{
}


// covers iobserver
// replace_dependency
TEST(REACT, RepDep_A009)
{
}


// covers iobserver
// destruction, depends on subject detach
TEST(REACT, ObsDeath_A010)
{
//	MockSubject subject;
//
//	MockObserver* o1 = MockObserver::build(&subject);
//	MockObserver* o2 = MockObserver::build(&subject);
//	MockObserver* o3 = MockObserver::build(&subject);
//
//	EXPECT_CALL(subject, detach(o1));
//	EXPECT_CALL(subject, detach(o2));
//	EXPECT_CALL(subject, detach(o3));
//
//	// these should detach observers from subject
//	delete o1;
//	delete o2;
//	delete o3;
//	// real detach
//	subject.mock_detach(o1);
//	subject.mock_detach(o2);
//	subject.mock_detach(o3);
}


#endif /* DISABLE_REACT_TEST */


#endif /* DISABLE_GRAPH_MODULE_TESTS */
