//
// Created by Mingkai Chen on 2017-03-10.
//

#ifndef DISABLE_GRAPH_MODULE_TESTS

#include <algorithm>

// #include "mocks/mock_subject.h"
// #include "mocks/mock_observer.h"
#include "fuzz.h"


#include "gmock/gmock.h"
struct IFoo
{
	virtual ~IFoo() {}
	virtual int foo() { return 2; }
}
struct MockFoo : public IFoo
{
	virtual ~MockFoo() {}
	MOCK_METHOD0(foo, int());
}


TEST(REACT, STUPID)
{
	MockFoo mfoo;
}


#define DISABLE_REACT_TEST
#ifndef DISABLE_REACT_TEST


// covers subject
// copy constructor and assignment,
// no audience, attach, detach
TEST(REACT, CopySub_A000)
{
	FUZZ::delim();
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
	FUZZ::delim();
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
	FUZZ::delim();
	mock_observer o1;
	mock_observer o2;
	mock_subject s1(&o1);
	mock_subject s2(&o2, &o1);

	EXPECT_CALL(o1, update(0, UPDATE)).Times(1);
	s1.notify(UPDATE); // o1 update gets s1 at idx 0

	EXPECT_CALL(o1, update(1, UPDATE)).Times(1);
	EXPECT_CALL(o2, update(0, UPDATE)).Times(1);
	s2.notify(UPDATE);
	// o2 update gets s2 at idx 0,
	// o1 update gets s2 at idx 1

	// suicide calls
	EXPECT_CALL(o1, update(0, UNSUBSCRIBE)).Times(1);
	s1.notify(UNSUBSCRIBE);

	EXPECT_CALL(o1, update(1, UNSUBSCRIBE)).Times(1);
	EXPECT_CALL(o2, update(0, UNSUBSCRIBE)).Times(1);
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
	FUZZ::delim();
	mock_observer o1;
	mock_observer o2;
	mock_subject* s1 = new mock_subject(&o1);
	mock_subject* s2 = new mock_subject(&o2, &o1);

	// suicide calls
	EXPECT_CALL(o1, update(0, UNSUBSCRIBE)).Times(1);
	delete s1;

	EXPECT_CALL(o1, update(1, UNSUBSCRIBE)).Times(1);
	EXPECT_CALL(o2, update(0, UNSUBSCRIBE)).Times(1);
	delete s2;
}


// covers subject
// attach
TEST(REACT, Attach_A003)
{
	FUZZ::delim();
	mock_observer o1;
	mock_observer o2;
	mock_subject s1;
	mock_subject s2;

	EXPECT_TRUE(s1.no_audience());
	EXPECT_TRUE(s2.no_audience());

	size_t i = FUZZ::getInt(1)[0];
	s1.mock_attach(&o1, i);
	s2.mock_attach(&o2, i);
	s2.mock_attach(&o1, i + 1);

	EXPECT_FALSE(s1.no_audience());
	EXPECT_FALSE(s2.no_audience());

	s1.mock_attach(&o1, i+1);
	s1.mock_detach(&o1);
	s2.mock_detach(&o2);

	EXPECT_TRUE(s1.no_audience());
	EXPECT_FALSE(s2.no_audience());

	s1.mock_detach(&o1);
	s2.mock_detach(&o1);
	s2.mock_detach(&o2);
}


// covers subject
// detach without index and with index
TEST(REACT, Detach_A004)
{
	FUZZ::delim();
	mock_observer o1;
	mock_observer o2;
	mock_subject s1(&o1, &o2);
	mock_subject s2(&o2, &o2);
	mock_subject s3(&o2, &o2);

	EXPECT_FALSE(s1.no_audience());
	s1.mock_detach(&o1);
	EXPECT_FALSE(s1.no_audience());
	s1.mock_detach(&o2);
	EXPECT_TRUE(s1.no_audience());

	EXPECT_FALSE(s2.no_audience());
	s2.mock_detach(&o2);
	EXPECT_TRUE(s2.no_audience());

	EXPECT_FALSE(s3.no_audience());
	s3.mock_detach(&o2, 1);
	EXPECT_FALSE(s3.no_audience());
	s3.mock_detach(&o2, 0);
	EXPECT_TRUE(s3.no_audience());
}


// covers iobserver
// default and dependency constructors
TEST(REACT, ObsConstruct_A005)
{
	FUZZ::delim();
	mock_subject* s1 = new mock_subject;
	mock_subject* s2 = new mock_subject;
	mock_observer2* o1 = new mock_observer2(s2);
	mock_observer2* o2 = new mock_observer2(s1, s2);

	std::vector<subject*> subs1 = o1->expose_dependencies();
	std::vector<subject*> subs2 = o2->expose_dependencies();

	ASSERT_EQ((size_t) 1, subs1.size());
	EXPECT_EQ(s2, subs1[0]);
	ASSERT_EQ((size_t) 2, subs2.size());
	EXPECT_EQ(s1, subs2[0]);
	EXPECT_EQ(s2, subs2[1]);

	EXPECT_CALL(*o1, commit_sudoku()).Times(1);
	EXPECT_CALL(*o2, commit_sudoku()).Times(2);
	// called twice since mock observer isn't destroyed when commit_sudoku is called
	// so deleting s2 will trigger another suicide call
	delete s1;
	delete s2;
	// again observers aren't destroyed
	delete o1;
	delete o2;
}


// covers iobserver
// copy constructor and assignment
TEST(REACT, CopyObs_A006)
{
	FUZZ::delim();
	mock_observer2* sassign1 = new mock_observer2;
	mock_observer2* sassign2 = new mock_observer2;

	mock_subject* s1 = new mock_subject;
	mock_subject* s2 = new mock_subject;
	mock_observer2* o1 = new mock_observer2(s2);
	mock_observer2* o2 = new mock_observer2(s1, s2);

	mock_observer2* cpy1 = new mock_observer2(*o1);
	mock_observer2* cpy2 = new mock_observer2(*o2);

	std::vector<subject*> subs1 = cpy1->expose_dependencies();
	std::vector<subject*> subs2 = cpy2->expose_dependencies();
	ASSERT_EQ((size_t) 1, subs1.size());
	EXPECT_EQ(s2, subs1[0]);
	ASSERT_EQ((size_t) 2, subs2.size());
	EXPECT_EQ(s1, subs2[0]);
	EXPECT_EQ(s2, subs2[1]);

	*sassign1 = *o1;
	*sassign2 = *o2;

	std::vector<subject*> subs3 = sassign1->expose_dependencies();
	std::vector<subject*> subs4 = sassign2->expose_dependencies();
	ASSERT_EQ((size_t) 1, subs3.size());
	EXPECT_EQ(s2, subs3[0]);
	ASSERT_EQ((size_t) 2, subs4.size());
	EXPECT_EQ(s1, subs4[0]);
	EXPECT_EQ(s2, subs4[1]);

	EXPECT_CALL(*o1, commit_sudoku()).Times(1);
	EXPECT_CALL(*o2, commit_sudoku()).Times(2);
	EXPECT_CALL(*cpy1, commit_sudoku()).Times(1);
	EXPECT_CALL(*cpy2, commit_sudoku()).Times(2);
	EXPECT_CALL(*sassign1, commit_sudoku()).Times(1);
	EXPECT_CALL(*sassign2, commit_sudoku()).Times(2);
	delete s1;
	delete s2;
	delete o1;
	delete o2;
	delete cpy1;
	delete cpy2;
	delete sassign1;
	delete sassign2;
}


// covers iobserver
// move constructor and assignment
TEST(REACT, MoveObs_A006)
{
	FUZZ::delim();
	mock_observer2* sassign1 = new mock_observer2;
	mock_observer2* sassign2 = new mock_observer2;

	mock_subject2* s1 = new mock_subject2;
	mock_subject2* s2 = new mock_subject2;
	mock_observer2* o1 = new mock_observer2(s2);
	mock_observer2* o2 = new mock_observer2(s1, s2);

	mock_observer2* mv1 = new mock_observer2(std::move(*o1));
	mock_observer2* mv2 = new mock_observer2(std::move(*o2));

	std::vector<subject*> subs1 = mv1->expose_dependencies();
	std::vector<subject*> subs2 = mv2->expose_dependencies();
	ASSERT_EQ((size_t) 1, subs1.size());
	EXPECT_EQ(s2, subs1[0]);
	ASSERT_EQ((size_t) 2, subs2.size());
	EXPECT_EQ(s1, subs2[0]);
	EXPECT_EQ(s2, subs2[1]);

	EXPECT_TRUE(o1->expose_dependencies().empty());
	EXPECT_TRUE(o2->expose_dependencies().empty());

	*sassign1 = std::move(*mv1);
	*sassign2 = std::move(*mv2);

	std::vector<subject*> subs3 = sassign1->expose_dependencies();
	std::vector<subject*> subs4 = sassign2->expose_dependencies();
	ASSERT_EQ((size_t) 1, subs3.size());
	EXPECT_EQ(s2, subs3[0]);
	ASSERT_EQ((size_t) 2, subs4.size());
	EXPECT_EQ(s1, subs4[0]);
	EXPECT_EQ(s2, subs4[1]);

	EXPECT_TRUE(mv1->expose_dependencies().empty());
	EXPECT_TRUE(mv2->expose_dependencies().empty());

	EXPECT_CALL(*sassign1, commit_sudoku()).Times(1);
	EXPECT_CALL(*sassign2, commit_sudoku()).Times(2);
	delete s1;
	delete s2;
	delete o1;
	delete o2;
	delete mv1;
	delete mv2;
	delete sassign1;
	delete sassign2;
}


// covers iobserver
// add_dependency
TEST(REACT, AddDep_A007)
{
	FUZZ::delim();
	mock_subject* s1 = new mock_subject;
	mock_subject* s2 = new mock_subject;
	mock_observer2* o1 = new mock_observer2;
	mock_observer2* o2 = new mock_observer2;

	EXPECT_TRUE(s1->no_audience());
	EXPECT_TRUE(s2->no_audience());
	o1->mock_add_dependency(s2);
	EXPECT_FALSE(s2->no_audience());
	o1->mock_add_dependency(s1);
	EXPECT_FALSE(s1->no_audience());

	o2->mock_add_dependency(s1);
	o2->mock_add_dependency(s2);

	std::vector<subject*> subs1 = o1->expose_dependencies();
	std::vector<subject*> subs2 = o2->expose_dependencies();

	ASSERT_EQ((size_t) 2, subs1.size());
	ASSERT_EQ((size_t) 2, subs2.size());

	EXPECT_EQ(s2, subs1[0]);
	EXPECT_EQ(s1, subs1[1]);
	EXPECT_EQ(s1, subs2[0]);
	EXPECT_EQ(s2, subs2[1]);

	EXPECT_CALL(*o1, commit_sudoku()).Times(2);
	EXPECT_CALL(*o2, commit_sudoku()).Times(2);
	delete s1;
	delete s2;
	delete o1;
	delete o2;
}


// covers iobserver
// remove_dependency
TEST(REACT, RemDep_A008)
{
	FUZZ::delim();
	mock_subject2* s1 = new mock_subject2;
	mock_subject2* s2 = new mock_subject2;
	mock_observer2* o1 = new mock_observer2(s2);
	mock_observer2* o2 = new mock_observer2(s1, s2);
	mock_observer2* o3 = new mock_observer2(s1, s2);

	o1->mock_remove_dependency(0);
	EXPECT_TRUE(o1->expose_dependencies().empty());

	o2->mock_remove_dependency(1);
	EXPECT_EQ((size_t) 1, o2->expose_dependencies().size());
	o2->mock_remove_dependency(0);
	EXPECT_TRUE(o2->expose_dependencies().empty());

	o3->mock_remove_dependency(0);
	EXPECT_EQ((size_t) 2, o3->expose_dependencies().size());
	o3->mock_remove_dependency(1);
	EXPECT_TRUE(o3->expose_dependencies().empty());

	delete s1;
	delete s2;
	delete o1;
	delete o2;
	delete o3;
}


// covers iobserver
// replace_dependency
TEST(REACT, RepDep_A009)
{
	FUZZ::delim();
	mock_subject2* s1 = new mock_subject2;
	mock_subject2* s2 = new mock_subject2;
	mock_observer2* o1 = new mock_observer2(s1);

	o1->mock_replace_dependency(nullptr, 1);
	ASSERT_EQ((size_t) 2, o1->expose_dependencies().size());
	o1->mock_replace_dependency(s2, 0);
	std::vector<subject*> subs1 = o1->expose_dependencies();
	ASSERT_EQ((size_t) 2, subs1.size());
	EXPECT_EQ(s2, subs1[0]);
	EXPECT_EQ(nullptr, subs1[1]);
	o1->mock_replace_dependency(s1, 1);
	subs1 = o1->expose_dependencies();
	ASSERT_EQ((size_t) 2, subs1.size());
	EXPECT_EQ(s2, subs1[0]);
	EXPECT_EQ(s1, subs1[1]);

	EXPECT_CALL(*o1, commit_sudoku()).Times(2);
	delete s1;
	delete s2;
	delete o1;
}


// covers iobserver
// destruction, depends on subject detach
TEST(REACT, ObsDeath_A010)
{
	FUZZ::delim();
	mock_subject s1;
	mock_subject2 s2;

	mock_observer2* o1 = new mock_observer2(&s1, &s2);
	iobserver* tempptr = o1;
	EXPECT_CALL(s1, detach(o1));
	delete o1;

	EXPECT_TRUE(s2.no_audience());
	s1.mock_detach(tempptr);
}


#endif /* DISABLE_REACT_TEST */


#endif /* DISABLE_GRAPH_MODULE_TESTS */
