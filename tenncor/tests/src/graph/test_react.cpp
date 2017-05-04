//
// Created by Mingkai Chen on 2017-03-10.
//

#ifndef DISABLE_GRAPH_MODULE_TESTS

#include <algorithm>

#include "gtest/gtest.h"

#include "mocks/mock_subject.h"
#include "mocks/mock_observer.h"
#include "fuzz.h"


#ifndef DISABLE_REACT_TEST


// covers subject
// copy constructor and assignment,
// no audience, attach, detach
TEST(REACT, CopySub_A000)
{
	FUZZ::reset_logger();
	mock_subject sassign1;
	mock_subject sassign2;

	mock_subject s1;
	mock_subject s2;
	mock_observer o1(&s2);
	
	std::vector<subject*> subjects = o1.expose_dependencies();
	ASSERT_EQ((size_t) 1, subjects.size());

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
	FUZZ::reset_logger();
	mock_subject sassign1;
	mock_subject sassign2;

	mock_subject s1;
	mock_subject s2;
	mock_observer o1(&s2);

	std::vector<subject*> subjects = o1.expose_dependencies();
	ASSERT_EQ((size_t) 1, subjects.size());

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
	mocker::usage_.clear();
	FUZZ::reset_logger();
	mock_subject s1;
	mock_subject s2;
	mock_observer o1(&s1, &s2);
	mock_observer o2(&s2);
	
	o1.inst_ = "o1";
	o2.inst_ = "o2";

	std::vector<subject*> subjects = o1.expose_dependencies();
	std::vector<subject*> subjects2 = o2.expose_dependencies();
	ASSERT_EQ((size_t) 2, subjects.size());
	ASSERT_EQ((size_t) 1, subjects2.size());

	s1.notify(UPDATE); // o1 update gets s1 at idx 0
	s2.notify(UPDATE);

	// o2 update gets s2 at idx 0,
	// o1 update gets s2 at idx 1
	EXPECT_TRUE(mocker::EXPECT_CALL("o1::update2", 2)); // todo: check receiving argument UPDATE
	EXPECT_TRUE(mocker::EXPECT_CALL("o2::update2", 1));
	mocker::usage_.clear();

	// suicide calls
	s1.notify(UNSUBSCRIBE);
	s2.notify(UNSUBSCRIBE);

	EXPECT_TRUE(mocker::EXPECT_CALL("o1::update2", 2)); // todo: check receiving argument UNSUBSCRIBE
	EXPECT_TRUE(mocker::EXPECT_CALL("o2::update2", 1));

	// detach to avoid doing anything with o1 and o2
	s1.mock_detach(&o1);
	s2.mock_detach(&o1);
	s2.mock_detach(&o2);
}


// covers subject
// destructor
TEST(REACT, SUBDEATH_A002)
{
	mocker::usage_.clear();
	FUZZ::reset_logger();
	mock_subject* s1 = new mock_subject;
	mock_subject* s2 = new mock_subject;
	mock_observer o1(s1, s2);
	mock_observer o2(s2);

	std::vector<subject*> subjects = o1.expose_dependencies();
	std::vector<subject*> subjects2 = o2.expose_dependencies();
	ASSERT_EQ((size_t) 2, subjects.size());
	ASSERT_EQ((size_t) 1, subjects2.size());

	o1.inst_ = "o1";
	o2.inst_ = "o2";

	// suicide calls
	delete s1;
	EXPECT_TRUE(mocker::EXPECT_CALL("o1::update2", 1)); // todo: check receiving argument UPDATE
	mocker::usage_.clear();

	delete s2;
	EXPECT_TRUE(mocker::EXPECT_CALL("o1::update2", 1));
	EXPECT_TRUE(mocker::EXPECT_CALL("o2::update2", 1));
}


// covers subject
// attach
TEST(REACT, Attach_A003)
{
	FUZZ::reset_logger();
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
	FUZZ::reset_logger();
	mock_subject s1;
	mock_subject s2;
	mock_subject s3;
	mock_observer o1(&s1);
	mock_observer o2({&s1, &s2, &s2});
	mock_observer o3(&s3, &s3);

	EXPECT_FALSE(s1.no_audience());
	s1.mock_detach(&o1);
	EXPECT_FALSE(s1.no_audience());
	s1.mock_detach(&o2);
	EXPECT_TRUE(s1.no_audience());

	EXPECT_FALSE(s2.no_audience());
	s2.mock_detach(&o2);
	EXPECT_TRUE(s2.no_audience());

	EXPECT_FALSE(s3.no_audience());
	s3.mock_detach(&o3, 1);
	EXPECT_FALSE(s3.no_audience());
	s3.mock_detach(&o3, 0);
	EXPECT_TRUE(s3.no_audience());

	o1.mock_clear_dependency();
	o2.mock_clear_dependency();
	o3.mock_clear_dependency();
}


// covers iobserver
// default and dependency constructors
TEST(REACT, ObsConstruct_A005)
{
	mocker::usage_.clear();
	FUZZ::reset_logger();
	mock_subject* s1 = new mock_subject;
	mock_subject* s2 = new mock_subject;
	mock_observer2* o1 = new mock_observer2(s2);
	mock_observer2* o2 = new mock_observer2(s1, s2);
	o1->inst_ = "o1";
	o2->inst_ = "o2";

	std::vector<subject*> subs1 = o1->expose_dependencies();
	std::vector<subject*> subs2 = o2->expose_dependencies();

	ASSERT_EQ((size_t) 1, subs1.size());
	EXPECT_EQ(s2, subs1[0]);
	ASSERT_EQ((size_t) 2, subs2.size());
	EXPECT_EQ(s1, subs2[0]);
	EXPECT_EQ(s2, subs2[1]);

	// called twice since mock observer isn't destroyed when commit_sudoku is called
	// so deleting s2 will trigger another suicide call
	delete s1;
	delete s2;
	// again observers aren't destroyed
	delete o1;
	delete o2;

	EXPECT_TRUE(mocker::EXPECT_CALL("o1::commit_sudoku", 1));
	EXPECT_TRUE(mocker::EXPECT_CALL("o2::commit_sudoku", 2));
}


// covers iobserver
// copy constructor and assignment
TEST(REACT, CopyObs_A006)
{
	mocker::usage_.clear();
	FUZZ::reset_logger();
	mock_observer2* sassign1 = new mock_observer2;
	mock_observer2* sassign2 = new mock_observer2;

	mock_subject* s1 = new mock_subject;
	mock_subject* s2 = new mock_subject;
	mock_observer2* o1 = new mock_observer2(s2);
	mock_observer2* o2 = new mock_observer2(s1, s2);

	mock_observer2* cpy1 = new mock_observer2(*o1);
	mock_observer2* cpy2 = new mock_observer2(*o2);

	o1->inst_ = "o1";
	o2->inst_ = "o2";
	cpy1->inst_ = "cpy1";
	cpy2->inst_ = "cpy2";
	sassign1->inst_ = "sassign1";
	sassign2->inst_ = "sassign2";

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

	delete s1;
	delete s2;
	delete o1;
	delete o2;
	delete cpy1;
	delete cpy2;
	delete sassign1;
	delete sassign2;
	EXPECT_TRUE(mocker::EXPECT_CALL("o1::commit_sudoku", 1));
	EXPECT_TRUE(mocker::EXPECT_CALL("o2::commit_sudoku", 2));
	EXPECT_TRUE(mocker::EXPECT_CALL("cpy1::commit_sudoku", 1));
	EXPECT_TRUE(mocker::EXPECT_CALL("cpy2::commit_sudoku", 2));
	EXPECT_TRUE(mocker::EXPECT_CALL("sassign1::commit_sudoku", 1));
	EXPECT_TRUE(mocker::EXPECT_CALL("sassign2::commit_sudoku", 2));
}


// covers iobserver
// move constructor and assignment
TEST(REACT, MoveObs_A006)
{
	mocker::usage_.clear();
	FUZZ::reset_logger();
	mock_observer2* sassign1 = new mock_observer2;
	mock_observer2* sassign2 = new mock_observer2;

	mock_subject2* s1 = new mock_subject2;
	mock_subject2* s2 = new mock_subject2;
	mock_observer2* o1 = new mock_observer2(s2);
	mock_observer2* o2 = new mock_observer2(s1, s2);

	mock_observer2* mv1 = new mock_observer2(std::move(*o1));
	mock_observer2* mv2 = new mock_observer2(std::move(*o2));

	o1->inst_ = "o1";
	o2->inst_ = "o2";
	mv1->inst_ = "mv1";
	mv2->inst_ = "mv2";
	sassign1->inst_ = "sassign1";
	sassign2->inst_ = "sassign2";

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

	delete s1;
	delete s2;
	delete o1;
	delete o2;
	delete mv1;
	delete mv2;
	delete sassign1;
	delete sassign2;
	EXPECT_TRUE(mocker::EXPECT_CALL("o1::commit_sudoku", 0));
	EXPECT_TRUE(mocker::EXPECT_CALL("o2::commit_sudoku", 0));
	EXPECT_TRUE(mocker::EXPECT_CALL("mv1::commit_sudoku", 0));
	EXPECT_TRUE(mocker::EXPECT_CALL("mv2::commit_sudoku", 0));
	EXPECT_TRUE(mocker::EXPECT_CALL("sassign1::commit_sudoku", 1));
	EXPECT_TRUE(mocker::EXPECT_CALL("sassign2::commit_sudoku", 2));
}


// covers iobserver
// add_dependency
TEST(REACT, AddDep_A007)
{
	mocker::usage_.clear();
	FUZZ::reset_logger();
	mock_subject* s1 = new mock_subject;
	mock_subject* s2 = new mock_subject;
	mock_observer2* o1 = new mock_observer2;
	mock_observer2* o2 = new mock_observer2;

	o1->inst_ = "o1";
	o2->inst_ = "o2";

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

	delete s1;
	delete s2;
	delete o1;
	delete o2;
	EXPECT_TRUE(mocker::EXPECT_CALL("o1::commit_sudoku", 2));
	EXPECT_TRUE(mocker::EXPECT_CALL("o2::commit_sudoku", 2));
}


// covers iobserver
// remove_dependency
TEST(REACT, RemDep_A008)
{
	FUZZ::reset_logger();
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
	mocker::usage_.clear();
	FUZZ::reset_logger();
	mock_subject2* s1 = new mock_subject2;
	mock_subject2* s2 = new mock_subject2;
	mock_observer2* o1 = new mock_observer2(s1);

	o1->inst_ = "o1";

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

	delete s1;
	delete s2;
	delete o1;
	EXPECT_TRUE(mocker::EXPECT_CALL("o1::commit_sudoku", 2));
}


// covers iobserver
// destruction, depends on subject detach
TEST(REACT, ObsDeath_A010)
{
	mocker::usage_.clear();
	FUZZ::reset_logger();
	mock_subject s1;
	mock_subject2 s2;

	s1.inst_ = "s1";

	mock_observer2* o1 = new mock_observer2(&s1, &s2);
	iobserver* tempptr = o1;
	delete o1;
	EXPECT_TRUE(mocker::EXPECT_CALL("s1::detach1", 1)); // todo: verify detach argument is o1

	EXPECT_TRUE(s2.no_audience());
	s1.mock_detach(tempptr);
}


#endif /* DISABLE_REACT_TEST */


#endif /* DISABLE_GRAPH_MODULE_TESTS */
