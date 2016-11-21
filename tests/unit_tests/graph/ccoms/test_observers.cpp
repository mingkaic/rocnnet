//
// Created by Mingkai Chen on 2016-11-13.
//

#include "mock_ccoms.h"
using ::testing::_;


// Behavior E000
TEST(CCOMS, DetachFromSubjectOnDeath_E000)
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
TEST(CCOMS, ObserverLeaf_E001)
{
	MockSubject solo;
	MockSubject leaf1;
	MockSubject leaf2;

	MockObserver* branch1 = MockObserver::build(&solo);
	MockObserver* branch2 = MockObserver::build(&leaf1, &leaf2);

	EXPECT_CALL(solo, detach(branch1));
	EXPECT_CALL(leaf1, detach(branch2));
	EXPECT_CALL(leaf2, detach(branch2));

	size_t count = 0;
	branch1->leaves_collect(
	[&count, &solo](ccoms::subject* subject)
	{
		EXPECT_TRUE(subject == &solo);
		count++;
	});
	EXPECT_EQ(1, count);
	count = 0;
	branch2->leaves_collect(
	[&count, &leaf1, &leaf2](ccoms::subject* subject)
	{
		EXPECT_TRUE(subject == &leaf1 ||
					subject == &leaf2);
		count++;
	});
	EXPECT_EQ(2, count);
	delete branch1;
	delete branch2;
	// real detach
	solo.mock_detach(branch1);
	leaf1.mock_detach(branch2);
	leaf2.mock_detach(branch2);
}


// Behavior E002
TEST(CCOMS, CopyDep_E002)
{

}

// 
//
//// expected behavior:
//// for all attached observers,
//// notifying subject updates all observers exactly once
//TEST(CCOMS, NotifyAndUpdate)
//{
//	mock_subject subject;
//	mock_observer o1(&subject);
//	mock_observer o2(&subject);
//	mock_observer o3(&subject);
//	// subject is now observed 3 times
//	// triggering subject notify will call each update 3 times
//	EXPECT_CALL(o1, update(_)).Times(1);
//	EXPECT_CALL(o2, update(_)).Times(1);
//	EXPECT_CALL(o3, update(_)).Times(1);
//
//	subject.notify();
//}
//
//TEST(CCOMS, LeafCollection)
//{
//	mock_subject leaf1;
//	mock_subject leaf2;
//	mock_subject leaf3;
//	mock_subject leaf4;
//	mock_subject leaf5;
//	EXPECT_CALL(leaf1, merge_leaves(_));
//	EXPECT_CALL(leaf2, merge_leaves(_)).Times(2);
//	EXPECT_CALL(leaf3, merge_leaves(_));
//	EXPECT_CALL(leaf4, merge_leaves(_));
//	EXPECT_CALL(leaf5, merge_leaves(_));
//
//	mock_intern branch1(&leaf1, &leaf2);
//	mock_intern branch2(&leaf4);
//	mock_intern branch3(&branch1, &leaf3);
//	mock_intern branch4(&leaf2, &leaf5);
//	mock_intern branch5(&branch3, &branch4); // test for set join
//	// branch1 has leaves 1, 2
//	// branch2 has leaves 4
//	// branch3 has leaves 1, 2, 3
//	// branch4 has leaves 2, 5
//	// branch5 has leaves 1, 2, 3, 5
//	// leaf1, leaf3, leaf4, leaf5 are called once
//	// leaf2 is called twice
//	size_t count = 0;
//	branch1.leaves_collect(
//	[&count, &leaf1, &leaf2](ccoms::subject* subject)
//	{
//		EXPECT_TRUE(subject == &leaf1 ||
//					subject == &leaf2);
//		count++;
//	});
//	EXPECT_EQ(2, count);
//	count = 0;
//	branch2.leaves_collect(
//	[&count, &leaf4](ccoms::subject* subject)
//	{
//		EXPECT_TRUE(subject == &leaf4);
//		count++;
//	});
//	EXPECT_EQ(1, count);
//	count = 0;
//	branch3.leaves_collect(
//	[&count, &leaf1, &leaf2, &leaf3](ccoms::subject* subject)
//	{
//		EXPECT_TRUE(subject == &leaf1 ||
//					subject == &leaf2 ||
//					subject == &leaf3);
//		count++;
//	});
//	EXPECT_EQ(3, count);
//	count = 0;
//	branch4.leaves_collect(
//	[&count, &leaf2, &leaf5](ccoms::subject* subject)
//	{
//		EXPECT_TRUE(subject == &leaf2 ||
//					subject == &leaf5);
//		count++;
//	});
//	EXPECT_EQ(2, count);
//	count = 0;
//	branch5.leaves_collect(
//	[&count, &leaf1, &leaf2, &leaf3, &leaf5](ccoms::subject* subject)
//	{
//		EXPECT_TRUE(subject == &leaf1 ||
//					subject == &leaf2 ||
//					subject == &leaf3 ||
//					subject == &leaf5);
//		count++;
//	});
//	EXPECT_EQ(4, count);
//}
