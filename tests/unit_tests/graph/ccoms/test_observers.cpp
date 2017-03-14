//
// Created by Mingkai Chen on 2016-11-13.
//

#include "mock_ccoms.h"
using ::testing::_;


// Behavior E000
TEST(react, DetachFromSubjectOnDeath_E000)
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
TEST(react, CopyDep_E001)
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
