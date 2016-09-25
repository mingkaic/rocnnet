//
//  main.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <iostream>
#include <ctime>
#if defined(TESTFLAG) || defined(TENSOR_TEST)
	#include <gtest/gtest.h>
	#include "tests/test_tensorshape.cpp"
	#include "tests/test_variable.cpp"
	#include "tests/test_operation.cpp"
#endif

#if defined(TESTFLAG) || defined(LAYER_TEST)
	#include <gtest/gtest.h>
	#include "tests/test_nnlayer.cpp"
#endif

int main(int argc, char * argv[]) {
	srand(time(NULL));
	#ifdef SPTEST
		SP_RTInit();
	#endif

	#if defined(TESTFLAG) || defined(TENSOR_TEST) || defined(LAYER_TEST)
		testing::InitGoogleTest(&argc, argv);
		int err = RUN_ALL_TESTS();
		std::cout << "running test\n";
		return err;
	#endif

    return 0;
}
