//
// Created by Mingkai Chen on 2016-11-25.
//

#include <iostream>

#pragma once
#ifndef ROCNNET_TEST_UTIL_H
#define ROCNNET_TEST_UTIL_H


void print (std::vector<double> raw)
{
	for (double r : raw)
	{
		std::cout << r << " ";
	}
	std::cout << "\n";
}


#endif //ROCNNET_TEST_UTIL_H
