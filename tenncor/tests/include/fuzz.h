//
// Created by Mingkai Chen on 2017-03-09.
//

#include <random>
#include <vector>
#include <unordered_set>
#include <cstdlib>
#include <limits>
#include <string>
#include <iostream>
#include <fstream>

#ifndef TENNCOR_FUZZ_H
#define TENNCOR_FUZZ_H

namespace FUZZ
{

void delim (void);

std::vector<double> getDouble (size_t len, 
	std::pair<double,double> range={0,0});

std::vector<size_t> getInt (size_t len, 
	std::pair<size_t,size_t> range={0,0});

std::string getString (size_t len, 
	std::string alphanum =
	"0123456789!@#$%^&*"
	"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	"abcdefghijklmnopqrstuvwxyz");

}

#endif //TENNCOR_FUZZ_H
