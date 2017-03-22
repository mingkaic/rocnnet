//
// Created by Mingkai Chen on 2017-03-09.
//

#include <random>
#include <vector>
#include <unordered_set>
#include <cstdlib>
#include <limits>
#include <string>

#ifndef TENNCOR_FUZZ_H
#define TENNCOR_FUZZ_H

// todo: increase test iteration (not really fuzz testing if we only test with 1 set of inputs)
// temporary fuzzer interface
struct FUZZ
{
	static std::vector<double> getDouble (size_t len, std::pair<double,double> range={0,0})
	{
		double min, max;
		if (range.first == range.second)
		{
			min = std::numeric_limits<double>::min();
			max = std::numeric_limits<double>::max();
		}
		else
		{
			min = range.first;
			max = range.second;
		}
		std::default_random_engine generator;

		std::vector<double> vec;
		std::uniform_real_distribution<double> dis(min, max);
		for (size_t i = 0; i < len; i++)
		{
			vec.push_back((double) dis(generator));
		}

		return vec;
	}

	static std::vector<size_t> getInt (size_t len, std::pair<size_t,size_t> range={0,0})
	{
		size_t min, max;
		if (range.first == range.second)
		{
			min = std::numeric_limits<size_t>::min();
			max = std::numeric_limits<size_t>::max();
		}
		else
		{
			min = range.first;
			max = range.second;
		}
		std::default_random_engine generator;

		std::vector<size_t> vec;
		std::uniform_int_distribution<size_t> dis(min, max);
		for (size_t i = 0; i < len; i++)
		{
			vec.push_back(dis(generator));
		}

		return vec;
	}
};

struct FUZZ_STRING
{
	static std::string get (size_t len, std::string alphanum =
		"0123456789!@#$%^&*"
		"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		"abcdefghijklmnopqrstuvwxyz")
	{
		std::vector<size_t> indices = FUZZ::getInt(len, {0, alphanum.size()-1});
		std::string s(len, ' ');
		std::transform(indices.begin(), indices.end(), s.begin(),
		[&alphanum](size_t index)
		{
			return alphanum[index];
		});
		return s;
	}
};

#endif //TENNCOR_FUZZ_H
