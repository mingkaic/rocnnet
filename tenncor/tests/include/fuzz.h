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
template <typename T>
struct FUZZ
{
	static std::vector<T> get (size_t len, std::pair<T,T> range={0,0})
	{
		T min, max;
		if (range.first == range.second)
		{
			min = std::numeric_limits<T>::min();
			max = std::numeric_limits<T>::max();
		}
		else
		{
			min = range.first;
			max = range.second;
		}
		std::default_random_engine generator;

		std::vector<T> vec;
		if (typeid(T) == typeid(size_t))
		{
			std::uniform_int_distribution<T> dis(min, max);
			for (size_t i = 0; i < len; i++)
			{
				vec.push_back(dis(generator));
			}
		}
		else
		{
			std::uniform_real_distribution<T> dis(min, max);
			for (size_t i = 0; i < len; i++)
			{
				vec.push_back((T) dis(generator));
			}
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
		std::vector<size_t> indices = FUZZ<size_t>::get(len, {0, alphanum.size()-1});
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
