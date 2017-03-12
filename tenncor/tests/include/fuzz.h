//
// Created by Mingkai Chen on 2017-03-09.
//

#include <random>
#include <vector>
#include <unordered_set>
#include <cstdlib>
#include <limits>

#ifndef TENNCOR_FUZZ_H
#define TENNCOR_FUZZ_H

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
		std::uniform_real_distribution<double> dis(min, max);

		std::vector<T> vec;
		for (size_t i = 0; i < len; i++)
		{
			vec.push_back((T) dis(generator));
		}

		return vec;
	}

	static std::vector<T> get (size_t len, std::pair<T,T> range, std::unordered_set<T> ignore)
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
		std::uniform_real_distribution<double> dis(min, max);

		std::vector<T> vec;
		size_t i = 0;
		while (i < len)
		{
			T v = (T) dis(generator);
			if (ignore.end() == ignore.find(v))
			{
				vec.push_back(v);
				i++;
			}
		}

		return vec;
	}
};

#endif //TENNCOR_FUZZ_H
