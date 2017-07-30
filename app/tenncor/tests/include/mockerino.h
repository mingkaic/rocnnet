//
// Created by Mingkai Chen on 2017-04-18.
//
// Implement custom mocker class to test call expectations
//

#include <string>
#include <unordered_map>

#ifndef TENNCOR_MOCKERINO_H
#define TENNCOR_MOCKERINO_H

struct mocker
{
	void label_incr (std::string key) const
	{
		std::string realkey = inst_+"::"+key;
		if (mocker::usage_.end() == mocker::usage_.find(realkey))
		{
			mocker::usage_[realkey] = 0;
		}
		mocker::usage_[realkey]++;
	}

	std::string inst_ = "";

	static std::unordered_map<std::string, size_t> usage_;

	static bool EXPECT_CALL (std::string key, size_t times)
	{
		return usage_[key] == times;
	}
};

#endif //TENNCOR_MOCKERINO_H
