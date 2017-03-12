//
//  utils.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <string>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <memory>
#include <vector>

#pragma once
#ifndef TENNCOR_UTILS_HPP
#define TENNCOR_UTILS_HPP

namespace nnutils
{

class formatter
{
public:
	enum convert_to_string
	{
		to_str
	};

	formatter (void);

	~formatter (void);

	// no copying
	formatter (const formatter&) = delete;
	formatter (formatter&&) = delete;
	formatter& operator = (const formatter&) = delete;
	formatter& operator = (formatter&&) = delete;

	template <typename T>
	formatter& operator << (const T& value)
	{
		stream_ << value;
		return *this;
	}

	template <typename T>
	formatter& operator << (const std::vector<T>& values)
	{
		auto it = values.begin();
		stream_ << *it;
		it++;
		while (it != values.end())
		{
			stream_ << "," << *it;
			it++;
		}
		return *this;
	}

	std::string str(void) const;

	operator std::string () const;

	std::string operator >> (convert_to_string);

private:
	std::stringstream stream_;
};

// generates a "unique" string based on pointer and current time
std::string uuid (const void* addr);

}

#endif /* TENNCOR_UTILS_HPP */