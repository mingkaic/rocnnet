/*!
 *
 *  utils.hpp
 *  cnnet
 *
 *  Purpose:
 *  utils contain commonly used stream formatter, and uuid generator
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include <string>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <memory>
#include <vector>
#include <ctime>
#include <chrono>

#pragma once
#ifndef TENNCOR_UTILS_HPP
#define TENNCOR_UTILS_HPP

namespace nnutils
{

//! stream formatter
class formatter
{
public:
	//! to_stream conversion enum trick
	enum convert_to_string
	{
		to_str
	};

	//! default constructor
	formatter (void);

	//! default destructor
	~formatter (void);

	//! no copying or moving to force string conversion
	//! format content is always passed as a string
	formatter (const formatter&) = delete;
	formatter (formatter&&) = delete;
	formatter& operator = (const formatter&) = delete;
	formatter& operator = (formatter&&) = delete;

	//! overload << operator for non-vector values to add to stream
	template <typename T>
	formatter& operator << (const T& value)
	{
		stream_ << value;
		return *this;
	}

	//! overload << operator for vectors
	//! add values to string delimited by comma
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

	//! explicit string conversion function
	std::string str(void) const;

	//! implicit string converter
	operator std::string () const;

	//! out stream as string
	std::string operator >> (convert_to_string);

private:
	std::stringstream stream_; // internal stream
};

//! generates a "unique" string based on input address and current time
//! uses cstdlib rand, so use srand to seed
// todo: upgrade to random generator then expose generator for seeding
std::string uuid (const void* addr);

}

#endif /* TENNCOR_UTILS_HPP */