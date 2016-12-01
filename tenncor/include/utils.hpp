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
#include <memory>

#pragma once
#ifndef utils_hpp
#define utils_hpp

namespace nnutils
{

class formatter
{
	private:
		std::stringstream stream_;

		formatter(const formatter&);
		formatter& operator = (formatter&);
	public:
		enum convert_to_string
		{
			to_str
		};

		formatter() {}
		~formatter() {}

		template <typename T>
		formatter& operator << (const T& value)
		{
			stream_ << value;
			return *this;
		}

		std::string str() const
		{
			return stream_.str();
		}

		operator std::string () const
		{
			return stream_.str();
		}

		std::string operator >> (convert_to_string)
		{
			return stream_.str();
		}
};

template <typename T, typename U>
std::vector<U> to_vec (std::vector<T> vec, std::function<U(T)> convert)
{
	std::vector<U> res;
	for (T v : vec)
	{
		res.push_back(convert(v));
	}
	return res;
}

}

#endif /* utils_hpp */