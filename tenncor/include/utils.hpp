//
//  utils.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef utils_hpp
#define utils_hpp

#include <string>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <memory>

namespace nnutils {

template<typename T>
struct weak_ptr_hash : public std::unary_function<std::weak_ptr<T>, size_t> {
	size_t operator () (const std::weak_ptr<T>& wp) {
		auto sp = wp.lock();
		return std::hash<decltype(sp)>()(sp);
	}
};

template<typename T>
struct weak_ptr_equal : public std::unary_function<std::weak_ptr<T>, bool> {
	bool operator () (const std::weak_ptr<T>& left, const std::weak_ptr<T>& right) {
		return !left.owner_before(right) && !right.owner_before(left);
	}
};

template <typename T>
using WEAK_SET = std::unordered_set<std::weak_ptr<T>, weak_ptr_hash<T>, weak_ptr_equal<T> >;

class formatter {
	private:
		std::stringstream _stream;

		formatter(const formatter &);
		formatter & operator = (formatter &);
	public:
		enum convert_to_string {
			to_str
		};

		formatter() {}
		~formatter() {}

		template <typename T>
		formatter & operator << (const T & value) {
			_stream << value;
			return *this;
		}

		std::string str() const {
			return _stream.str();
		}

		operator std::string () const {
			return _stream.str();
		}

		std::string operator >> (convert_to_string) {
			return _stream.str();
		}
};

}

#endif /* utils_hpp */