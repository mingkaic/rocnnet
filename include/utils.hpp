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

namespace nnutils {

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