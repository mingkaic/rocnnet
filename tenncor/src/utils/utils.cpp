#include "utils/utils.hpp"

#ifdef TENNCOR_UTILS_HPP

namespace nnutils
{

std::string uuid (const void* addr)
{
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::hex << now_c << (size_t) addr;
    
    return ss.str();
}

formatter::formatter (void) {}

formatter::~formatter (void) {}

std::string formatter::str (void) const
{
	return stream_.str();
}

formatter::operator std::string () const
{
	return stream_.str();
}

std::string formatter::operator >> (convert_to_string)
{
	return stream_.str();
}

}

#endif