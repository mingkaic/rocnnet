#include <iostream>

#include "utils/utils.hpp"

#ifdef TENNCOR_UTILS_HPP

namespace nnutils
{

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

std::string uuid (const void* addr)
{
	static std::uniform_int_distribution<size_t> tok_dist(0, 15);
	auto now = std::chrono::system_clock::now();
	std::time_t now_c = std::chrono::system_clock::to_time_t(now);

	std::stringstream ss;
	ss << std::hex << now_c << (size_t) addr;

	for (size_t i = 0; i < 16; i++)
	{
		size_t token = tok_dist(get_generator());
		ss << std::hex << token;
	}
	return ss.str();
}

std::default_random_engine& get_generator (void)
{
	static std::default_random_engine common_generator(std::time(nullptr));
	return common_generator;
}

void seed_generator (size_t val)
{
	get_generator().seed(val);
}

}

#endif