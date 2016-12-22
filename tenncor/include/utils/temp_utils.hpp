// hopefully "temporary" fixes

#include <chrono>
#include <string>
#include <sstream>

#pragma once
#ifndef temp_hpp
#define temp_hpp

namespace r_temp
{

// generates a "unique" string based on pointer and current time
std::string temp_uuid (const void* addr);

}

#endif /* temp_hpp */