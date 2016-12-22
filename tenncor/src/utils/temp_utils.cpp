#include "utils/temp_utils.hpp"

#ifdef temp_hpp

namespace r_temp
{

std::string temp_uuid (const void* addr)
{
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << addr << now_c;
    
    return ss.str();
}

}

#endif