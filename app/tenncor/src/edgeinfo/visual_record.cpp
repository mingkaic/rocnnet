//
// Created by Ming Kai Chen on 2017-11-07.
//

#include "edgeinfo/visual_record.hpp"

#ifdef visual_record_hpp

namespace rocnnet_record
{

#ifdef VIS_RCD

std::unique_ptr<igraph_record> record_status::rec = std::make_unique<visual_record>();
bool record_status::rec_good = true;

#endif

}

#endif
