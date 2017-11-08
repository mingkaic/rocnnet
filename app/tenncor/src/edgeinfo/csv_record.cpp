//
// Created by Mingkai Chen on 2017-01-30.
//

#include "edgeinfo/csv_record.hpp"

#ifdef csv_record_hpp

namespace rocnnet_record
{

#ifdef CSV_RCD

std::unique_ptr<igraph_record> record_status::rec = std::make_unique<csv_record>(CSV_RCD);
bool record_status::rec_good = true;

#endif

}

#endif