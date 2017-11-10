//
// Created by Ming Kai Chen on 2017-11-08.
//

#include "edgeinfo/igraph_record.hpp"

#ifdef igraph_record_hpp

namespace rocnnet_record
{

igraph_record::~igraph_record (void)
{
	record_status::rec_good = false;
}

}

#endif
