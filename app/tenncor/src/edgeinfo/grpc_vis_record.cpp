//
// Created by Ming Kai Chen on 2017-11-07.
//

#include "edgeinfo/grpc_vis_record.hpp"

#ifdef grpc_vis_record_hpp

namespace rocnnet_record
{

#ifdef RPC_RCD

std::unique_ptr<igraph_record> record_status::rec =
	std::make_unique<rpc_record>("localhost", 10980);
bool record_status::rec_good = true;

#endif /* RPC_RCD */

}

#endif
