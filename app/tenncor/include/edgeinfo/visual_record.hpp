//
//  visual_record.hpp
//  cnnet
//
//  Created by Ming Kai Chen on 2017-11-07.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#include "edgeinfo/igraph_record.hpp"

#pragma once
#ifdef VIS_RCD

#ifndef visual_record_hpp
#define visual_record_hpp

namespace rocnnet_record
{

class visual_record final : public igraph_record
{
public:
	visual_record (void) {}

	~visual_record (void) {}

private:
};

}

#endif /* visual_record_hpp */

#endif /* VIS_RCD */
