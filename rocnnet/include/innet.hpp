//
//  innet.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-01-05.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef innet_hpp
#define innet_hpp

#include <string>

namespace nnet
{

class innet
{
	protected:
		std::string scope_;

	public:
		innet (std::string scope = "") : scope_(scope) {}
		virtual ~innet (void) {}

		// COPY
		virtual innet* clone (std::string scope = "") = 0;

		// MOVE
};

}

#endif /* innet_hpp */
