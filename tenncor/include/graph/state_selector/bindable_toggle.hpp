//
//  bindable_toggle.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-19.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "executor/ibindable.hpp"
#include "toggle.hpp"

#pragma once
#ifndef bindable_toggle_hpp
#define bindable_toggle_hpp

namespace nnet
{

// TODO replace all gradient nodes with bindable_toggle

template <typename T>
class bindable_toggle : public ibindable<T>, public toggle<T>
{
	public:
	    // activate this, force deactivate all bounded toggles
	    virtual void activate (void)
		{
		    declare_active();
		    toggle<T>::activate();
		}
		
		virtual void deactivate (void)
		{
		    if (this->active_)
		    {
    		    reset (void);
    			update(ccoms::caller_info());
		    }
		}
};

}

#endif /* bindable_toggle_hpp */