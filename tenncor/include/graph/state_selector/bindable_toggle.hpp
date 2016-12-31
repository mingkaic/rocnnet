//
//  bindable_toggle.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-19.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "executor/ibindable.hpp"
#include "push_toggle.hpp"

#pragma once
#ifndef bindable_toggle_hpp
#define bindable_toggle_hpp

namespace nnet
{

// TODO replace all gradient nodes with bindable_toggle

template <typename T>
class bindable_toggle : public ibindable<T>, public push_toggle<T>
{
	private:
		// we need this to remind ourselves that active state is pushed to the audiences
		// this is necessary due to the "push" nature;
		// active state is reset after every activate, so we can't check this->active_
		bool last_active_ = false;

	protected:
		bindable_toggle (ivariable<T>* def, ivariable<T>* active, std::string name) :
			push_toggle<T>(def, active, name) {}

	public:
		static bindable_toggle<T>* build (ivariable<T>* def, ivariable<T>* active, std::string name = "")
		{
			return new bindable_toggle<T>(def, active, name);
		}

	    // activate this, force deactivate all bounded toggles
	    virtual void activate (std::string tid = "")
		{
		    this->declare_active(tid);
		    push_toggle<T>::activate();
			last_active_ = true;
		}
		
		virtual void deactivate (std::string tid)
		{
			// update audience data
		    if (true == last_active_)
		    {
    			this->update(ccoms::caller_info());
				last_active_ = false;
		    }
		}
};

}

#endif /* bindable_toggle_hpp */