//
//  toggle.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-19.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "iselector.hpp"

#pragma once
#ifndef toggle_hpp
#define toggle_hpp

namespace nnet
{

// unlike selector, toggle between 2 states
// changes state via swap accessor method
// active variable is used when activation method is called
// otherwise dependencies will always default variable

template <typename T>
class toggle : public iselector<T>
{
	private:
		void reset (void) { this->active_ = 0; }
		
	protected:
		toggle (ivariable<T>* def, ivariable<T>* active, std::string name) :
			iselector<T>(std::vector<ivariable<T>*>{def, active}, name) {}
			
	public:
		static toggle<T>* build (ivariable<T>* def, ivariable<T>* active, std::string name = "")
		{
			return new toggle(def, active, name);
		}
		
		// COPY
		virtual toggle<T>* clone (void) { return new toggle<T>(*this); }
	
		virtual void activate (void)
		{
			this->active_ = 1;
			update(ccoms::caller_info());
		}
		
		virtual void update (ccoms::caller_info info, ccoms::update_message msg = ccoms::update_message())
		{
			std::vector<ivariable<T>*> args =
				nnutils::to_vec<ccoms::subject*, ivariable<T>*>(this->dependencies_, sub_to_var<T>);
			msg.grad_ = nullptr;
			this->notify(msg);
			reset();
		}
};

}

#endif /* toggle_hpp */