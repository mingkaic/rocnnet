//
//  buffer.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-01.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/iconnector.hpp"

#pragma once
#ifndef buffer_hpp
#define buffer_hpp

namespace nnet
{

// basically same as single argument operations
// except it doesn't do anything and it can remove dependencies
// death to the current dependency auto kills buffer
template <typename T>
class buffer : public iconnector<T>
{
	protected:
		void change_dep (ivariable<T>* var)
		{
			this->replace_dep(var_to_sub<T>(var), 0);
			// leaf update
			this->leaves_update();
			// actual update which notifies audiences asking them to leaf notify
			ccoms::update_message msg;
			msg.leave_update_ = true;
			this->update(ccoms::caller_info(), msg); // updates leaves
		}

		buffer (const buffer<T>& other, std::string name);
		virtual ivariable<T>* clone_impl (std::string name);

		buffer (ivariable<T>* leaf, std::string name);

	public:
		static buffer<T>* build (ivariable<T>* leaf, std::string name = "buffer")
		{
			return new buffer(leaf, name);
		}

		virtual ~buffer (void) {}

		// COPY
		buffer<T>* clone (std::string name = "");
		buffer<T>& operator = (const buffer<T>& other);

		// data assignment
		buffer<T>& operator = (ivariable<T>& ivar);

		// MOVE

		// get stored dependency
		virtual ivariable<T>* get (void) const;

		// implemented from ivariable
		virtual tensorshape get_shape (void) const;
		virtual tensor<T>* get_eval (void);
		virtual ivariable<T>* get_gradient (void);

		// directly pass notification directly to audience
		virtual void update (ccoms::caller_info info, ccoms::update_message msg = ccoms::update_message())
		{
			this->notify(msg);
		}
		
		virtual igraph<T>* get_jacobian (void)
		{
			if (iconnector<T>* c = dynamic_cast<iconnector<T>*>(get()))
			{
				return c->get_jacobian();
			}
			return nullptr;
		}
};

}

#include "../../../src/graph/buffer/buffer.ipp"

#endif /* buffer_hpp */
