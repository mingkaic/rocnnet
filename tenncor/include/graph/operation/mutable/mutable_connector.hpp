//
//  mutable_connector.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-27.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/operation/iconnector.hpp"
#include <functional>

#pragma once
#ifndef mutable_connect_hpp
#define mutable_connect_hpp

namespace nnet
{

template <typename T>
using MAKE_CONNECT = std::function<inode<T>*(std::vector<varptr<T> >&)>;

// designed to cover all the edge cases of mutable connectors
// created permanent connector ic_ can potentially destroy mutable connector
// if arguments are destroyed (triggering a chain reaction)

template <typename T>
class mutable_connector : public iconnector<T>
{
	private:
		// we don't listen to inode when it's incomplete
		MAKE_CONNECT<T> op_maker_;
		std::vector<varptr<T> > arg_buffers_;
		// ic_ is a potential dependency
		iconnector<T>* ic_ = nullptr;

		void connect (void);
		void disconnect (void);

	protected:
		mutable_connector (MAKE_CONNECT<T> maker, size_t nargs);
		// ic_ uniqueness forces explicit copy constructor
		mutable_connector (const mutable_connector<T>& other);

	public:
		static mutable_connector<T>* build (MAKE_CONNECT<T> maker, size_t nargs);
		virtual ~mutable_connector (void)
		{
			if (nullptr != ic_)
			{
				delete ic_;
			}
		}

		// COPY
		virtual mutable_connector<T>* clone (void);
		mutable_connector<T>& operator = (const mutable_connector<T>& other);

		// inode METHODS
		virtual tensorshape get_shape(void);
		virtual tensor<T>* get_eval(void);
		virtual bindable_toggle<T>* get_gradient(void);
		virtual functor<T>* get_jacobian (void);

		// ICONNECTOR METHODS
		virtual void update (caller_info info, update_message msg = update_message());

		// MUTABLE METHODS
		// return true if replacing
		// replacing will destroy then remake ic_
		bool add_arg (inode<T>* var, size_t idx);
		// return true if removing existing var at index idx
		bool remove_arg (size_t idx);
		bool valid_args (void);

		// ACCESSORS
		// get arguments
		size_t nargs (void) const;
		virtual void get_args (std::vector<inode<T>*>& args) const;
};

}

#include "../../../../src/graph/mutable/mutable_connector.ipp"

#endif /* mutable_connect_hpp */
