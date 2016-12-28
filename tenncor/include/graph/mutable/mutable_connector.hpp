//
//  mutable_connector.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-27.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/iconnector.hpp"
#include <functional>

#pragma once
#ifndef mutable_connect_hpp
#define mutable_connect_hpp

namespace nnet
{

template <typename T>
using MAKE_CONNECT = std::function<ivariable<T>*(std::vector<varptr<T> >&)>;

// designed to cover all the edge cases of mutable connectors
// created permanent connector ic_ can potentially destroy mutable connector
// if arguments are destroyed (triggering a chain reaction)

template <typename T>
class mutable_connector : public iconnector<T>
{
	private:
		// we don't listen to ivariable when it's incomplete
		MAKE_CONNECT<T> op_maker_;
		std::vector<varptr<T> > arg_buffers_;
		std::unique_ptr<iconnector<T> > ic_ = nullptr;

		void connect (void) {
			for (varptr<T> arg : arg_buffers_)
			{
				if (arg.get() == nullptr)
				{
					return;
				}
			}
			if (nullptr == ic_)
			{
				iconnector<T>* con = dynamic_cast<iconnector<T>*>(op_maker_(arg_buffers_));
				ic_ = std::unique_ptr<iconnector<T> >(con);
				this->add_dependency(con);
			}
		}

		void disconnect (void)
		{
			if (nullptr != ic_)
			{
				this->kill_dependencies();
				ic_ = nullptr;
			}
		}

	protected:
		mutable_connector (MAKE_CONNECT<T> maker, size_t nargs) :
			iconnector<T>(std::vector<ivariable<T>*>{}, ""),
			op_maker_(maker), arg_buffers_(nargs, nullptr) {}

		// ic_ uniqueness forces explicit copy constructor
		mutable_connector (const mutable_connector<T>& other) :
			iconnector<T>(other), op_maker_(other.op_maker_),
			arg_buffers_(other.arg_buffers_) {}

	public:
		static mutable_connector<T>* build (MAKE_CONNECT<T> maker, size_t nargs)
		{
			return new mutable_connector<T>(maker, nargs);
		}

		// COPY
		virtual mutable_connector<T>* clone (void)
		{
			return new mutable_connector<T>(*this);
		}
		mutable_connector<T>& operator = (const mutable_connector<T>& other)
		{
			if (&other != this)
			{
				iconnector<T>::operator = (other);
				op_maker_ = other.op_maker_;
				arg_buffers_ = other.arg_buffers_;
			}
			return *this;
		}

		// return true if replacing
		// replacing will destroy then remake ic_
		bool add_arg (ivariable<T>* var, size_t idx)
		{
			bool replace = nullptr != arg_buffers_[idx];
			arg_buffers_[idx] = var;
			if (replace)
			{
				disconnect();
			}
			connect();
			return replace;
		}

		// return true if removing existing var at index idx
		bool remove_arg (size_t idx)
		{
			if (nullptr != arg_buffers_[idx])
			{
				disconnect();
				arg_buffers_[idx] = nullptr;
				return true;
			}
			return false;
		}

		// FROM IVARIABLE
		virtual tensorshape get_shape(void)
		{
			if (nullptr == ic_)
			{
				return tensorshape();
			}
			return ic_->get_shape();
		}

		virtual tensor<T>* get_eval(void)
		{
			if (nullptr == ic_)
			{
				return nullptr;
			}
			return ic_->get_eval();
		}

		virtual bindable_toggle<T>* get_gradient(void)
		{
			if (nullptr == ic_)
			{
				return nullptr;
			}
			return ic_->get_gradient();
		}

		virtual functor<T>* get_jacobian (void)
		{
			if (nullptr == ic_)
			{
				return nullptr;
			}
			return ic_->get_jacobian();
		}

		// FROM ICONNECTOR
		virtual void update (ccoms::caller_info info, ccoms::update_message msg = ccoms::update_message())
		{
			this->notify(msg);
		}
};

}

#endif /* mutable_connect_hpp */
