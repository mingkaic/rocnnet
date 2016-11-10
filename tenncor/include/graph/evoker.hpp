//
//  evoker.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-23.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef evoker_ops
#define evoker_ops

#include "../memory/session.hpp"
#include "../tensor.hpp"

namespace nnet {

template <typename T>
class ievoker;

template <typename T>
class variable;

template <typename T>
class placeholder;

template <typename T>
class ievoker {
	protected:
		virtual T* get_raw (tensor<T>& t) const { return t.raw_data_; }
		virtual const T* get_raw (const tensor<T>& t) const { return t.raw_data_; }

		virtual ievoker<T>* clone_impl (std::string name) = 0;

	public:
		virtual ~ievoker (void) {}
		ievoker<T>* clone (std::string name = "") { return clone_impl(name); }

		// TODO consider changing return type to a pointer (shared)
		virtual const tensor<T>& eval (void) = 0;
};

}

#endif /* evoker_hpp */
