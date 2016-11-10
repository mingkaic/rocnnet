//
//  placeholder.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <list>
#include <ctime>
#include <random>
#include <new>
#include <memory>

#pragma once
#ifndef placeholder_hpp
#define placeholder_hpp

#include "ileaf.hpp"

namespace nnet {

template <typename T>
class placeholder : public ileaf<T> {
	private:
		placeholder (const placeholder<T>& other, std::string name);

	protected:
		virtual EVOKER_PTR<T> clone_impl (std::string name);

	public:
		placeholder (const tensor_shape& shape, std::string name = "");
		placeholder (const tensor_shape& shape, 
			initializer<T>* init, 
			std::string name = "");
			
		// COPY
		std::shared_ptr<placeholder<T> > clone (std::string name = "") {
			return std::static_pointer_cast<placeholder<T>, ievoker<T> >(clone_impl(name));
		}

		// DATA ASSIGNMENT
		// remember to notify after change
		virtual placeholder<T>& operator = (VAR_PTR<T> other);
		// assign raw data according to 1 dimension representation of inner tensor
		virtual placeholder<T>& operator = (std::vector<T> data);
		virtual placeholder<T>& operator = (const tensor<T>& data);

		// MOVES
		// todo: implement move clone
		virtual placeholder<T>& operator = (placeholder<T>&& other) = default;

		// FACTORIES
		static std::shared_ptr<placeholder<T> > make (std::string name = "") {
			VAR_PTR<T> inst = ivariable<T>::make_shared(new placeholder(name));
			return std::static_pointer_cast<placeholder<T> >(inst);
		}
		static std::shared_ptr<placeholder<T> > make (const tensor_shape& shape, std::string name = "") {
			VAR_PTR<T> inst = ivariable<T>::make_shared(new placeholder(shape, name));
			return std::static_pointer_cast<placeholder<T> >(inst);
		}
};

template <typename T>
using PLACEHOLDER_PTR = std::shared_ptr<placeholder<T> >;

template <typename T>
constexpr auto PLACEHOLDER_TO_VAR = std::static_pointer_cast<ivariable<T>, placeholder<T> >;

}

#include "../../../src/graph/variable/placeholder.ipp"

#endif /* placeholder_hpp */
