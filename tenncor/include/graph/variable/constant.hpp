//
//  constant.hpp
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
#ifndef constant_hpp
#define constant_hpp

#include "ileaf.hpp"

namespace nnet {

// Never notifies... should consider inheriting from different parent
template <typename T>
class constant : public ileaf<T> {
	private:
		constant (const constant<T>& other, std::string name);

	protected:
		virtual ievoker<T>* clone_impl (std::string name);

	public:
		constant (T scalar);
		constant (std::vector<T> raw, tensor_shape shape);

		// COPY
        constant<T>* clone (std::string name = "") {
			return static_cast<constant<T>*>(clone_impl(name));
		}

		// CONSTANT IS NOT A FIRST WORLD CITIZEN :(
		virtual void detach (ccoms::iobserver* viewer) {
			ccoms::subject::detach(viewer);
			if (this->no_audience()) {
				// no audience, no point to live x_x
				delete this;
			}
		}
};

}

#include "../../../src/graph/variable/constant.ipp"

#endif /* constant_hpp */