//
//  update.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-23.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/evoker.hpp"

#ifndef update_hpp
#define update_hpp

namespace nnet {

// ELEMENT WISE ASSIGNMENT ONLY (shape dependent assignment may change output shape, which is undesirable)

template <typename T>
class update : public ievoker<T> {
	protected:
		std::shared_ptr<variable<T> > dest;
		VAR_PTR<T> src;
		// determines how element-wise assignment works, defaults to direct assignment
		std::function<void(T&,T)> assign = [](T& left, T right) { left = right; };

		update (update<T>& other);
		virtual EVOKER_PTR<T> clone_impl (std::string name);

	public:
		update (std::shared_ptr<variable<T> > dest,
				VAR_PTR<T> src);
		update (std::shared_ptr<variable<T> > dest,
				VAR_PTR<T> src,
				std::function<void(T&,T)> assign);
		virtual ~update (void) {}

		std::shared_ptr<update<T> > clone (std::string name = "") {
			return std::static_pointer_cast<update<T>, ievoker<T> >(clone_impl(name));
		}

		virtual const tensor<T>& eval (void);
};

template <typename T>
class update_sub : public update<T> {
	public:
		update_sub (std::shared_ptr<variable<T> > dest, VAR_PTR<T> src);
		virtual ~update_sub (void) {}

		std::shared_ptr<update<T> > clone (std::string name = "") {
			return std::static_pointer_cast<update_sub<T>, ievoker<T> >(this->clone_impl(name));
		}
};

}

#include "../../src/optimization/update.ipp"

#endif /* update_hpp */
