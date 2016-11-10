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
	private:
		update (update<T>& other);

	protected:
		variable<T>* dest_;
		ivariable<T>* src_;

		// determines how element-wise assignment works, defaults to direct assignment
		std::function<void(T&,T)> assign_ = [](T& left, T right) { left = right; };

		void copy (update<T>& other);

		virtual ievoker<T>* clone_impl (std::string name);

	public:
		update (variable<T>* dest,
				ivariable<T>* src);
		update (variable<T>* dest,
				ivariable<T>* src,
				std::function<void(T&,T)> assign);
		virtual ~update (void) {
			delete dest_;
			delete src_;
		}

        update<T>* clone (std::string name = "") {
			return static_cast<update<T>*>(this->clone_impl(name));
		}

		virtual const tensor<T>& eval (void);
};

template <typename T>
class update_sub : public update<T> {
	private:
		update_sub (update_sub<T>& other);

    protected:
        virtual ievoker<T>* clone_impl (std::string name);

	public:
		update_sub (variable<T>* dest, ivariable<T>* src);

        update_sub<T>* clone (std::string name = "") {
			return static_cast<update_sub<T>*>(this->clone_impl(name));
		}
};

}

#include "../../src/optimization/update.ipp"

#endif /* update_hpp */
