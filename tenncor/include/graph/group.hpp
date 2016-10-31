//
//  group.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-30.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef group_hpp
#define group_hpp

#include "evoker.hpp"

namespace nnet {

template <typename T>
class group : public ievoker<T> {
	private:
		std::vector<EVOKER_PTR<T> > acts_;

	protected:
		virtual EVOKER_PTR<T> clone_impl (std::string name) {
			return std::shared_ptr<group<T> >(new group(acts_));
		}

	public:
		group (void) {}
		group (std::vector<EVOKER_PTR<T> > acts) : acts_(acts) {}
		virtual ~group (void) {}

		std::shared_ptr<group<T> > clone (std::string name = "") {
			return std::static_pointer_cast<group<T>, ievoker<T> >(clone_impl(name));
		}

		void add (EVOKER_PTR<T> evok) {
			acts_.push_back(evok);
		}

		virtual const tensor<T>& eval (void) {
			const tensor<T>* ptr;
			for (EVOKER_PTR<T> evok : acts_) {
				ptr = &(evok->eval());
			}
			return *ptr;
		}
};

}

#endif /* group_hpp */
