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
		std::vector<EVOKER_PTR<T> > _acts;

	protected:
		virtual EVOKER_PTR<T> clone_impl (std::string name) {
			return std::shared_ptr<group<T> >(new group(_acts));
		}

	public:
		group (void) {}
		group (std::vector<EVOKER_PTR<T> > acts) : _acts(acts) {}
		virtual ~group (void) {}

		std::shared_ptr<group<T> > clone (std::string name = "") {
			return std::static_pointer_cast<group<T>, ievoker<T> >(clone_impl(name));
		}

		void add (EVOKER_PTR<T> evok) {
			_acts.push_back(evok);
		}

		virtual const tensor<T>& eval (void) {
			const tensor<T>* ptr;
			for (EVOKER_PTR<T> evok : _acts) {
				ptr = &(evok->eval());
			}
			return *ptr;
		}
};

}

#endif /* group_hpp */
