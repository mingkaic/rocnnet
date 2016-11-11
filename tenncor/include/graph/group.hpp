//
//  group.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-30.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

// DEPRECATED TODO: replace with accumulator
#pragma once
#ifndef group_hpp
#define group_hpp

#include "evoker.hpp"

namespace nnet {

// sequentially evaluates stored evokers
template <typename T>
class group : public ievoker<T> {
	private:
		std::vector<ievoker<T>*> acts_;

	protected:
		virtual ievoker<T>* clone_impl (std::string name) {
			return new group(acts_);
		}

	public:
		group (void) {}
		group (std::vector<ievoker<T>*> acts) : acts_(acts) {}
		virtual ~group (void) {}

		group<T>* clone (std::string name = "") {
			return static_cast<group<T>*>(clone_impl(name));
		}

		void add (ievoker<T>*evok) {
			acts_.push_back(evok);
		}

		virtual const tensor<T>& eval (void) {
			const tensor<T>* ptr;
			for (ievoker<T>*evok : acts_) {
				ptr = &(evok->eval());
			}
			return *ptr;
		}
};

// asynchronously evaluate stored evokers
template <typename T>
class async_group : public ievoker<T> {
	private:
		std::unordered_set<ievoker<T>*> acts_;

	protected:
		virtual ievoker<T>* clone_impl (std::string name) {
			return new async_group(acts_);
		}

	public:
		async_group (void) {}
		async_group (std::unordered_set<ievoker<T>*> acts) : acts_(acts) {}
		virtual ~async_group (void) {}

        async_group<T>* clone (std::string name = "") {
			return static_cast<async_group<T>*>(clone_impl(name));
		}

		void add (ievoker<T>*evok) {
			acts_.emplace(evok);
		}

		virtual const tensor<T>& eval (void);
};

}

#endif /* group_hpp */
