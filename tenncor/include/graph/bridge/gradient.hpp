//
//  gradient.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-12.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef gradient_hpp
#define gradient_hpp

#include "iexecutor.hpp"
#include "graph/variable/constant.hpp"

namespace nnet {

template <typename T>
class gradient : public iexecutor<T> {
	private:
		ivariable<T>* root_;
		std::unordered_map<ivariable<T>*, ivariable<T>*> leaf_map;
		constant<T> one;

		gradient (const gradient<T>& other) : one(1) {}

	protected:
		virtual iexecutor<T>* clone_impl (void) {
			return new gradient(*this);
		}

	public:
		gradient (ivariable<T>* root, ivariable<T>* leaf = nullptr) : root_(root), one(1) {
			if (leaf) {
				this->add(leaf);
			}
		}

		gradient<T>* clone (void) { return static_cast<gradient<T> >(clone_impl()); }

		virtual void execute (void) {
			std::vector<ivariable<T>*> leaves = this->srcs_;
			if (ioperation<T>* op_ = dynamic_cast<ioperation<T>*>(root_)) {
				if (leaves.empty()) {
					// grab leaves from root if no leaves were specified
					op_->leaves_collect([&leaves](ccoms::subject* leaf) {
						if (ivariable<T>* var = dynamic_cast<ivariable<T>*>(leaf)) {
							leaves.push_back(var);
						}
					});
				}
				for (ivariable<T>* leaf : leaves) {
					// notify parents that leaf is notifying (since parents can observe multiple subjects)
					leaf->notify(leaf);
					std::stack<ivariable<T>*> jacobs;
					op_->channel(jacobs);
					ivariable<T>* top = op_;
					while (false == jacobs.empty()) {
						ivariable<T>* jac = jacobs.top();
						jacobs.pop();

						if (tensor_jacobi<T>* ten_jac =
							dynamic_cast<tensor_jacobi<T>*>(jac->get_eval())) {
							ten_jac->set_root(top);
							top = jac;
						}
					}
					leaf_map[leaf] = top;
				}
			} else {
				// root is a non-operation
				if (leaves.empty()) {
					leaf_map[root_] = &one;
				} else {
					for (ivariable<T>* leaf : leaves) {
						leaf_map[leaf] = leaf == root_ ? &one : nullptr;
					}
				}
			}
		}

		void extract (std::function<void(ivariable<T>*,ivariable<T>*)> collector) {
			for (auto leef : leaf_map) {
				collector(leef.first, leef.second);
			}
		}
};

}

#include "../../../src/graph/bridge/gradient.ipp"

#endif /* gradient_hpp */
