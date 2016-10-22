//
//  functions.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-09-30.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#pragma once
#ifndef functions_ops
#define functions_ops

#include "unar_ops.hpp"
#include "bin_ops.hpp"

namespace nnet {

template <typename T>
using TEN_FUNC = std::function<void(VAR_PTR<T>&, VAR_PTR<T>)>;

template <typename T>
void sigmoid (VAR_PTR<T>& out, VAR_PTR<T> in) {
	// f(x) = 1/(1+e^-x)
	VAR_PTR<T> negres = std::make_shared<neg<T> >(in);
	VAR_PTR<T> expres = std::make_shared<exp<T> >(negres);
	VAR_PTR<T> denom = std::make_shared<add<T> >(1, expres);
	out = std::make_shared<div<T> >(1, denom);
}

template <typename T>
void tanh (VAR_PTR<T>& out, VAR_PTR<T> in) {
	// f(x) = (e^(2*x)+1)/(e^(2*x)-1)
	VAR_PTR<T> pres = std::make_shared<add<T> >(in); // 2*x
	VAR_PTR<T> expres = std::make_shared<exp<T> >(pres);
	VAR_PTR<T> numer = std::make_shared<sub<T> >(expres, 1);
	VAR_PTR<T> denom = std::make_shared<add<T> >(expres, 1);
	out = std::make_shared<div<T> >(numer, denom);
}

template <typename T>
// clones every node from root to anything in leaf_set_src
void clone (VAR_PTR<T>& dest_root,
			VAR_PTR<T> src_root,
			std::unordered_set<VAR_PTR<T> > leaf_set_src,
			std::unordered_set<VAR_PTR<T> >& leaf_set_dest) {
	std::queue<WEAK_VAR_PTR<T> > q;
	std::unordered_map<WEAK_VAR_PTR<T>, WEAK_VAR_PTR<T> > src_to_dest;
	dest_root = src_root->clone();
	WEAK_VAR_PTR<T> cur = src_root;
	WEAK_VAR_PTR<T> cpy = dest_root;
	// storage
	q.push(src_root);
	src_to_dest[src_root] = dest_root;

	while (false == q.empty()) {
		cur = q.front();
		q.pop();
		cpy = src_to_dest[cur];
		if (leaf_set_src.end() == leaf_set_src.find(cpy)) {
			if (std::weak_ptr<iunar_ops<T> > op = std::dynamic_pointer_cast<iunar_ops<T> >(cur)) {
				std::weak_ptr<iunar_ops<T> > cur_cpy = std::dynamic_pointer_cast<iunar_ops<T> >(cpy);
				std::weak_ptr<ivariable<T> > next_cpy = nullptr;
				if (src_to_dest.end() == src_to_dest.find(op->var)) {
					next_cpy = op->var->clone();
					q.push(op->var);
					src_to_dest[op->var] = next_cpy;
				} else {
					next_cpy = src_to_dest[op->var];
				}
				(*cur_cpy)(next_cpy); // attach copies
			} else if (std::weak_ptr<ibin_ops<T> > op = std::dynamic_pointer_cast<ibin_ops<T> >(cur)) {
				std::weak_ptr<ibin_ops<T> > cur_cpy = std::dynamic_pointer_cast<ibin_ops<T> >(cpy);
				std::weak_ptr<ivariable<T> > a_cpy = nullptr;
				std::weak_ptr<ivariable<T> > b_cpy = nullptr;
				if (src_to_dest.end() == src_to_dest.find(op->a)) {
					a_cpy = op->a->clone();
					q.push(op->a);
					src_to_dest[op->a] = a_cpy;
				} else {
					a_cpy = src_to_dest[op->a];
				}
				if (src_to_dest.end() == src_to_dest.find(op->b)) {
					b_cpy = op->b->clone();
					q.push(op->b);
					src_to_dest[op->b] = b_cpy;
				} else {
					b_cpy = src_to_dest[op->b];
				}
				(*cur_cpy)(a_cpy, b_cpy);
			}
		} else {
			leaf_set_dest.emplace(cpy);
		}
	}
}

}

#endif /* functions_ops */
