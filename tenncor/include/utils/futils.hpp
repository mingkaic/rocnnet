/*!
 *
 *  futils.hpp
 *  cnnet
 *
 *  Purpose:
 *  define commonly used activation functions
 *  and useful graph operations
 *
 *  Created by Mingkai Chen on 2016-09-30.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include "graph/operation/immutable/elementary.hpp"
#include "graph/varptr.hpp"

#ifndef TENNCOR_FUTILS_HPP
#define TENNCOR_FUTILS_HPP

namespace nnet
{

template <typename T>
varptr<T> sigmoid (varptr<T> x)
{
	// f(x) = 1/(1+e^-x)
	return (T)1 / ((T)1 + exp(-x));
}

template <typename T>
varptr<T> tanh (varptr<T> x)
{
	// f(x) = (e^(2*x)+1)/(e^(2*x)-1)
	varptr<T> etx = exp((T)2 * x);
	return (etx + (T)1) / (etx - (T)1);
}

// TODO: fix graph copy
//template <typename T>
// clones every node from root to anything in leaf_set_src
//inode<T>* clone (inode<T>* src_root,
//			std::unordered_set<inode<T>* > leaf_set_src,
//			std::unordered_set<inode<T>* >& leaf_set_dest) {
//	std::queue<inode<T>* > q;
//	std::unordered_map<inode<T>*, inode<T>* > src_to_dest;
//	inode<T>* dest_root = src_root->clone();
//	inode<T>* cur = src_root;
//	inode<T>* cpy = dest_root;
//	// storage
//	q.push(src_root);
//	src_to_dest[src_root] = dest_root;
//
//	while (false == q.empty()) {
//		cur = q.front();
//		q.pop();
//		cpy = src_to_dest[cur];
//		if (leaf_set_src.end() == leaf_set_src.find(cpy)) {
//			if (std::weak_ptr<iunar_ops<T> > op = std::dynamic_pointer_cast<iunar_ops<T> >(cur)) {
//				std::weak_ptr<iunar_ops<T> > cur_cpy = std::dynamic_pointer_cast<iunar_ops<T> >(cpy);
//				std::weak_ptr<inode<T> > next_cpy = nullptr;
//				if (src_to_dest.end() == src_to_dest.find(op->var)) {
//					next_cpy = op->var->clone();
//					q.push(op->var);
//					src_to_dest[op->var] = next_cpy;
//				} else {
//					next_cpy = src_to_dest[op->var];
//				}
//				(*cur_cpy)(next_cpy); // attach copies
//			} else if (std::weak_ptr<ibin_ops<T> > op = std::dynamic_pointer_cast<ibin_ops<T> >(cur)) {
//				std::weak_ptr<ibin_ops<T> > cur_cpy = std::dynamic_pointer_cast<ibin_ops<T> >(cpy);
//				std::weak_ptr<inode<T> > a_cpy = nullptr;
//				std::weak_ptr<inode<T> > b_cpy = nullptr;
//				if (src_to_dest.end() == src_to_dest.find(op->a)) {
//					a_cpy = op->a->clone();
//					q.push(op->a);
//					src_to_dest[op->a] = a_cpy;
//				} else {
//					a_cpy = src_to_dest[op->a];
//				}
//				if (src_to_dest.end() == src_to_dest.find(op->b)) {
//					b_cpy = op->b->clone();
//					q.push(op->b);
//					src_to_dest[op->b] = b_cpy;
//				} else {
//					b_cpy = src_to_dest[op->b];
//				}
//				(*cur_cpy)(a_cpy, b_cpy);
//			}
//		} else {
//			leaf_set_dest.emplace(cpy);
//		}
//	}
//	return dest_root;
//}

}

#endif /* TENNCOR_FUTILS_HPP */
