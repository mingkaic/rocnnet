//
//  gradient.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-12.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef gradient_hpp

namespace nnet
{

template <typename T>
void gradient<T>::clear_map (void)
{
	for (auto leaf_pair : leaf_map_)
	{
		delete leaf_pair.second; // we own the placeholders
	}
	leaf_map_.clear();
}

template <typename T>
void gradient<T>::copy (const gradient<T>& other)
{
	// shallow copy since we don't own content
	g_root_ = other.g_root_;
	leaf_map_ = other.leaf_map_;
}

template <typename T>
gradient<T>::gradient (const gradient<T>& other)
{
	copy(other);
}

template <typename T>
iexecutor<T>* gradient<T>::clone_impl (void)
{
	return new gradient(*this);
}

template <typename T>
gradient<T>::gradient (ivariable<T>* root, ivariable<T>* leaf) :
	g_root_(root->get_gradient())
{
	// collect leaves from root
	if (ioperation<T>* root_op = dynamic_cast<ioperation<T>*>(root))
	{
		// collect potential sources
		root_op->leaves_collect([this](ivariable<T>* src)
		{
			potential_srcs_.push_back(src);
		});

		// take jacobian instead if available
		if (iconnector<T>* gconnect = dynamic_cast<iconnector<T>*>(g_root_))
		{
			if (igraph<T>* j = gconnect->get_jacobian())
			{
				// by default grad_jacobi most likely has its root hidden
				// do away with the graph behavior by exposing its root
				g_root_ = j;
			}
		}
	}
	else // either leaf, graph, or buffer
	{
		potential_srcs_.push_back(root);
	}
	
	// deal with leaf if necessary
	if (leaf)
	{
		this->add(leaf);
	}
}

template <typename T>
gradient<T>::~gradient (void)
{
	clear_map();
}

template <typename T>
gradient<T>* gradient<T>::clone (void)
{
	return static_cast<gradient<T>*>(clone_impl());
}

template <typename T>
gradient<T>& gradient<T>::operator = (const gradient<T>& other)
{
	copy(other);
}

template <typename T>
void gradient<T>::freeze (void)
{
	clear_map();
	// select leaves (as dependencies or potential srcs)
	std::vector<ivariable<T>*> leaves = this->dependencies_;
	if (this->dependencies_.empty())
	{
		leaves = potential_srcs_;
	}
	// populate leaf_map_
	for (ivariable<T>* var : leaves)
	{
		// expect gradients to be the same shape as leaves
		leaf_map_[var] = new placeholder<T>(std::vector<size_t>{}, "grad_in:" + var->get_name());
	}
}

template <typename T>
void gradient<T>::execute (void)
{
	// notify leaves and extract gradient to leaf_map
	auto it = leaf_map_.begin();
	ivariable<T>* it_leaf = it->first;
	ivariable<T>* it_grad = it_leaf->get_gradient();
	for (auto leaf_pair : leaf_map_)
	{
		ivariable<T>* leaf_grad = leaf_pair.first->get_gradient();
		leaf_grad->notify(it_grad); // special notify to nullify all leaf nodes except *it
	}
	// assign g_root's tensor to leaf_map's placeholder
	tensor<T>* root_res = g_root_->get_eval();
	*(leaf_map_[it_leaf]) = *root_res;

	// now that every leaf except it_grad is nulled
	// we only need to notify the previous leaf and the current leaf
	// nullifying previous and un-nullifying current
	ivariable<T>* previous = it_leaf;
	for (it++; leaf_map_.end() != it; it++)
	{
		it_leaf = it->first;
		it_grad = it_leaf->get_gradient();
		previous->get_gradient()->notify(it_grad);
		it_grad->notify(it_grad);
		*(leaf_map_[it_leaf]) = *root_res;
		previous = it_leaf;
	}
}

template <typename T>
void gradient<T>::collect_grad (GRAD_GATHER<T> collector)
{
	for (auto leaf_pair : leaf_map_)
	{
		collector(leaf_pair.first, leaf_pair.second);
	}
}

}

#endif