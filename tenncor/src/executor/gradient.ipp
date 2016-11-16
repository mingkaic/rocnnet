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
	// set up gradient tree from root's lowest leaf
	if (ioperation<T>* op = dynamic_cast<ioperation<T>*>(root))
	{
		// collect potential sources
		op->leaves_collect([this](ccoms::subject* src)
		{
			potential_srcs_.push_back(src);
		});
		// collect all the jacobians (change/remove once channel-notify merge concludes)
		// >>
		std::stack<ivariable<T>*> jacobs;
		op->channel(jacobs); // expensive operation (traverses entire tree)
		// connect jacobians
		ivariable<T>* top = op; // start at the root
		while (false == jacobs.empty())
		{
			ivariable<T>* jac = jacobs.top(); // these jacs are hidden jacobian nodes
			jacobs.pop();
			// grab tensor_jacobi
			tensor_jacobi<T>* ten_jac = 
				static_cast<tensor_jacobi<T>*>(jac->get_eval());
			// connect lowest jacobians to higher jacobians
			ten_jac->set_root(top);
			top = jac;
		}
		g_root_ = top; // make resulting jacobian node the true gradient root
		// << collection concludes here
	}
	else
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
	return static_cast<gradient<T> >(clone_impl());
}

template <typename T>
gradient<T>& gradient<T>::operator = (const gradient<T>& other)
{
	copy(other);
}

template <typename T>
void gradient<T>::freeze (void)
{
	// populate leaf_map_
	std::vector<ccoms::subject*> leaves = this->dependencies_;
	if (this->dependencies_.empty())
	{
		leaves = potential_srcs_;
	}
	for (ccoms::subject* leaf : leaves)
	{
		if (ivariable<T>* var =
			dynamic_cast<ivariable<T>*>(leaf))
		{
			// expect gradients to be the same shape as leaves
			leaf_map_[var] = new placeholder<T>(var->get_shape());
		}
	}
}

template <typename T>
void gradient<T>::execute (void)
{
	// notify leaves and extract gradient to leaf_map
	auto it = leaf_map_.begin();
	ivariable<T>* it_grad = it->first->get_gradient();
	for (auto leaf_pair : leaf_map_)
	{
		ivariable<T>* leaf_grad = leaf_pair.first->get_gradient();
		leaf_grad->notify(it_grad); // special notify to nullify all leaf nodes except *it
	}
	// assign g_root's tensor to leaf_map's placeholder
	*(leaf_map_[it_grad]) = *(g_root_->get_eval());
	// now that every leaf except *it is nulled
	// we only need to notify the previous leaf and the current leaf
	// nullifying previous and un-nullifying current
	ivariable<T>* previous = it_grad;
	for (it++; leaf_map_.end() != it; it++)
	{
		it_grad = it->first->get_gradient();
		previous->notify(it_grad);
		it_grad->notify(it_grad);
		*(leaf_map_[it_grad]) = *(g_root_->get_eval());
		previous = it_grad;
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