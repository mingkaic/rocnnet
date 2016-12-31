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
			if (functor<T>* j = gconnect->get_jacobian())
			{
				// by default grad_jacobi most likely has its root hidden
				// do away with the graph behavior by exposing its root
				g_root_ = j;
				j->init(); // force initialization to make things easier
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
	bindable_toggle<T>* togg_last = nullptr;
	for (ivariable<T>* var : leaves)
	{
		// expect gradients to be the same shape as leaves
		leaf_map_[var] = new placeholder<T>(std::vector<size_t>{}, "grad_in:" + var->get_name());

		// bind all the bindable_toggle
		if (bindable_toggle<T>* togg = dynamic_cast<bindable_toggle<T>*>(var->get_gradient()))
		{
			if (nullptr != togg_last)
			{
				togg->bind(gid_, togg_last);
			}
			else
			{
				togg_last = togg;
			}
		}
	}
}

template <typename T>
void gradient<T>::execute (void)
{
	for (auto leaf_pair : leaf_map_)
	{
		ivariable<T>* gleaf = leaf_pair.first->get_gradient();
		if (bindable_toggle<T>* togg = dynamic_cast<bindable_toggle<T>*>(gleaf))
		{
			togg->activate(gid_);
		}
		tensor<T>* root_res = g_root_->get_eval();
		*(leaf_pair.second) = *root_res;
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