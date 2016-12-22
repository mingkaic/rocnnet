//
//  iconnector.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-01.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef iconnector_hpp

namespace nnet
{

template <typename T>
void iconnector<T>::merge_leaves (std::unordered_set<ivariable<T>*>& src)
{
	src.insert(this->leaves_.begin(), this->leaves_.end());
}

template <typename T>
iconnector<T>::iconnector (std::vector<ivariable<T>*> dependencies, std::string name) :
	ccoms::iobserver(nnutils::to_vec<ivariable<T>*, ccoms::subject*>(dependencies, var_to_sub<T>)),
	ivariable<T>(name)
{
	for (ccoms::subject* dep : this->dependencies_)
	{
		ivariable<T>* leaf = sub_to_var<T>(dep);
		leaf->merge_leaves(leaves_);
	}
}

// template <typename T>
// iconnector<T>* iconnector<T>::clone (std::string name)
// {
// 	return static_cast<iconnector<T>*>(clone_impl(name));
// }

template <typename T>
void iconnector<T>::leaves_collect (std::function<void(ivariable<T>*)> collector)
{
	for (ivariable<T>* leaf : leaves_)
	{
		collector(leaf);
	}
}

}

#endif