//
//  const_immutable.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2017-09-17.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#include "graph/leaf/constant.hpp"

#ifdef TENNCOR_IMMUTABLE_HPP

namespace nnet
{

template <typename T>
const_immutable<T>* const_immutable<T>::get (inode<T>* x)
{
	return new const_immutable<T>(x);
}


template <typename T>
const_immutable<T>::const_immutable (inode<T>* x) :
	immutable<T>(std::vector<inode<T>*>{x},
	[](std::vector<tensorshape> shapes) { return shapes[0]; },
	new transfer_func<T>([](T* dest, std::vector<const T*> src, shape_io shape)
	{
		size_t n_elems = shape.outs_.n_elems();
		std::memcpy(dest, src[0], sizeof(T) * n_elems);
	}),
	[](std::vector<std::pair<inode<T>*,inode<T>*> >)
	{
		return constant<T>::get_shared_zero();
	}, "const_immutable")
{
	this->gcache_.clear();
	this->jacobians_.clear();
}

}

#endif
