//
// Created by Mingkai Chen on 2017-05-09.
//

#ifdef TENNCOR_MAPPABLE_HPP

namespace nnet
{

template <typename T>
mappable<T>* mappable<T>::get (inode<T>* arg, size_t idx)
{
	return new mappable<T>(arg, idx);
}

template <typename T>
mappable<T>* mappable<T>::clone (void) const
{
	return static_cast<mappable<T>*>(clone_impl());
}

template <typename T>
mappable<T>* mappable<T>::move (void)
{
	return static_cast<mappable<T>*>(move_impl());
}

template <typename T>
mappable<T>::mappable (inode<T>* arg, size_t idx) :
	immutable<T>((std::vector<inode<T>*>{arg}),
	[idx](std::vector<tensorshape> shapes)
	{
		tensorshape outshape = shapes[0];
		outshape.group_dim(idx);
		return outshape;
	},
	[](T* out, const tensorshape& outs, std::vector<const T*>& in, std::vector<tensorshape>&)
	{
		std::memcpy(out, in[0], sizeof(T) * outs.n_elems());
	},
	[](std::vector<inode<T>*> in, variable<T>* leaf)
	{
		return in[0]->get_leaf(leaf);
	}, "mappable") {}

template <typename T>
inode<T>* mappable<T>::clone_impl (void) const
{
	return new mappable(*this);
}

template <typename T>
inode<T>* mappable<T>::move_impl (void)
{
	return new mappable(std::move(*this));
}

}

#endif
