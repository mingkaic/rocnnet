//
// Created by Mingkai Chen on 2017-07-03.
//

#ifdef TENNCOR_SHAPE_DEP_HPP

namespace nnet
{

template <typename T>
shape_dep<T>::~shape_dep (void)
{
	delete shape_info;
}

template <typename T>
shape_dep<T>* shape_dep<T>::get (std::vector<inode<T>*> args,
	SHAPE_EXTRACT forward, tensorshape shape, std::string name)
{
	size_t n_args = args.size();
	assert(n_args > 0);
	std::unordered_set<inode<T>*> audience;
	if (args[0]->find_audience(name, audience))
	{
		// share nodes when possible
		for (inode<T>* aud : audience)
		{
			if (shape_dep<T>* saud = dynamic_cast<shape_dep<T>*>(aud))
			{
				std::vector<inode<T>*> aud_args = aud->get_arguments();
				if (n_args == aud_args.size())
				{
					bool all_aud = true;
					for (size_t i = 0; i < n_args; i++)
					{
						all_aud = all_aud && args[i] == aud_args[i];
					}
					if (all_aud) return saud;
				}
			}
		}
	}
	return new shape_dep<T>(args, forward, shape, name);
}

template <typename T>
shape_dep<T>* shape_dep<T>::clone (void) const
{
	return static_cast<shape_dep<T>*>(this->clone_impl());
}

template <typename T>
shape_dep<T>* shape_dep<T>::move (void)
{
	return static_cast<shape_dep<T>*>(this->move_impl());
}

template <typename T>
shape_dep<T>& shape_dep<T>::operator = (const shape_dep<T>& other)
{
	if (this != &other)
	{
		base_immutable<T>::operator = (other);
		shape_ = other.shape_;
		shape_info = other.shape_info->clone();
	}
	return *this;
}

template <typename T>
shape_dep<T>& shape_dep<T>::operator = (shape_dep<T>&& other)
{
	if (this != &other)
	{
		base_immutable<T>::operator = (std::move(other));
		shape_ = std::move(other.shape_);
		shape_info = other.shape_info->move();
		other.shape_info = nullptr;
	}
	return *this;
}

template <typename T>
shape_dep<T>::shape_dep (std::vector<inode<T>*> args,
	SHAPE_EXTRACT forward, tensorshape shape, std::string label) :
base_immutable<T>(args, label),
shape_info(new shape_extracter<T>(forward)),
shape_(shape)
{
	shape_.assert_is_fully_defined();
	this->jacobians_.clear();
	this->update(std::unordered_set<size_t>{});
}

template <typename T>
shape_dep<T>::shape_dep (const shape_dep<T>& other) :
	base_immutable<T>(other)
{
	shape_ = other.shape_;
	shape_info = other.shape_info->clone();
}

template <typename T>
shape_dep<T>::shape_dep (shape_dep<T>&& other) :
	base_immutable<T>(std::move(other))
{
	shape_ = std::move(other.shape_);
	shape_info = other.shape_info->move();
	other.shape_info = nullptr;
}

template <typename T>
inode<T>* shape_dep<T>::clone_impl (void) const
{
	return new shape_dep(*this);
}

template <typename T>
inode<T>* shape_dep<T>::move_impl (void)
{
	return new shape_dep(std::move(*this));
}

template <typename T>
base_immutable<T>* shape_dep<T>::arg_clone (std::vector<inode<T>*> args) const
{
	return new shape_dep(args, shape_info->get_shaper(), shape_, this->get_label());
}

template <typename T>
void shape_dep<T>::forward_pass (void)
{
	std::vector<tensorshape> shapes;
	for (subject* sub : this->dependencies_)
	{
		shapes.push_back(this->take_eval(static_cast<inode<T>*>(sub))->get_shape());
	}
	if (nullptr == this->data_)
	{
		this->data_ = new tensor<T>(shape_);
	}
	(*shape_info)(*this->data_, shapes);
}

template <typename T>
void shape_dep<T>::backward_pass (variable<T>* leaf)
{
	this->gcache_[leaf] = nnet::constant<T>::get_shared_zero();
}

}

#endif