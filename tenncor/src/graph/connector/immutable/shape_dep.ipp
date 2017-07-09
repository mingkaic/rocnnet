//
// Created by Mingkai Chen on 2017-07-03.
//

#ifdef TENNCOR_SHAPE_DEP_HPP

namespace nnet
{

template <typename T>
shape_dep<T>::~shape_dep (void)
{
	delete shaper_;
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
		shaper_ = other.shaper_->clone();
	}
	return *this;
}

template <typename T>
shape_dep<T>& shape_dep<T>::operator = (shape_dep<T>&& other)
{
	if (this != &other)
	{
		base_immutable<T>::operator = (std::move(other));
		shaper_ = other.shaper_->move();
		other.shaper_ = nullptr;
	}
	return *this;
}

template <typename T>
typename iconnector<T>::summary_series shape_dep<T>::summarize (void) const
{
	return {};
}

template <typename T>
shape_dep<T>::shape_dep (std::vector<inode<T>*> args,
	SHAPE_EXTRACT forward, tensorshape shape, std::string label) :
base_immutable<T>(args, label),
shaper_(new shape_extracter<T>(forward))
{
	this->mergible_ = false;
	this->data_ = new tensor<T>(shape);
	this->jacobians_.clear();
	this->update({});
}

template <typename T>
shape_dep<T>::shape_dep (const shape_dep<T>& other) :
	base_immutable<T>(other)
{
	shaper_ = other.shaper_->clone();
}

template <typename T>
shape_dep<T>::shape_dep (shape_dep<T>&& other) :
	base_immutable<T>(std::move(other))
{
	shaper_ = other.shaper_->move();
	other.shaper_ = nullptr;
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
void shape_dep<T>::forward_pass (void)
{
	std::vector<tensorshape> shapes;
	for (subject* sub : this->dependencies_)
	{
		shapes.push_back(this->take_eval(static_cast<inode<T>*>(sub))->get_shape());
	}
	(*shaper_)(this->data_, shapes);
}

template <typename T>
void shape_dep<T>::backward_pass (variable<T>*) {}

}

#endif