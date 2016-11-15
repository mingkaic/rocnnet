//
//  ileaf.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef ileaf_hpp

namespace nnet
{

// INITIALIZER MANAGING INTERFACE

template <typename T>
struct ileaf<T>::dyn_init : public initializer<T>
{
	private:
		tensor<T>* hold = nullptr;

	public:
		dyn_init (tensor<T>& in) : hold(&in) {}

		virtual void operator () (tensor<T>& in)
		{
			hold = &in;
		}
		virtual initializer<T>* clone (void)
		{
			return new dyn_init(*hold);
		}

		virtual ileaf<T>::dyn_init& operator = (const std::vector<T>& in)
		{
			this->delegate_task(*hold, [&in](T* raw, size_t len)
			{
				std::copy(in.begin(), in.end(), raw);
			});
			return *this;
		}
};

template <typename T>
void ileaf<T>::copy (const ileaf<T>& other, std::string name)
{
	if (nullptr != init_)
	{
		delete init_;
	}
	init_ = other.init_->clone();
	is_init_ = other.is_init_;
	ivariable<T>::copy(other, name);
}

template <typename T>
ileaf<T>::ileaf (const ileaf<T>& other, std::string name) :
	ivariable<T>(other, name),
	init_(other.init_->clone()),
	is_init_(other.is_init_) {}

template <typename T>
ileaf<T>::ileaf (const tensorshape& shape, initializer<T>* init, std::string name) :
	ivariable<T>(shape, name), init_(init) {}

template <typename T>
ileaf<T>::~ileaf (void)
{
	if (nullptr != init_)
	{
		delete init_;
	}
}

template <typename T>
ileaf<T>* ileaf<T>::clone (std::string name)
{
	return static_cast<ileaf<T>*>(clone_impl(name));
}

template <typename T>
ileaf<T>& ileaf<T>::operator = (const ileaf<T>& other)
{
	if (this != &other)
	{
		this->copy(other);
	}
	return *this;
}

template <typename T>
bool ileaf<T>::can_init (void) const
{
	return init_ != nullptr;
}

template <typename T>
ivariable<T>* ileaf<T>::get_gradient (void)
{
	return this;
}

}

#endif