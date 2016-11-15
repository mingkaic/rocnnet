//
//  constant.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef constant_hpp

namespace nnet
{

// CONSTANT IMPLEMENTATION

template <typename T>
constant<T>::constant (const constant<T>& other, std::string name) :
	ileaf<T>(other, name) {}

template <typename T>
ivariable<T>* constant<T>::clone_impl (std::string name)
{
	return new constant(*this, name);
}

template <typename T>
constant<T>::constant (T scalar) :
	ileaf<T>(std::vector<size_t>{1},
	new const_init<T>(scalar), 
	nnutils::formatter() << scalar)
{
	this->out_->allocate(new ram_alloc());
	(*this->init_)(*(this->out_));
	this->is_init_ = true;
}

template <typename T>
constant<T>::constant (std::vector<T> raw, tensorshape shape) :
	ileaf<T>(shape,
	new typename ileaf<T>::dyn_init(*(this->out_)),
	nnutils::formatter() << raw.front() << ".." << raw.back() << raw.end())
{
	this->out_->allocate(new ram_alloc());
	(*this->init_) = raw;
	this->is_init_ = true;
}

template <typename T>
constant<T>* constant<T>::clone (std::string name)
{
	return static_cast<constant<T>*>(clone_impl(name));
}

template <typename T>
void constant<T>::detach (ccoms::iobserver* viewer) {
	ccoms::subject::detach(viewer);
	if (this->no_audience()) {
		// no audience, no point to live x_x
		delete this;
	}
}

}

#endif