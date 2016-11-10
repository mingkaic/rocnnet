//
//  ivariable.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef ivariable_hpp

namespace nnet {

// INITIALIZERS

template <typename T>
void const_init<T>::operator () (tensor<T>& in) {
	this->delegate_task(in, [this](T* raw_data, size_t size) {
		std::fill(raw_data, raw_data+size, value_);
	});
}

template <typename T>
void random_uniform<T>::operator () (tensor<T>& in) {
	this->delegate_task(in, [this](T* raw_data, size_t size) {
		for (size_t i = 0; i < size; i++) {
			raw_data[i] =  distribution_(session::get_generator());
		}
	});
}

// VARIABLE INTERFACE

template <typename T>
void ivariable<T>::copy (
	ivariable<T> const & other,
	std::string name) {
	 out_ = other.out_;
	if (0 == name.size()) {
		name = other.name+"_cpy";
	}
	this->name = name;
}

template <typename T>
ivariable<T>::ivariable (void) {
	session& sess = session::get_instance();
	sess.register_obj(*this);
}

template <typename T>
ivariable<T>::~ivariable (void){
	session& sess = session::get_instance();
	sess.unregister_obj(*this);
//	std::unordered_set<ioperation<T>*> copy = consumers;
//	for (ioperation<T>* cons : copy) {
//		cons->deconsume(*this);
//	}
}

template <typename T>
ivariable<T>& ivariable<T>::operator = (const ivariable<T>& other) {
	if (this != &other) {
		copy(other);
	}
	return *this;
}

// variable interface inner class

template <typename T>
class ivariable<T>::gradient_leaf : public ileaf<T> {
	private:
		gradient_leaf (WEAK_VAR_PTR<T> integral) {
			this->out_.set_shape(std::vector<size_t>{1});
			this->out_.allocate(std::make_shared<memory_alloc>());
			this->is_init = true;

			this->name = "leaf<" + integral.lock()->get_name() + ">";
			this->integral = integral;
			typename ileaf<T>::open_init* assigner;
			this->init_ = assigner = new typename ileaf<T>::open_init(this->out_);
			*assigner = std::vector<T>{0}; // initialize as zero
		}

	protected:
		virtual EVOKER_PTR<T> clone_impl (std::string name) {
			return gradient_leaf::make(this->integral);
		}

	public:
		static std::shared_ptr<gradient_leaf> make (WEAK_VAR_PTR<T> integral) {
			VAR_PTR<T> inst = ivariable<T>::make_shared(new gradient_leaf(integral));
			return std::static_pointer_cast<gradient_leaf>(inst);
		}
		virtual ~gradient_leaf (void) {}

		virtual void activate (VAR_PTR<T> active) {
			if (false == this->out_.is_alloc()) {
				this->out_.allocate(std::make_shared<memory_alloc>(), std::vector<size_t>{1});
			}
			// perform matrix calculus when necessary
			// when tensor variables can encasulate other variables
			// e.g.: A = [x y] where dA/dx is possible
			typename ileaf<T>::open_init* assigner =
				dynamic_cast<typename ileaf<T>::open_init*>(this->init_);
			if (active == this->integral.lock()) {
				*assigner = std::vector<T>{1};
			}
		}

		virtual void deactivate (void) {
			if (false == this->out_.is_alloc()) {
				this->out_.allocate(std::make_shared<memory_alloc>(), std::vector<size_t>{1});
			}
			// when tensor variables can encasulate other variables
			// e.g.: A = [x y] where dA/dx is possible
			typename ileaf<T>::open_init* assigner =
					dynamic_cast<typename ileaf<T>::open_init*>(this->init_);
			*assigner = std::vector<T>{0};
		}
};

}

#endif