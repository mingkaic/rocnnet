//
//  ileaf.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-08-29.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef ileaf_hpp

namespace nnet {

// INITIALIZER MANAGING INTERFACE

template <typename T>
struct ileaf<T>::open_init : public initializer<T> {
	private:
		tensor<T>* hold = nullptr;

	public:
		open_init (tensor<T>& in) : hold(&in) {}

		virtual void operator () (tensor<T>& in) {
			hold = &in;
		}
		virtual initializer<T>* clone (void) {
			return new open_init(*hold);
		}

		virtual ileaf<T>::open_init& operator = (const std::vector<T>& in) {
			this->delegate_task(*hold, [&in](T* raw_data, size_t size) {
				std::copy(in.begin(), in.end(), raw_data);
			});
			return *this;
		}
};

template <typename T>
ileaf<T>& ileaf<T>::operator = (const VAR_PTR<T>& other) {
	if (this != other.get()) {
		if (nullptr != this->init_) {
			delete this->init_;
		}

		if (const std::shared_ptr<ileaf<T> > vptr = std::dynamic_pointer_cast<ivar_init<T> >(other)) {
			this->copy(*vptr);
		} else {
			ivariable<T>::copy(*other);
		}
	}
	return *this;
}

}

#endif