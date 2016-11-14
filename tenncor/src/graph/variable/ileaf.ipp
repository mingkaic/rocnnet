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
struct ileaf<T>::dyn_init : public initializer<T> {
	private:
		tensor<T>* hold = nullptr;

	public:
		dyn_init (tensor<T>& in) : hold(&in) {}

		virtual void operator () (tensor<T>& in) {
			hold = &in;
		}
		virtual initializer<T>* clone (void) {
			return new dyn_init(*hold);
		}

		virtual ileaf<T>::dyn_init& operator = (const std::vector<T>& in) {
			this->delegate_task(*hold, [&in](T* raw, size_t len) {
				std::copy(in.begin(), in.end(), raw);
			});
			return *this;
		}
};

template <typename T>
ileaf<T>& ileaf<T>::operator = (const ileaf<T>& other) {
	if (this != &other) {
		this->copy(other);
	}
	return *this;
}

}

#endif