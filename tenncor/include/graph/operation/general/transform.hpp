//
//  transform.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-09.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifndef transform_hpp
#define transform_hpp

#include "graph/operation/ioperation.hpp"

namespace nnet {

template <typename T>
T mean (const std::vector<T>& data) {
	T ans = 0;
	for (T raw : data) {
		ans += raw;
	}
	ans /= data.size();
	return ans;
}

// special tensor transform

template <typename T>
class transform : public ioperation<T> {
	private:
		std::function<void(T*&,const T*,tensorshape)> collect_;
		std::function<tensorshape(tensorshape)> shape_;
		BUILD_DERIVE<T> der_;

	protected:
		virtual void setup_gradient (void);
		virtual ivariable<T>* clone_impl (std::string name);
		virtual tensorshape shape_eval (void);

		transform (const transform<T>& other, std::string name) :
			ioperation<T>(other, name),
			collect_(other.collect_),
			shape_(other.shape_),
			der_(other.der_) {}

	public:
		transform (ivariable<T>* arg,
					std::function<void(T*&,const T*,tensorshape)> op,
					std::function<tensorshape(tensorshape)> trans,
					BUILD_DERIVE<T> der, std::string name = "");

		// COPY
		transform<T>* clone (std::string name = "") {
			return static_cast<transform<T>*>(clone_impl(name));
		}

		// MOVES
		// TODO: implement

		virtual void update (ccoms::subject* caller);
};

template <typename T>
ivariable<T>* clip_norm (const ivariable<T>* a, T cap);

template <typename T>
ivariable<T>* transpose (const ivariable<T>* a);

// fit to watch
template <typename T>
ivariable<T>* fit (const ivariable<T>* a, const ivariable<T>* watch);

template <typename T>
ivariable<T>* extend (const ivariable<T>* a, size_t index, size_t multiplier);

template <typename T>
ivariable<T>* compress (const ivariable<T>* a, size_t index,
	std::function<T(const std::vector<T>&)> collector = mean<T>);

}

#include "../../../../src/graph/operation/general/transform.ipp"

#endif /* transform_hpp */