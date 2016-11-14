//
//  elementary.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-24.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifndef elementary_hpp
#define elementary_hpp

#include "graph/operation/ioperation.hpp"

namespace nnet {

template <typename T>
class elementary : public ioperation<T> {
	private:
		std::function<void(T&, T)> for_each_;
		BUILD_DERIVE<T> der_;
		
	protected:
		virtual void setup_gradient (void);
		virtual tensorshape shape_eval (void);
		virtual ivariable<T>* clone_impl (std::string name);

		elementary (const elementary<T>& other, std::string name) :
			ioperation<T>(other, name),
			for_each_(other.for_each_),
			der_(other.der_) {}

	public:
		elementary (std::vector<ivariable<T>*> args,
					std::function<void(T&, T)> op,
					BUILD_DERIVE<T> der,
					std::string name = "");

		// COPY
		elementary<T>* clone (std::string name = "") {
			return static_cast<elementary<T>*>(clone_impl(name));
		}
		virtual elementary<T>& operator = (const elementary<T>& other) {
			if (this != &other) {
				for_each_ = other.for_each_;
				der_ = other.der_;
				this->copy(other);
			}
			return *this;
		}
		
		// MOVES
		// TODO: implement

		virtual void update (ccoms::subject* caller);
};

// operators that will replace elementary operation objects
template<typename T>
ivariable<T>* operator + (const ivariable<T>* a);

template<typename T>
ivariable<T>* operator - (const ivariable<T>* a);

template<typename T>
ivariable<T>* sin (const ivariable<T>* a);

template<typename T>
ivariable<T>* cos (const ivariable<T>* a);

template<typename T>
ivariable<T>* tan (const ivariable<T>* a);

template<typename T>
ivariable<T>* csc (const ivariable<T>* a);

template<typename T>
ivariable<T>* sec (const ivariable<T>* a);

template<typename T>
ivariable<T>* cot (const ivariable<T>* a);

template<typename T>
ivariable<T>* exp (const ivariable<T>* a);

template <typename T>
ivariable<T>* root (const ivariable<T>* a); // TODO implement

template <typename T>
ivariable<T>* pow (const ivariable<T>* a, T scalar); // TODO implement

template<typename T>
ivariable<T>* clip_val (const ivariable<T>* a, T min, T max);

template<typename T>
ivariable<T>* operator + (T a, const ivariable<T>* b);

template<typename T>
ivariable<T>* operator + (const ivariable<T>* a, T b);

template<typename T>
ivariable<T>* operator + (const ivariable<T>* a, const ivariable<T>* b);

template<typename T>
ivariable<T>* operator - (T a, const ivariable<T>* b);

template<typename T>
ivariable<T>* operator - (const ivariable<T>* a, T b);

template<typename T>
ivariable<T>* operator - (const ivariable<T>* a, const ivariable<T>* b);

template<typename T>
ivariable<T>* operator * (T a, const ivariable<T>* b);

template<typename T>
ivariable<T>* operator * (const ivariable<T>* a, T b);

template<typename T>
ivariable<T>* operator * (const ivariable<T>* a, const ivariable<T>* b);

template<typename T>
ivariable<T>* operator / (T a, const ivariable<T>* b);

template<typename T>
ivariable<T>* operator / (const ivariable<T>* a, T b);

template<typename T>
ivariable<T>* operator / (const ivariable<T>* a, const ivariable<T>* b);

}

#include "../../../../src/graph/operation/general/elementary.ipp"

#endif /* elementary_hpp */
