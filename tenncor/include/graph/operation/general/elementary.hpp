//
//  elementary.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-24.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/operation/ioperation.hpp"
#include "graph/variable/constant.hpp"

#pragma once
#ifndef elementary_hpp
#define elementary_hpp

namespace nnet
{

template <typename T>
class elementary : public ioperation<T> {
	private:
		std::function<void(T&, T)> for_each_;
		BUILD_DERIVE<T> der_;
		
	protected:
		virtual void setup_gradient (void);
		virtual tensorshape shape_eval (void);

		elementary (const elementary<T>& other, std::string name);
			
		virtual ivariable<T>* clone_impl (std::string name);

	public:
		elementary (std::vector<ivariable<T>*> args,
			std::function<void(T&, T)> op,
			BUILD_DERIVE<T> der,
			std::string name = "");

		// COPY
		elementary<T>* clone (std::string name = "");
		virtual elementary<T>& operator = (const elementary<T>& other);
		
		// MOVES
		// TODO: implement

		virtual void update (ccoms::subject* caller);
};

// operators that will replace elementary operation objects
template<typename T>
varptr<T> operator + (varptr<T> a);

template<typename T>
varptr<T> operator - (varptr<T> a);

template<typename T>
varptr<T> sin (const ivariable<T>* a);

template<typename T>
varptr<T> cos (const ivariable<T>* a);

template<typename T>
varptr<T> tan (const ivariable<T>* a);

template<typename T>
varptr<T> csc (const ivariable<T>* a);

template<typename T>
varptr<T> sec (const ivariable<T>* a);

template<typename T>
varptr<T> cot (const ivariable<T>* a);

template<typename T>
varptr<T> exp (const ivariable<T>* a);

template <typename T>
varptr<T> root (const ivariable<T>* a); // TODO implement

template <typename T>
varptr<T> pow (const ivariable<T>* a, T scalar); // TODO implement

template<typename T>
varptr<T> clip_val (const ivariable<T>* a, T min, T max);

template<typename T>
varptr<T> operator + (T a, varptr<T> b);

template<typename T>
varptr<T> operator + (varptr<T> a, T b);

template<typename T>
varptr<T> operator + (varptr<T> a, varptr<T> b);

template<typename T>
varptr<T> operator - (T a, varptr<T> b);

template<typename T>
varptr<T> operator - (varptr<T> a, T b);

template<typename T>
varptr<T> operator - (varptr<T> a, varptr<T> b);

template<typename T>
varptr<T> operator * (T a, varptr<T> b);

template<typename T>
varptr<T> operator * (varptr<T> a, T b);

template<typename T>
varptr<T> operator * (varptr<T> a, varptr<T> b);

template<typename T>
varptr<T> operator / (T a, varptr<T> b);

template<typename T>
varptr<T> operator / (varptr<T> a, T b);

template<typename T>
varptr<T> operator / (varptr<T> a, varptr<T> b);

}

#include "../../../../src/graph/operation/general/elementary.ipp"

#endif /* elementary_hpp */
