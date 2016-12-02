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
		BUILD_DERIVE<T> der_;
		
	protected:
		virtual void setup_gradient (void);

		elementary (const elementary<T>& other, std::string name);
			
		virtual ivariable<T>* clone_impl (std::string name);

		// protect elementary constructor to ensure heap allocation
		elementary (std::vector<ivariable<T>*> args,
			TEN_OP<T> op,
			BUILD_DERIVE<T> der,
			std::string name);

	public:
		static ivariable<T>* build (std::vector<ivariable<T>*> args,
			TEN_OP<T> op,
			BUILD_DERIVE<T> der,
			std::string name = "")
		{
			return new elementary<T>(args, op, der, name);
		}
	
		// COPY
		elementary<T>* clone (std::string name = "");
		virtual elementary<T>& operator = (const elementary<T>& other);
		
		// MOVES
		// TODO: implement
};

// operators that will replace elementary operation objects
template<typename T>
varptr<T> operator + (const varptr<T> a);

template<typename T>
varptr<T> operator - (const varptr<T> a);

template<typename T>
varptr<T> sin (const varptr<T> a);

template<typename T>
varptr<T> cos (const varptr<T> a);

template<typename T>
varptr<T> tan (const varptr<T> a);

template<typename T>
varptr<T> csc (const varptr<T> a);

template<typename T>
varptr<T> sec (const varptr<T> a);

template<typename T>
varptr<T> cot (const varptr<T> a);

template<typename T>
varptr<T> exp (const varptr<T> a);

template <typename T>
varptr<T> sqrt (const varptr<T> a); // TODO implement

template <typename T>
varptr<T> pow (const varptr<T> a, T scalar); // TODO implement

template<typename T>
varptr<T> clip_val (const varptr<T> a, T min, T max);

template <typename T>
varptr<T> clip_norm (const varptr<T> a, T cap);

template<typename T>
varptr<T> operator + (T a, const varptr<T> b);

template<typename T>
varptr<T> operator + (const varptr<T> a, T b);

template<typename T>
varptr<T> operator + (const varptr<T> a, const varptr<T> b);

template<typename T>
varptr<T> operator - (T a, const varptr<T> b);

template<typename T>
varptr<T> operator - (const varptr<T> a, T b);

template<typename T>
varptr<T> operator - (const varptr<T> a, const varptr<T> b);

template<typename T>
varptr<T> operator * (T a, const varptr<T> b);

template<typename T>
varptr<T> operator * (const varptr<T> a, T b);

template<typename T>
varptr<T> operator * (const varptr<T> a, const varptr<T> b);

template<typename T>
varptr<T> operator / (T a, const varptr<T> b);

template<typename T>
varptr<T> operator / (const varptr<T> a, T b);

template<typename T>
varptr<T> operator / (const varptr<T> a, const varptr<T> b);

}

#include "../../../../src/graph/operation/general/elementary.ipp"

#endif /* elementary_hpp */
