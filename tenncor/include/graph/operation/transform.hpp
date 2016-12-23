//
//  transform.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-09.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/operation/ioperation.hpp"

#pragma once
#ifndef transform_hpp
#define transform_hpp

namespace nnet
{

template <typename T>
T mean (const std::vector<T>& data)
{
	T ans = 0;
	for (T raw : data)
	{
		ans += raw;
	}
	ans /= data.size();
	return ans;
}

// special tensor transform

template <typename T>
class transform : public ioperation<T>
{
	private:
		BUILD_DERIVE<T> der_; // shallow

	protected:
		virtual ivariable<T>* setup_gradient (void);

		// protect transform constructor to ensure heap allocation
		transform (std::vector<ivariable<T>*> args, TEN_OP<T> op, SHAPE trans,
			BUILD_DERIVE<T> der, std::string name);
		
	public:
		static transform<T>* build (std::vector<ivariable<T>*> args,
			TEN_OP<T> op, SHAPE trans,
			BUILD_DERIVE<T> der, std::string name = "")
		{
			return new transform<T>(args, op, trans, der, name);
		}

		// COPY
		transform<T>* clone (void);

		// MOVES
		// TODO: implement
};

template <typename T>
varptr<T> transpose (const varptr<T> a);

// fit to watch
template <typename T>
varptr<T> fit (const varptr<T> a, const varptr<T> watch);

template <typename T>
varptr<T> extend (const varptr<T> a, size_t index, size_t multiplier);

// compression of index -1 compresses all elements in a (result is a scalar)
template <typename T>
varptr<T> compress (const varptr<T> a, int index = -1,
	std::function<T(const std::vector<T>&)> collector = mean<T>);

}

#include "../../../src/graph/operation/transform.ipp"

#endif /* transform_hpp */