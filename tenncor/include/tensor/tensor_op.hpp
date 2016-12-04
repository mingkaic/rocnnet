//
//  tensor.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-1.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <functional>
#include <stdexcept>
#include "tensor.hpp"

#pragma once
#ifndef tensor_op_hpp
#define tensor_op_hpp

namespace nnet
{

struct shapeinfo
{
	tensorshape res_shape_;
	std::vector<tensorshape> arg_shape_;
};

// TODO: allow TEN_OP to handler tensorshape before the change
// TODO: simplify tensor operation by implementing intermediate shape-data buffer object
// TEN_OP takes the resulting tensorshape, resulting raw data address, 
// and a vector of argument data address blocks
template <typename T>
using TEN_OP = std::function<void(shapeinfo,T*,std::vector<const T*>)>;

using SHAPE = std::function<tensorshape(std::vector<tensorshape>)>;

// operates by pull protocol (non-reactive)
template <typename T, typename A=ram_alloc>
class tensor_op : public tensor<T,A>
{
	private:
		const T zero = 0;

		TEN_OP<T> op_;
		SHAPE shape_;
		std::vector<const T*> raws_;
		shapeinfo info_;

	protected:
		void copy (const tensor_op<T,A>& other);
		tensor_op (const tensor_op<T,A>& other);
		virtual tensor<T,A>* clone_impl (void);

		virtual void raw_update (void);

		// inherited and override. evaluates before returning raw. 
		// told you it's not overengineered
		virtual T* get_raw (void);

	public:
		tensor_op (TEN_OP<T> op, SHAPE shaper);
		tensor_op (TEN_OP<T> op, SHAPE shaper, const alloc_attrib& attrib);

		tensor_op<T,A>* clone (void);
		virtual tensor_op<T,A>& operator = (tensor_op<T,A>& other);

		// buffer arguments
		// null tensors's raw are recorded as null as well
		virtual const tensor_op<T,A>& operator () (std::vector<tensor<T,A>*> args);
		
		// overwrite tensor's get
		virtual T get (std::vector<size_t> indices);
};

}

#include "../../src/tensor/tensor_op.ipp"

#endif /* tensor_op_hpp */
