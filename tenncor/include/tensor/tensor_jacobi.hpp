//
//  tensor_jacobi.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-12.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "tensor.hpp"

#pragma once
#ifndef tensor_jacobi_hpp
#define tensor_jacobi_hpp

namespace nnet
{

template <typename T, typename A=ram_alloc>
class tensor_jacobi : public tensor<T, A>
{
	private:
		bool transposeA_;
		bool transposeB_;
		ivariable<T>* k_ = nullptr; // no ownership on k
		std::vector<std::shared_ptr<ivariable<T> > > owner_;
		ivariable<T>* root_ = nullptr; // deleted when clear

		void clear_ownership (void);

	protected:
		void copy (const tensor_jacobi<T,A>& other);
		tensor_jacobi (const tensor_jacobi<T,A>& other);
		virtual tensor<T,A>* clone_impl (void);

		// inherited and override to copy root tensor before releasing raw data
		virtual T* get_raw (void);

	public:
		tensor_jacobi (bool transposeA, bool transposeB);
		virtual ~tensor_jacobi (void);

		// COPY
		tensor_jacobi<T,A>* clone (void);
		tensor_jacobi<T,A>& operator = (const tensor_jacobi<T,A>& other);

		// non inherited
		void set_root (ivariable<T>* root);

		// construct root via input arguments. should be executed at evaluation time
		virtual const tensor_jacobi<T,A>& operator () (
			ivariable<T>* arga, 
			ivariable<T>* argb);
};

}

#include "../../src/tensor/tensor_jacobi.ipp"

#endif /* tensor_jacobi_hpp */
