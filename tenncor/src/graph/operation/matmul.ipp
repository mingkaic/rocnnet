//
//  matmul.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-08.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef matop_hpp

namespace nnet
{

// MATRIX MULTIPLICATION

template <typename T>
size_t matmul<T>::common_dim (void)
{
	std::vector<size_t> t = sub_to_var<T>(this->dependencies_[0])->get_shape().as_list();
	if (transposeA_)
	{
		return 2 == t.size() ? t[1] : 1;
	}
	return t[0];
}

template <typename T>
void matmul<T>::setup_gradient (void)
{
	// matmul'(f, g) = jacobian(f, g)
	ivariable<T>* arga = sub_to_var<T>(this->dependencies_[0]);
	ivariable<T>* argb = sub_to_var<T>(this->dependencies_[1]);
	assert(arga && argb);

	// matmul is special in that its grad_jacobi_ is the default jacobian leaf one
	// grad_ is the jacobian instead
	ioperation<T>* fgrad = dynamic_cast<ioperation<T>*>(
		fit<double>(constant<T>::build(1), this).get());
	
	this->grad_ = std::unique_ptr<iconnector<T> >(fgrad);
		
	functor<T>* jacobian = functor<T>::build(fgrad,
	[this, arga, argb](varptr<T> leaf)
	{
		varptr<T> grada = arga->get_gradient();
		varptr<T> gradb = argb->get_gradient();
	
		varptr<T> mA = matmul<T>::build(leaf, argb, transposeA_, !transposeB_);
		varptr<T> mB = matmul<T>::build(arga, leaf, !transposeA_, transposeB_);

		varptr<T> res = not_zero(mA * grada, mB * gradb);
		
		return res;
	});
	
	iconnector<T>* oga = dynamic_cast<iconnector<T>*>(arga->get_gradient());
	iconnector<T>* ogb = dynamic_cast<iconnector<T>*>(argb->get_gradient());
	
	// check if oga or ogb for jacobi
	functor<T>* candidate_a = oga ? oga->get_jacobian() : nullptr;
	functor<T>* candidate_b = ogb ? ogb->get_jacobian() : nullptr;
	// we can only have one candidate
	assert(nullptr == candidate_a || nullptr == candidate_b);
	functor<T>* chief = candidate_a ? candidate_a : candidate_b;
	if (chief)
	{
		// chief absorb prime's root as this leaf/leaves
		chief = chief->append_functor(jacobian);
	}
	else
	{
		// first chief in the line of succession
		chief = jacobian;
	}

	fgrad->set_jacobian(chief);
}

template <typename T>
matmul<T>::matmul (ivariable<T>* a, ivariable<T>* b,
	bool transposeA, bool transposeB) :
	ioperation<T>((std::vector<ivariable<T>*>{a, b}),
	"(" + a->get_name() + "•" + b->get_name() + ")"),
	transposeA_(transposeA), transposeB_(transposeB)
{
	this->shaper_ = [this](std::vector<tensorshape> shapes) -> tensorshape
	{
		tensorshape t1s = shapes[0];
		tensorshape t2s = shapes[1];

		if (5 > (t1s.n_dims() + t2s.n_dims()))
		{
			std::vector<size_t> al = t1s.as_list();
			std::vector<size_t> bl = t2s.as_list();

			size_t ax = t1s.n_dims() ? al[0] : 0;
			size_t ay = t1s.n_dims() > 1 ? al[1] : 1;
			size_t bx = t2s.n_dims() ? bl[0] : 0;
			size_t by = t2s.n_dims() > 1 ? bl[1] : 1;

			if (ay == bx && transposeA_ && transposeB_)
			{
				return std::vector<size_t>{by, ax};
			}
			else if (ay == by && transposeA_)
			{
				return std::vector<size_t>{bx, ax};
			}
			else if (ax == bx && transposeB_)
			{
				return std::vector<size_t>{by, ay};
			}
			else if (ax == by && !transposeA_ && !transposeB_)
			{
				return std::vector<size_t>{bx, ay};
			}
		}
		return tensorshape();
	};
	this->out_ = std::make_unique<tensor_op<T> >(
	[this](shapeinfo info, T* dest, std::vector<const T*> srcs)
	{
		info.res_shape_.assert_is_fully_defined();
		std::vector<size_t> dims = info.res_shape_.as_list();
		size_t dimX = dims[0]; size_t dimY = dims[1];
		size_t dimZ = common_dim();

		const T* rawa = srcs[0];
		const T* rawb = srcs[1];
		for (size_t y = 0; y < dimY; y++)
		{
			for (size_t x = 0; x < dimX; x++)
			{
				dest[x+y*dimX] = 0;
				for (size_t z = 0; z < dimZ; z++)
				{
					size_t aidx = transposeA_ ? y+z*dimY : z+y*dimZ;
					size_t bidx = transposeB_ ? z+x*dimZ : x+z*dimX;
					dest[x+y*dimX] += rawa[aidx] * rawb[bidx];
				}
			}
		}
	}, this->shaper_);
	// try to update
	if (session::pre_shape_eval())
	{
		this->shape_eval().assert_is_fully_defined();
	}
	update(ccoms::caller_info());
}

template <typename T>
matmul<T>* matmul<T>::clone (void)
{
	return new matmul(*this);
}

template <typename T>
void matmul<T>::update (ccoms::caller_info info, ccoms::update_message msg)
{
	// UPDATING ARGUMENTS
	// caller is never used because we know matmul will never be the parent of a gradient leaf
	ivariable<T>* a = sub_to_var<T>(this->dependencies_[0]);
	ivariable<T>* b = sub_to_var<T>(this->dependencies_[1]);
	tensor<T>* at = a->get_eval();
	tensor<T>* bt = b->get_eval();

	this->valid_tensor_ = at && bt;
	if (this->valid_tensor_)
	{
		(*this->out_)(std::vector<tensor<T>*>{at, bt});
	}

	msg.grad_ = nullptr;
	this->notify(msg);
}

}

#endif