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

// TODO: replace with tensor of tensors.
// jacobian takes its root AND its buffer k as dependencies
template <typename T>
class matmul<T>::jgraph : public igraph<T>
{
	protected:
		void copy (const jgraph& other, std::string name = "")
		{
			iconnector<T>::copy(other, name);
		}
		jgraph (const jgraph& other, std::string name) : igraph<T>(other, name) {}

		virtual ivariable<T>* clone_impl (std::string name)
		{
			return new jgraph(*this, name);
		}

		virtual buffer<T>* get_leaf (void) const
		{
			return dynamic_cast<buffer<T>*>(this->dependencies_[1]->get_owner());
		}

		jgraph (ivariable<T>* root, buffer<T>* leaf) :
			igraph<T>(root, leaf) {}

	public:
		static jgraph* build (ivariable<T>* root, buffer<T>* leaf)
		{
			return new jgraph(root, leaf);
		}

		// COPY
		jgraph* clone (std::string name = "")
		{
			return static_cast<jgraph*>(clone_impl(name));
		}
		jgraph& operator = (const jgraph& other)
		{
			if (this != &other)
			{
				copy(other);
			}
			return *this;
		}

		virtual void connect_graph (igraph<T>* g_other)
		{
			buffer<T>* l_buffer = get_leaf();
			ivariable<T>* g_root = g_other->get_root();
			if (l_buffer->get() != g_root)
			{
				*l_buffer = *g_root;
			}
		}
		virtual void update_leaf (std::function<ivariable<T>*(ivariable<T>*,size_t)> lassign)
		{
			buffer<T>* b = get_leaf();
			// reassign current buffer
			*b = *lassign(b->get(), 0);
		}

		// override from buffer
		virtual tensorshape get_shape (void) const { return this->get_root()->get_shape(); }
		// special jacobian: eval the leaf instead of root
		virtual tensor<T>* get_eval (void) { return get_leaf()->get_eval(); }
		// jacobian special: evaluate leaf
		virtual ivariable<T>* get_gradient (void) { return this->get_root()->get_gradient(); }
		virtual void update (ccoms::caller_info info, ccoms::update_message msg = ccoms::update_message())
		{
			size_t callerid = info.caller_idx_;
			// ignore leaf
			if (callerid != 0) return;

			// core jacobian logic
			if (igraph<T>* jac = dynamic_cast<igraph<T>*>(msg.jacobi_))
			{
				// connect the jacobian from below the tree as the top jacobian
				jac->connect_graph(this);
			}
			else
			{
				msg.jacobi_ = this;
			}

			msg.grad_ = nullptr;
			this->notify(msg);
		}
};

// MATRIX MULTIPLICATION

template <typename T>
size_t matmul<T>::common_dim (void) const
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

	varptr<T> grada = arga->get_gradient();
	varptr<T> gradb = argb->get_gradient();

	// matmul is special in that its grad_jacobi_ is the default jacobian leaf one
	// grad_ is the jacobian instead
	this->grad_ = dynamic_cast<iconnector<T>*>(fit<double>(constant<T>::build(1), this).get());
	buffer<T>* temp = buffer<T>::build(this->grad_);
	varptr<T> mA = matmul<T>::build(temp, argb, transposeA_, !transposeB_);
	varptr<T> mB = matmul<T>::build(arga, temp, !transposeA_, transposeB_);

	// TODO: we must shut down shape eval here (make pattern easier?)
	varptr<T> res = mA * grada + mB * gradb;
	// reset shape eval here if necessary

	this->grad_jacobi_ = jgraph::build(res, temp);
}

template <typename T>
matmul<T>::matmul (const matmul<T>& other, std::string name) :
	ioperation<T>(other, name),
	transposeA_(other.transposeA_),
	transposeB_(other.transposeB_) {}

template <typename T>
ivariable<T>* matmul<T>::clone_impl (std::string name) {
	return new matmul<T>(*this, name);
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
matmul<T>* matmul<T>::clone (std::string name)
{
	return static_cast<matmul<T>*>(clone_impl(name));
}

template <typename T>
matmul<T>& matmul<T>::operator = (const matmul<T>& other)
{
	if (this != &other)
	{
		if (const matmul<T>* mptr = dynamic_cast<const matmul<T>*>(&other))
		{
			transposeA_ = mptr->transposeA_;
			transposeB_ = mptr->transposeB_;
		}
		this->copy(other);
	}
	return *this;
}

template <typename T>
void matmul<T>::update (ccoms::caller_info info, ccoms::update_message msg)
{
	// common update protocol
	this->message_update(msg);

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