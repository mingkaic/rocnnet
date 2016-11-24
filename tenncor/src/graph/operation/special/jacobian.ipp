//
//  jacobian.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-06.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef jacobian_hpp

namespace nnet
{

template <typename T>
class jacobian<T>::hidden_jacobi : public ioperation<T>
{
	protected:
		virtual void setup_gradient (void) {}
		// this isn't ever used... tensor_jacobi evaluates its own shape
		virtual tensorshape shape_eval (void) { return std::vector<size_t>{}; }

		virtual ivariable<T>* clone_impl (std::string name)
		{
			return new hidden_jacobi(*this, name);
		}

		hidden_jacobi (const hidden_jacobi& other, std::string name) :
			ccoms::iobserver(other),
			ivariable<T>(other, name),
			ioperation<T>(other, name) {}

	public:
		hidden_jacobi (jacobian<T>* outer, bool transposeA, bool transposeB) :
			ccoms::iobserver(std::vector<ccoms::subject*>{outer}),
			ivariable<T>(std::vector<size_t>{}, "jacobian_hidden"),
			ioperation<T>(std::vector<ivariable<T>*>{outer}, "jacobian_hidden")
		{
			this->out_ = std::make_unique<tensor_jacobi<T> >(transposeA, transposeB);
		}

        // inner copy
		hidden_jacobi* clone (std::string name = "")
		{
			return dynamic_cast<hidden_jacobi*>(clone_impl(name));
		}

        hidden_jacobi& operator = (const hidden_jacobi& other)
        {		
        	if (this != &other)
        	{
        		this->copy(other);
        	}
        	return *this;
        }

		hidden_jacobi& operator () (ivariable<T>* a, ivariable<T>* b)
		{
			(*this->out_)(std::vector<tensor<T>*>{a->get_eval(), b->get_eval()});
			return *this;
	    }

		virtual void update (ccoms::subject* caller)
		{
			this->notify();
		}
};

template <typename T>
jacobian<T>::jacobian (const jacobian<T>& other, std::string name) :
	ccoms::iobserver(other),
	ivariable<T>(other, name),
	ioperation<T>(other, name)
{
	hidden_ = std::unique_ptr<hidden_jacobi>(other.hidden_->clone());
}

template <typename T>
ivariable<T>* jacobian<T>::clone_impl (std::string name)
{
	return new jacobian<T>(*this, name);
}

template <typename T>
bool jacobian<T>::channel (std::stack<ivariable<T>*>& jacobi)
{
	jacobi.push(hidden_.get());
	// ioperation's channel looks in dependencies indiscriminately,
	// which could be branching to some gradient state node (TODO: optimize once gradient order is implemented)
	// however, in jacobian's case, its dependencies are guaranteed non-gradient nodes,
	// so don't bother using ioperation<T>::jacobi to propagate channel
	size_t jacobi_count = 0;
	for (ccoms::subject* sub : this->dependencies_)
	{
		if (ioperation<T>* o = dynamic_cast<ioperation<T>*>(sub))
		{
			// operation node's gradient SHOULD be an operation
			ioperation<T>* go = dynamic_cast<ioperation<T>*>(o->get_gradient());
			if (go->channel(jacobi))
			{
				jacobi_count++;
			}
		}
	}
	if (jacobi_count > 1)
	{
		throw std::logic_error(
		    "jacobian branch conflict occurred at " + this->get_name());
	}
	return jacobi_count != 0;
}

template <typename T>
jacobian<T>::jacobian (ivariable<T>* arga, ivariable<T>* argb,
	bool transposeA, bool transposeB, std::string name) :
	ccoms::iobserver(std::vector<ccoms::subject*>{arga, argb}),
	ivariable<T>(std::vector<size_t>{}, name),
	ioperation<T>(std::vector<ivariable<T>*>{arga, argb}, name)
{
	hidden_ = std::make_unique<hidden_jacobi>(this, transposeA, transposeB);
    this->out_ = std::make_unique<tensor<T> >(1);
    this->valid_tensor_ = true; // jacobian is always valid
}

template <typename T>
jacobian<T>* jacobian<T>::clone (std::string name)
{
	return static_cast<jacobian<T>*>(clone_impl(name));
}

template <typename T>
jacobian<T>& jacobian<T>::operator = (const jacobian<T>& other)
{
	if (this != &other)
	{
		hidden_ = std::unique_ptr<hidden_jacobi>(other.hidden_->clone());
		this->copy(other);
	}
	return *this;
}

template <typename T>
void jacobian<T>::update (ccoms::subject* caller)
{
	ivariable<T>* a = dynamic_cast<ivariable<T>*>(this->dependencies_[0]);
	ivariable<T>* b = dynamic_cast<ivariable<T>*>(this->dependencies_[1]);
	if (a && b)
	{
		(*hidden_)(a, b);
	}
	this->notify();
}

}

#endif