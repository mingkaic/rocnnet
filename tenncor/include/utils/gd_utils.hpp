//
// Created by Mingkai Chen on 2017-04-27.
//

#include "graph/varptr.hpp"
#include "graph/leaf/variable.hpp"
#include "graph/connector/immutable/elementary.hpp"

#pragma once
#ifndef ROCNNET_GD_UTILS_HPP
#define ROCNNET_GD_UTILS_HPP


namespace nnet
{

template <typename T>
using grad_process = std::function<varptr<T>(varptr<T>,variable<T>*)>;

template <typename T>
varptr<T> grad_identity (varptr<T> grad, variable<T>*)
{
	return grad;
}

//! gradient descent algorithm abstraction
template <typename T>
struct gd_updater
{
	virtual ~gd_updater(void) {}

	gd_updater<T>* clone (void) { return clone_impl(); }

	gd_updater<T>* move (void) { return move_impl(); }

	virtual std::vector<variable_updater<T> > calculate (inode<T>* root,
		grad_process<T> intermediate_process = grad_identity<T>)
	{
		std::vector<variable_updater<T> > updates;
		typename inode<T>::GRAD_CACHE leafset;
		root->get_leaves(leafset);
		for (auto lit : leafset)
		{
			variable<T>* Wb = lit.first;
			if (ignored_.end() == ignored_.find(Wb))
			{
				varptr<T> gres = root->get_gradient(Wb);
				updates.push_back(process_update(gres, Wb, intermediate_process));
			}
		}
		return updates;
	}

	void ignore_subtree (inode<T>* subroot)
	{
		typename inode<T>::GRAD_CACHE leafset;
		subroot->get_leaves(leafset);
		for (auto lit : leafset)
		{
			variable<T>* Wb = lit.first;
			ignored_.emplace(Wb);
		}
	}

	double learning_rate_ = 0.5;

protected:
	virtual gd_updater<T>* clone_impl (void) = 0;

	virtual gd_updater<T>* move_impl (void) = 0;

	virtual variable_updater<T> process_update (varptr<T>& gres,
		variable<T>* leaf, grad_process<T> intermediate_process) = 0;

private:
	std::unordered_set<variable<T>*> ignored_;
};

//! vanilla gradient descent algorithm
struct vgb_updater : gd_updater<double>
{
	vgb_updater* clone (void) { return static_cast<vgb_updater*>(clone_impl()); }

	vgb_updater* move (void) { return static_cast<vgb_updater*>(move_impl()); }

protected:
	virtual gd_updater<double>* clone_impl (void)
	{
		return new vgb_updater(*this);
	}

	virtual gd_updater<double>* move_impl (void)
	{
		return new vgb_updater(std::move(*this));
	}

	virtual variable_updater<double> process_update (varptr<double>& gres,
		variable<double>* leaf, grad_process<double> intermediate_process);
};

//! momentum based gradient descent
//! overview: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
//! updates using incremental update of velocity on error manifold

// Standard momentum:
// 1. velocity_t = discount * velocity_(t-1) - learning * J'[v]
// 2. delta(var) = velocity_t, update by gd
// lim t->inf (velocity) = -learning * J'[v] / (1-discount)

// Nestrov momentum:
// 1. delta(var) = velocity_t-1, update by gd
// 2. velocity_t = discount * velocity_(t-1) - learning * J'[v]
struct momentum_updater : gd_updater<double>
{
	momentum_updater* clone (void) { return static_cast<momentum_updater*>(clone_impl()); }

	momentum_updater* move (void) { return static_cast<momentum_updater*>(move_impl()); }

protected:
	virtual gd_updater<double>* clone_impl (void)
	{
		return new momentum_updater(*this);
	}

	virtual gd_updater<double>* move_impl (void)
	{
		return new momentum_updater(std::move(*this));
	}

	virtual variable_updater<double> process_update (varptr<double>& /*gres*/,
		variable<double>* /*leaf*/, grad_process<double> /*intermediate_process*/)
	{
		throw std::bad_function_call();
		return [](void) {};
	}
};

// Separate adaptive learning rates
// introduce leaf local_gain linked to weight/bias variables
// delta(var) = -epsilon * local_gain[v] * J'[v]
// if J'[v]_t * J'[v]_(t-1) > 0:
// then local_gain[v] += 0.05
// else local_gain[v] *= 0.95
struct adadelta_updater : gd_updater<double>
{
	adadelta_updater* clone (void) { return static_cast<adadelta_updater*>(clone_impl()); }

	adadelta_updater* move (void) { return static_cast<adadelta_updater*>(move_impl()); }

	double rho_ = 0.95;

	double epsilon_ = std::numeric_limits<double>::epsilon();

protected:
	virtual gd_updater<double>* clone_impl (void)
	{
		return new adadelta_updater(*this);
	}

	virtual gd_updater<double>* move_impl (void)
	{
		return new adadelta_updater(std::move(*this));
	}

	virtual variable_updater<double> process_update (varptr<double>& /*gres*/,
		variable<double>* /*leaf*/, grad_process<double> /*intermediate_process*/)
	{
		throw std::bad_function_call();
		return [](void) {};
	}
};

struct adagradupdater : gd_updater<double>
{
	adagradupdater* clone (void) { return static_cast<adagradupdater*>(clone_impl()); }

	adagradupdater* move (void) { return static_cast<adagradupdater*>(move_impl()); }

	double init_accum_ = 0.1;

protected:
	virtual gd_updater<double>* clone_impl (void)
	{
		return new adagradupdater(*this);
	}

	virtual gd_updater<double>* move_impl (void)
	{
		return new adagradupdater(std::move(*this));
	}

	virtual variable_updater<double> process_update (varptr<double>& /*gres*/,
		variable<double>* /*leaf*/, grad_process<double> /*intermediate_process*/)
	{
		throw std::bad_function_call();
		return [](void) {};
	}
};


//! root mean square propagation
// rms_delta = J'(v)_t
// rms_t = (1 - discount) * rms_t-1 + discount * rms_delta^2
// delta(var) = v_t = learning * rms_delta / rms_t
struct rmspropupdater : gd_updater<double>
{
	rmspropupdater* clone (void) { return static_cast<rmspropupdater*>(clone_impl()); }

	rmspropupdater* move (void) { return static_cast<rmspropupdater*>(move_impl()); }

	double discount_factor_ = 0.9;

	double momentum_ = 0;

	double epsilon_ = std::numeric_limits<double>::epsilon();

protected:
	virtual gd_updater<double>* clone_impl (void)
	{
		return new rmspropupdater(*this);
	}

	virtual gd_updater<double>* move_impl (void)
	{
		return new rmspropupdater(std::move(*this));
	}

	virtual variable_updater<double> process_update (varptr<double>& /*gres*/,
		variable<double>* /*leaf*/, grad_process<double> /*intermediate_process*/)
	{
		throw std::bad_function_call();
		return [](void) {};
	}
};

}


#endif /* ROCNNET_GD_UTILS_HPP */
