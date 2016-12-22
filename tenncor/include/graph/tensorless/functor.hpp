//
//  functor.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-06.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "graph/iconnector.hpp"

#pragma once
#ifndef functor_hpp
#define functor_hpp

namespace nnet
{
	
template <typename T>
using BUILD_FUNCT = std::function<ivariable<T>*(varptr<T>)>;

template <typename T>
class functor : public iconnector<T>
{
	private:
		std::vector<functor<T>*> succession_;
		BUILD_FUNCT<T> builder_;
		ivariable<T>* root_ = nullptr; // delay instantiated
		
		// mimics its dependency in every way, acts as an insolator to safely destroy graph
		// also be smart and actually allocate on heap (no static builder_ here)
		class buffer : public iconnector<T>
		{
			private:
				ivariable<T>* get (void) const { return sub_to_var<T>(this->dependencies_[0]); }
		
			public:
				buffer (ivariable<T>* var) : 
					iconnector<T>(std::vector<ivariable<T>*>{var}, var->get_name()) {}
				virtual buffer* clone (void) { return new buffer(*this); }
		
				virtual tensorshape get_shape (void) { return get()->get_shape(); }
				virtual tensor<T>* get_eval (void) { return get()->get_eval(); }
				virtual ivariable<T>* get_gradient (void) { return get()->get_gradient(); }
				virtual functor<T>* get_jacobian (void)
				{
					if (iconnector<T>* c = dynamic_cast<iconnector<T>*>(get()))
					{
						return c->get_jacobian();
					}
					return nullptr;
				}
		
				// directly pass notification directly to audience
				virtual void update (ccoms::caller_info info, ccoms::update_message msg = ccoms::update_message())
				{ this->notify(msg); }
		};
		
		buffer* leaf_ = nullptr;
		
		void remake_leaf (void) // refreshes entire functor
		{
			if (leaf_)
			{
				delete leaf_;
			}
			leaf_ = new buffer(sub_to_var<T>(this->dependencies_[0]));
			leaf_->set_death((void**) &leaf_);
			root_ = nullptr;
		}

		functor (const functor<T>& other) :
			iconnector<T>(other),
			builder_(other.builder_),
			succession_(other.succession_) { remake_leaf(); }

		functor (const functor<T>& src, functor<T>* top) : functor<T>(src)
		{
			builder_ = top->builder_;
			succession_ = top->succession_;
			succession_.push_back(const_cast<functor<T>*>(&src));
		}

		virtual ivariable<T>* clone_impl (std::string name) { return new functor(*this); }
		
		functor (ivariable<T>* leaf,  BUILD_FUNCT<T> build) :
			builder_(build),
			iconnector<T>(std::vector<ivariable<T>*>{leaf}, "") { remake_leaf(); }

	public:
		static functor* build (ivariable<T>* leaf, 
			std::function<ivariable<T>*(varptr<T>)> build)
		{
			if (nullptr == leaf) return nullptr;
			return new functor(leaf, build);
		}

		// kill functor
		virtual ~functor (void)
		{
			if (leaf_) delete leaf_;
		}

		// COPY
		virtual functor* clone (void)
		{
			return new functor(*this);
		}
		functor& operator = (const functor& other)
		{
			if (this != &other)
			{
				iconnector<T>::operator = (other);
				succession_ = other.succession_;
				builder_ = other.builder_;
				remake_leaf();
			}
			return *this;
		}

		virtual std::string get_name (void) const
		{
			if (nullptr == root_) return "";
			return root_->get_name();
		}
		ivariable<T>* init (void)
		{
			if (nullptr == root_)
			{
				root_ = builder_(leaf_);
				for (auto rit = succession_.rbegin(); succession_.rend() != rit; rit++)
				{
					root_ = ((*rit)->builder_)(root_);
				}
			}
			return root_;
		}
		
		// spawn a new functor appending input leaf to this
		virtual functor<T>* append_leaf (ivariable<T>* base_root)
		{
			functor<T>* cpy = new functor<T>(base_root, builder_);
			cpy->succession_ = this->succession_;
			return cpy;
		}
		// make new functor with appending other's root to this leaf
		virtual functor<T>* append_functor (functor<T>* other)
		{
			// we take other's leaf but this build and succession stack as the new functor
			return new functor(*other, this);
		}

		virtual tensorshape get_shape (void) { return init()->get_shape(); }
		virtual tensor<T>* get_eval (void) { return init()->get_eval(); }
		virtual ivariable<T>* get_gradient (void) { return init()->get_gradient(); }
		virtual functor<T>* get_jacobian (void) {
			if (iconnector<T>* c = dynamic_cast<iconnector<T>*>(init()))
			{
				return c->get_jacobian();
			}
			return nullptr;
		}
		
		virtual void update (ccoms::caller_info info, ccoms::update_message msg = ccoms::update_message())
		{ this->notify(msg); }
};
	
}

#include "../../../src/graph/tensorless/functor.ipp"

#endif /* functor_hpp */
