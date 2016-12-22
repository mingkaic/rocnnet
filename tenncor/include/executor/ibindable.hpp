//
//  ibindable.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-19.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <experimental/optional>

#pragma once
#ifndef ibindable_hpp
#define ibindable_hpp

namespace nnet
{

template <typename T>
class ibindable
{
	private:
		struct bound_vector
		{
			std::experimental::optional<size_t> active_;
			std::vector<ibindable<T>*> bounds_;
			
			bound_vector (ibindable<T>* a, ibindable<T>* b) :
				bounds_({a, b}) {}
		};
		
		std::shared_ptr<bound_vector> shared_bound_ = nullptr;
		
	protected:
		void declare_active (void)
		{
			for (ibindable<T>* b : shared_bound_->bounds)
			{
				if (this != b)
				{
					b->deactivate();
				}
			}
		}
		
	public:
		virtual void deactivate (void) = 0;
	
		void bind (ibindable<T>& other)
		{
			// no-existant bound
			if (nullptr == shared_bound_ && 
				nullptr == other.shared_bound_)
			{
				shared_bound_ = std::make_shared<bound_vector>(this, &other);
			}
			// conflict
			else if (nullptr != shared_bound_ && 
					nullptr != other.shared_bound_)
			{
				shared_bound_ = std::make_shared<bound_vector>(*shared_bound_);
				shared_bound_->active_.reset();
				shared_bound_->bounds_.insert(other.shared_bound_.begin(), other.shared_bound_.end());
			}
			else if (nullptr == shared_bound_)
			{
				shared_bound_ = other.shared_bound_; 
				shared_bound_->bounds.insert(this);
			}
			else
			{
				other.shared_bound_ = shared_bound_;
				shared_bound_->bounds.insert(&other);
			}
		}
		
		// return if this is active
		bool check_active (void)
		{
			if (nullptr == shared_bound_ || 
				shared_bound_->active_)
			{
				return false;
			}
			size_t idx = shared_bound_->active_;
			return this == shared_bound_->bounds_[idx];
		}
		
		void reset (void) { if (shared_bound_) shared_bound_->active_.reset(); }
};

}

#endif /* ibindable_hpp */