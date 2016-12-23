//
//  ibindable.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-12-19.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

// TODO: uncomment when c++17
#include <experimental/optional>
#include "utils/temp_utils.hpp"

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
			std::experimental::optional<size_t> active_; // stores the last toggle activated
			std::vector<ibindable<T>*> bounds_;
			
			bound_vector (ibindable<T>* a, ibindable<T>* b) :
				bounds_({a, b}) {}
		};

		std::string last_key_ = "";
		std::unordered_map<std::string, std::shared_ptr<bound_vector> > shared_bounds_;

	protected:
		void declare_active (std::string tid)
		{
			if (tid.empty())
			{
				tid = last_key_;
			}
			auto sb_pair = shared_bounds_.find(tid);
			if (shared_bounds_.end() == sb_pair) return;

			for (ibindable<T>* b : sb_pair->second->bounds_)
			{
				if (this != b)
				{
					b->deactivate(tid);
				}
			}
		}
		
	public:
		virtual void deactivate (std::string tid) = 0;

		void bind (std::string bid, ibindable<T>* other)
		{
			// store bind id
			last_key_ = other->last_key_ = bid;

			auto my_vect = shared_bounds_.find(last_key_);
			auto their_v = other->shared_bounds_.find(last_key_);
			auto my_end = shared_bounds_.end();
			auto their_e = other->shared_bounds_.end();

			// conflict bound
			if (my_end != my_vect && their_e != their_v)
			{
				// choose my_vect as the dominant vector (merge other into mine)
				std::vector<ibindable<T>*>& mybounds = my_vect->second->bounds_;
				std::vector<ibindable<T>*>& theirbounds = their_v->second->bounds_;
				mybounds.insert(mybounds.end(), theirbounds.begin(), theirbounds.end());
				// their vector take mine
				their_v->second = my_vect->second;
			}
			// desired bound does not exist in neither vector
			else if (my_end == my_vect && their_e == their_v)
			{
				shared_bounds_[last_key_] =
				other->shared_bounds_[last_key_] =
					std::make_shared<bound_vector>(this, other);
			}
			else if (my_end == my_vect)
			{
				their_v->second->bounds_.push_back(this);
				shared_bounds_[last_key_] = their_v->second;
			}
			else // if (their_e == their_v)
			{
				my_vect->second->bounds_.push_back(this);
				other->shared_bounds_[last_key_] = my_vect->second;
			}
			// reset once c++17 is used
//			shared_bounds_[last_key_]->active_.reset();
		}
		
		// check if this instance is active in any shared_bounds
		bool check_active (void)
		{
			bool act = false;
			for (auto sb_pair : shared_bounds_)
			{
				if (false == sb_pair.second->active_)
				{
					size_t idx = sb_pair.second->active_;
					act = act || this == sb_pair.second->bounds_[idx];
				}
			}
			return act;
		}

		void which_active (std::unordered_set<std::string>& id_set)
		{
			for (auto sb_pair : shared_bounds_)
			{
				if (false == sb_pair.second->active_)
				{
					size_t idx = sb_pair.second->active_;
					if (this == sb_pair.second->bounds_[idx])
					{
						id_set.emplace(sb_pair.first);
					}
				}
			}
		}
};

}

#endif /* ibindable_hpp */