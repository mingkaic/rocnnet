//
//  session.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-09-26.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

// #include <any> // since 2017
#include <unordered_set>
#include <random>
#include <ctime>

#include "graph/react/subject.hpp"

#pragma once
#define session_hpp_S
#ifndef session_hpp_S
#define session_hpp

namespace nnet
{
//TODO: >>>>> UPDATE <<<<<
//template <typename T>
//class inode;

template <typename T>
class variable;

// singleton object controller
// manage graph info
class session
{
	public:
		static session& get_instance (void);
		static std::default_random_engine& get_generator (void);
		static bool pre_shape_eval (void);

		// delete all copier and movers
		session (session const&) = delete;
		session (session&&) = delete;
		session& operator = (session const&) = delete;
		session& operator = (session &&) = delete;

		// member setter
		void seed_rand_eng (size_t seed);
		// member getter
		std::default_random_engine& get_rand_generator (void);

		// object management
		// void register_obj (inode<std::any>& obj);

		template <typename T>
		void register_obj (inode<T>& obj) { registry_.emplace(&obj); }

		template <typename T>
		void unregister_obj (inode<T>& obj) { registry_.erase(&obj); }
		
		bool ptr_registered (subject* ptr) { return registry_.end() != registry_.find(ptr); }

		template <typename T>
		void initialize_all (void)
		{
			for (void* ivar : registry_)
			{
				// cast void* to inode<T>*
				variable<T>* var = dynamic_cast<variable<T>*>((inode<T>*) ivar);
				if (nullptr != var && var->can_init())
				{
					var->initialize();
				}
			}
		}

		template <typename T>
		void save_all (void); // TODO implement

		// controls whether to shape evaluate at object construction during runtime
		// potentially save space when running slow operations
		// this option is not always desirable since full shape definition is required for shape evaluation
		void enable_shape_eval (void);
		void disable_shape_eval (void);

		// TODO implement: input is resultant operator required to deep copy the graph
		template <typename T>
		inode<T>* copy (inode<T>* src_res) { return nullptr; }

	protected:
		bool shape_eval = false;
		session (void) : generator_(std::time(NULL)) {}
		~session (void)
		{
			// kill everything in registry_ in case session is killed early
			for (subject* so : registry_)
			{
				delete so;
			}
			registry_.clear();
		}

	private:
		// std::unordered_set<std::any> registry_;
		std::unordered_set<subject*> registry_;
		std::default_random_engine generator_;
};

}

#endif /* session_hpp */
