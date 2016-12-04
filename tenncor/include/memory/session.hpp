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
#include "memory/safe_ptr.hpp"

#pragma once
#ifndef session_hpp
#define session_hpp

namespace nnet
{

template <typename T>
class ivariable;

template <typename T>
class variable;

// singleton object controller
class session
{
	private:
		// std::unordered_set<std::any> registry;
		std::unordered_set<void*> registry;
		std::default_random_engine generator_;

	protected:
		bool shape_eval = false;
		session (void) : generator_(std::time(NULL)) {}
		~session (void) {}

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
		// void register_obj (ivariable<std::any>& obj);

		template <typename T>
		void register_obj (ivariable<T>& obj) { registry.emplace(&obj); }

		template <typename T>
		void unregister_obj (ivariable<T>& obj) { registry.erase(&obj); }
		
		bool ptr_registered (void* ptr)
		{ 
			return registry.end() != registry.find(ptr);
		}

		template <typename T>
		void initialize_all (void)
		{
			for (void* ivar : registry)
			{
				// cast void* to ivariable<T>*
				variable<T>* var = dynamic_cast<variable<T>*>((ivariable<T>*) ivar);
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

		// input is resultant operator required to deep copy the graph
		template <typename T>
		ivariable<T>* copy (ivariable<T>* src_res) {
			return nullptr;
		}
};

}

#endif /* session_hpp */
