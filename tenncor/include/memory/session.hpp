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

#pragma once
#ifndef session_hpp
#define session_hpp

namespace nnet {

template <typename T>
class ivariable;

template <typename T>
class variable;

// singleton object controller
class session {
	private:
		// std::unordered_set<std::any> registry;
		std::unordered_set<void*> registry;
		std::default_random_engine generator;

	protected:
		bool shape_eval = false;
		session (void) : generator(std::time(NULL)){}
		~session (void) {}

	public:
		static session& get_instance (void);
		static std::default_random_engine& get_generator (void);
		static bool pre_shape_eval (void) {
			session& inst = get_instance();
			return inst.shape_eval;
		}

		// delete all copiers
		session (session const&) = delete;
		session (session&&) = delete;
		session& operator = (session const&) = delete;
		session& operator = (session &&) = delete;

		// member setter
		void seed_rand_eng (size_t seed) { generator.seed(seed); }
		// member getter
		std::default_random_engine& get_rand_generator (void) { return generator; }

		// object management
		// void register_obj (ivariable<std::any>& obj);

		template <typename T>
		void register_obj (ivariable<T>& obj) {
			registry.emplace(&obj);
		}

		template <typename T>
		void unregister_obj (ivariable<T>& obj) {
			registry.erase(&obj);
		}

		template <typename T>
		void initialize_all (void) {
			// replace void* with ivariable<T>*
			for (void* ivar : registry) {
				variable<T>* var =
					dynamic_cast<variable<T>*>((ivariable<T>*) ivar);
				if (nullptr != var && var->can_init()) {
					var->initialize();
				}
			}
		}

		template <typename T>
		void save_all (void); // TODO implement

		// controls whether to shape evaluate at object construction during runtime
		// potentially save space when running slow operations
		// this option is not always desirable since full shape definition is required for shape evaluation
		void enable_shape_eval (void) { shape_eval = true; }
		void disable_shape_eval (void) { shape_eval = false; }

		// input is resultant operator required to deep copy the graph
		template <typename T>
		ivariable<T>* copy (ivariable<T>* src_res) {
			return nullptr;
		}
};

}

#endif /* session_hpp */
