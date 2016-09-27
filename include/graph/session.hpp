//
//  session.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-09-26.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

// #include <any> // since 2017
#include <unordered_set>
#include "variable.hpp"

#pragma once
#ifndef session_hpp
#define session_hpp

namespace nnet {

// singleton object controller
class session {
    private:
        // std::set<ivariable<std::any>*> registry;
        std::unordered_set<void*> registry;

    protected:
        session (void) {}
        ~session (void) {}

    public:
        const session & get_instance (void);

        // delete all copiers
        session (session const&) = delete;
        session (session&&) = delete;
        session& operator = (session const&) = delete;
        session& operator = (session &&) = delete;

        // object management
        // void register_obj (ivariable<std::any>& obj);

        template <typename T>
        void register_obj (ivariable<T>& obj) {
            registry.insert(&obj);
        }

        template <typename T>
        void initialize_all (void) {
            // replace void* with ivariable<T>*
            for (void* ivar : registry) {
                variable<T>* var = dynamic_cast<variable<T>*>(ivar);
                var->initialize();
            }
        }

        // input is resultant operator required to deep copy the graph
        template <typename T>
        ivariable<T>* copy (ivariable<T>* src_res) {
            return nullptr;
        }
};

}

#endif /* session_hpp */
