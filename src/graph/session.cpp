//
//  session.cpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-09-26.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include "../../include/graph/session.hpp"

#ifdef session_hpp

namespace nnet {

session & session::get_instance (void) {
    static session my_instance;
    return my_instance;
}

// void register_obj (ivariable<std::any>& obj) {
//     registry.insert(&obj);
// }

}

#endif
