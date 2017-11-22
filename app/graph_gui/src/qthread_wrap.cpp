//
// Created by mingkaichen on 11/15/17.
//

#include "../include/qthread_wrap.hpp"

#ifdef QTHREAD_WRAP_HPP

namespace tenncor_graph
{

void qthread_wrap::run (void)
{
	runnable_();
}

}

#endif
