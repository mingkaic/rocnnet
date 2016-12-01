//
//  iexecutor.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-12.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef executor_hpp

namespace nnet
{

template <typename T>
iexecutor<T>* iexecutor<T>::clone (void)
{
    return clone_impl();
}

template <typename T>
void iexecutor<T>::add (ivariable<T>* node)
{
    dependencies_.push_back(node);
}

}

#endif
