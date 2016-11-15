//
//  group.ipp
//  cnnet
//
//  Created by Mingkai Chen on 2016-10-23.
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#ifdef group_hpp

namespace nnet
{
    
template <typename T>
void async_group<T>::add (iexecutor<T>* exe)
{
	acts_.emplace(exe);
}

template <typename T>
void async_group<T>::execute (void) {} // not implemented

template <typename T>
void group<T>::add (iexecutor<T>* exe)
{
	acts_.push_back(exe);
}

template <typename T>
void group<T>::execute (void)
{
	// stage
	for (iexecutor<T>* exe : acts_)
	{
		exe->freeze();
	}
	// execute
	for (iexecutor<T>* exe : acts_)
	{
		exe->execute();
	}
}

}

#endif