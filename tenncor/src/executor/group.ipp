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
group<T>::~group (void)
{
	for (auto exe_pair : acts_)
	{
		if (exe_pair.second)
		{
			delete exe_pair.first;
		}
	}
}

template <typename T>
void group<T>::add (iexecutor<T>* exe, bool owns)
{
	acts_.push_back(std::pair<iexecutor<T>*,bool>(exe, owns));
}

template <typename T>
void group<T>::execute (void)
{
	// stage
	for (auto exe_pair : acts_)
	{
		exe_pair.first->freeze();
	}
	// execute
	for (auto exe_pair : acts_)
	{
		exe_pair.first->execute();
	}
}

}

#endif