//
// Created by Mingkai Chen on 2017-10-01.
//

#include <mutex>
#include "thread/stop_flag.hpp"

#ifdef STOP_FLAG_HPP

namespace nnet
{

thread_local stop_flag thread_stop_flag;

void stop_flag::set (void)
{
	flag_.store(true, std::memory_order_relaxed); // no constraints
	cond_.notify_one(); // should have only stopped once per thread
}

bool stop_flag::is_set (void) const
{
	return flag_.load(std::memory_order_relaxed);
}

void stop_throw (void)
{
	if (thread_stop_flag.is_set())
	{
		throw thread_interrupted(); // force worker to stop operation
	}
}

void stop_wait (void)
{
	std::mutex mut;
	std::unique_lock<std::mutex> lk(mut);
	thread_stop_flag.get_condition().wait(lk,
	[]{ return thread_stop_flag.is_set(); });
}

}

#endif
