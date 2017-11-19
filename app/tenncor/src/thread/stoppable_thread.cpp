//
// Created by Mingkai Chen on 2017-10-01.
//

#include "thread/stoppable_thread.hpp"

#ifdef STOPPABLE_THREAD_HPP

namespace nnet
{

stoppable_thread::stoppable_thread (void) {}

stoppable_thread::~stoppable_thread (void)
{
	if (joinable())
	{
		join(); // wait on work unless interrupted
	}
}

stoppable_thread::stoppable_thread (stoppable_thread&& other) noexcept :
	internal_op_(std::move(other.internal_op_)),
	flag_(std::move(other.flag_)) {}

stoppable_thread& stoppable_thread::operator = (stoppable_thread&& other) noexcept
{
	if (this != &other)
	{
		internal_op_ = std::move(other.internal_op_);
		flag_ = std::move(other.flag_);
	}

	return *this;
}

void stoppable_thread::join (void)
{
	return internal_op_.join();
}

void stoppable_thread::detach (void)
{
	return internal_op_.detach();
}

bool stoppable_thread::joinable (void) const
{
	return internal_op_.joinable();
}

void stoppable_thread::stop (void)
{
	if (flag_)
	{
		flag_->set();
	}
}

}

#endif
