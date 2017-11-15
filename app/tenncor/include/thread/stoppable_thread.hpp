//
// Created by Mingkai Chen on 2017-10-01.
//

#include <thread>
#include <future>
#include <experimental/tuple>

#include "thread/stop_flag.hpp"

#pragma once
#ifndef STOPPABLE_THREAD_HPP
#define STOPPABLE_THREAD_HPP

namespace nnet
{

class stoppable_thread
{
public:
	stoppable_thread (void);

	template <typename FUNC, typename... ARGS>
	explicit stoppable_thread (FUNC&& f, ARGS&&... args)
	{
		std::promise<stop_flag*> promise_flag;

		internal_op_ = std::thread([&promise_flag,
			f = std::forward<FUNC>(f),
			args = std::make_tuple(std::forward<ARGS>(args)...)]
		{
			promise_flag.set_value(&thread_stop_flag);

			try
			{
				std::experimental::apply(f, args);
			}
			catch (thread_interrupted const&) {}
		});

		// expose flag to main thread
		flag_ = promise_flag.get_future().get();
	}

	~stoppable_thread (void);

	stoppable_thread (stoppable_thread&& other) noexcept;

	stoppable_thread& operator = (stoppable_thread&& other) noexcept;

	void join (void);

	void detach (void);

	bool joinable (void) const;

	void stop (void);

private:
	std::thread internal_op_;
	
	stop_flag* flag_;
};

}

#endif /* STOPPABLE_THREAD_HPP */
