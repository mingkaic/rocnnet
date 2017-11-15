//
// Created by Mingkai Chen on 2017-10-01.
//

#include <atomic>
#include <condition_variable>

#pragma once
#ifndef STOP_FLAG_HPP
#define STOP_FLAG_HPP

namespace nnet
{

class stop_flag
{
public:
	void set (void);

	bool is_set (void) const;

	std::condition_variable& get_condition (void) { return cond_; }

private:
	std::atomic<bool> flag_;

	std::condition_variable cond_;
};

struct thread_interrupted {};

extern thread_local stop_flag thread_stop_flag;

void stop_throw (void); // exit thread by throwing

void stop_wait (void); // wait until stop_flag is set

}

#endif /* STOP_FLAG_HPP */
