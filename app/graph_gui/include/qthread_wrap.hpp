//
// Created by mingkaichen on 2017-11-15.
//

#include <experimental/tuple>

#include <QThread>

#pragma once
#ifndef QTHREAD_WRAP_HPP
#define QTHREAD_WRAP_HPP

namespace tenncor_graph
{

class qthread_wrap : public QThread
{
Q_OBJECT
public:
	template <typename FUNC, typename... ARGS>
	explicit qthread_wrap (FUNC&& f, ARGS&&... args)
	{
		runnable_ = [f = std::forward<FUNC>(f),
			args = std::make_tuple(std::forward<ARGS>(args)...)]
		{
			std::experimental::apply(f, args);
		};
	}

	virtual ~qthread_wrap (void) {}

private:
	void run (void);

	std::function<void(void)> runnable_;
};

}

#endif /* QTHREAD_WRAP_HPP */
