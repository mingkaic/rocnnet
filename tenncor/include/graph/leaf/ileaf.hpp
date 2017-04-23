/*!
 *
 *  ileaf.hpp
 *  cnnet
 *
 *  Purpose:
 *  leaf interface abstractly defines
 *  all pure subject nodes in the graph
 *
 *  Created by Mingkai Chen on 2016-08-29.
 *  Copyright © 2016 Mingkai Chen. All rights reserved.
 *
 */

#include "graph/inode.hpp"

#pragma once
#ifndef TENNCOR_ILEAF_HPP
#define TENNCOR_ILEAF_HPP

namespace nnet
{

template <typename T>
class ileaf : public inode<T>
{
public:
	//! virtual destructor for constant and ivariable
	virtual ~ileaf (void) {}

	// >>>> CLONE, COPY && MOVE ASSIGNMENTS <<<<
	//! clone function
	ileaf<T>* clone (void) const;

	//! move function
	ileaf<T>* move (void);

	//! declare copy assignment to deep copy over data
	virtual ileaf<T>& operator = (const ileaf<T>& other);

	//! declare move assignment to move over data
	virtual ileaf<T>& operator = (ileaf<T>&& other);

	// >>>> ACCESSORS <<<<
	//! utility function: get data shape
	virtual tensorshape get_shape (void) const;

	//! get forward passing value
	//! return nullptr if leaf is not init
	virtual const tensor<T>* get_eval (void) const;

	//! check if data is available
	//! (if the node is initialized)
	virtual bool good_status (void) const;

protected:
	// >>>> CONSTRUCTORS <<<<
	//! assign initializer
	ileaf (const tensorshape& shape, std::string name);

	// >>>> COPY && MOVE CONSTRUCTORS <<<<
	//! declare copy constructor to deep copy over data
	ileaf (const ileaf<T>& other);

	//! declare move constructor to move over data
	ileaf (ileaf<T>&& other);

	// >>>> TENSOR CONTENT <<<<
	//! tensor data
	std::unique_ptr<tensor<T> > data_ = nullptr;

	// >>>> INITIALIZER DATA <<<<
	//! tensor state (good or bad) true = good
	bool is_init_ = false;

	//! dynamically initialize tensors
	//! used by placeholder
	class assignment;

	template <typename U>
	friend void assign_add (ileaf<U>* orig, std::vector<U>& raw_data);

private:
	//! copy helper
	void copy_helper (const ileaf<T>& other);

	//! move helper
	void move_helper (ileaf<T>&& other);
};

template <typename T>
void assign_add (ileaf<T>* orig, std::vector<T>& raw_data)
{
	transfer_func<T> assign(
	[orig](std::vector<tensorshape>)
	{
		return orig->get_shape();
	},
	[&raw_data](T* dest, const tensorshape& shape, std::vector<const T*>&, std::vector<tensorshape>&)
	{
		size_t ns = shape.n_elems();
		memcpy(&raw_data[0], dest, sizeof(T) * ns);
	});
	assign(*orig->data_, {});
}

}

#include "../../../src/graph/leaf/ileaf.ipp"

#endif /* TENNCOR_ILEAF_HPP */
