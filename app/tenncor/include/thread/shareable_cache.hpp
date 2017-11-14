//
//  shareable_cache.hpp
//  cnnet
//
//  Created by Ming Kai Chen on 2017-11-12.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#include <mutex>
#include <memory>
#include <unordered_map>

#pragma once
#ifndef SHAREABLE_HPP
#define SHAREABLE_HPP

namespace nnet
{

// cache_nodes make use of shared pointers to prevent dangling
// pointers to obsolete nodes when cache updates
template <typename T>
struct cache_node
{
	cache_node (std::shared_ptr<cache_node<T> > prev, T data);

	std::shared_ptr<cache_node<T> > next_ = nullptr;
	std::shared_ptr<cache_node<T> > prev_;
	T data_;
};

// shared_cache considers 2 types of actors: consumers and producers
// this structure, expects there to be multiple consumers, but only 1 producer
// cache makes use of a key-value system to reduce duplicate/obsolete information
// key is unique, meaning when a new key-value pair is inserted,
// the old pair with the same key is obsolete
// because producers may reference some obsolete pairs, we handle obsolecse by shared_pointers to nodes
// obsolete nodes always link to a valid node or nullptr
// obsolete nodes have prev of null
// the producer determines a key-value relationship using hash function (hasher_)
// each consumer holds one shared_ptr to some node in the cache (content_)
// if the shared_ptr is null, the consumer is expected to consume from the start of the cache
// otherwise, the consumer consumes the immediate valid node referenced by the shared_ptr
template <typename K, typename T>
class shareable_cache
{
public:
	shareable_cache (std::function<K(const T&)> hasher);

	// assert that iter's data has been read
	// grab data from iter->next_ and increment iter if possible
	T get_latest (std::shared_ptr<cache_node<T> >& iter) const;

	// add data mapped by key remove previous iterators mapped by it if necessary
	void add_latest (T data);

private:
	class cache_list;

	std::function<K(const T&)> hasher_;

	std::unordered_map<K,std::shared_ptr<cache_node<T> > > umap_;

	cache_list content_;

	mutable std::mutex mutex_;

	mutable std::condition_variable cond_;
};

}

#include "../../src/thread/shareable_cache.ipp"

#endif /* SHAREABLE_HPP */
