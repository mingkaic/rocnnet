//
//  Created by Ming Kai Chen on 2017-11-12.
//

#ifdef SHAREABLE_HPP

namespace nnet
{

template <typename T>
cache_node<T>::cache_node (std::shared_ptr<cache_node<T> > prev, T data) :
	prev_(prev), data_(data) {}

template <typename K, typename T>
class shareable_cache<K, T>::cache_list
{
public:
	std::shared_ptr<cache_node<T> > begin (void) const
	{
		return head_;
	}

	std::shared_ptr<cache_node<T> > push_back (T data)
	{
		if (nullptr == tail_) // && nullptr == head_->next_
		{
			tail_ = head_ = std::make_shared<cache_node<T> >(nullptr, data);
		}
		else
		{
			tail_ = tail_->next_ = std::make_shared<cache_node<T> >(tail_, data);
		}
		return tail_;
	}

	// remove it node from the link
	// invariant: broken it chain always ends in an unbroken node or null
	// after: if it == nullptr, then everything starting from head_ is unread
	void unlink (std::shared_ptr<cache_node<T> > it)
	{
		if (nullptr != it)
		{
			bool is_tail = tail_ == it;
			if (head_ == it)
			{
				head_ = it->next_;
			}
			if (is_tail)
			{
				tail_ = it->prev_;
			}
			if (nullptr != it->prev_)
			{
				it->prev_->next_ = it->next_;
			}
			if (nullptr != it->next_)
			{
				it->next_->prev_ = it->prev_;
			}
			it->prev_ = nullptr;
			if (is_tail)
			{
				// if broken node was a tail, link to the closest unbroken (new tail)
				it->next_ = tail_;
			}
			// invariant: if (tail_ == nullptr) then head_ == nullptr && it == nullptr
		}
	}

	// sometimes, nodes can be broken (if it->prev_ is null and is not head)
	// update input it shared_ptr to reference the next unbroken node
	// return true if it was broken
	bool grab_truth (std::shared_ptr<cache_node<T> >& it) const
	{
		if (head_ == it || nullptr == it) return false;
		bool broken = nullptr == it->prev_;
		while (nullptr == it->prev_)
		{
			it = it->next_;
		}
		return broken;
	}

private:
	std::shared_ptr<cache_node<T> > head_ = nullptr;

	std::shared_ptr<cache_node<T> > tail_ = nullptr; // time saving for pushback
};

template <typename K, typename T>
shareable_cache<K, T>::shareable_cache (std::function<K(const T&)> hasher) :
	hasher_(hasher) {}

template <typename K, typename T>
optional<T> shareable_cache<K, T>::get_latest (std::shared_ptr<cache_node<T> >& iter) const
{
	std::unique_lock<std::mutex> locker(mutex_);

	optional<T> value;
	if (nullptr == iter)
	{
		// check head of content_
		if (iter = content_.begin())
		{
			value = iter->data_;
		}
	}
	// ensure iter is not broken
	else if (content_.grab_truth(iter) &&
		nullptr != iter)
	{
		// iter was broken, we grab data from iter instead,
		// since we've never visited it before
		value = iter->data_;
		// we don't increment since we just read iter
	}
	else if (iter->next_)
	{
		value = iter->next_->data_;
		iter = iter->next_;
	}
	if (!value)
	{
		cond_.wait(locker); // no current wait to check for spurious wait
	}
	return value;
}

template <typename K, typename T>
void shareable_cache<K, T>::add_latest (T data)
{
	std::unique_lock<std::mutex> locker(mutex_);

	K key = hasher_(data);
	auto iter = umap_.find(key);
	if (umap_.end() != iter)
	{
		content_.unlink(iter->second);
	}
	umap_[key] = content_.push_back(data);
	cond_.notify_one();
}

template <typename K, typename T>
void shareable_cache<K, T>::escape_wait (void)
{
	cond_.notify_one();
}

}

#endif
