//
// Created by Mingkai Chen on 2016-11-15.
//

#ifndef ROCNNET_MOCK_CCOMS_H
#define ROCNNET_MOCK_CCOMS_H

#include "gmock/gmock.h"
#include "graph/ccoms/iobserver.hpp"

class mock_subject : public ccoms::subject
{
	public:
		mock_subject (void) {}
		~mock_subject (void) {}

		MOCK_METHOD1(merge_leaves, void(std::unordered_set<ccoms::subject*>&));
		MOCK_METHOD1(attach, void(ccoms::iobserver*));
		MOCK_METHOD1(detach, void(ccoms::iobserver*));
		MOCK_METHOD1(notify, void(ccoms::subject*));
		MOCK_METHOD0(notify, void());
};

class mock_observer : public ccoms::iobserver
{
	public:
		mock_observer (ccoms::subject* sub) : 
			ccoms::iobserver(std::vector<ccoms::subject*>{sub}) {}
		mock_observer (ccoms::subject* sub1, ccoms::subject* sub2) :
 			ccoms::iobserver(std::vector<ccoms::subject*>{sub1, sub2}) {}
		~mock_observer (void) {}

		MOCK_METHOD0(safe_destroy, bool(void));
		MOCK_METHOD1(merge_leaves, void(std::unordered_set<ccoms::subject*>&));
		MOCK_METHOD1(leaves_collect, void(std::function<void(ccoms::subject*)>));
		MOCK_METHOD1(update, void(ccoms::subject*));
};

//class mock_intern : virtual public ccoms::iobserver, virtual public ccoms::subject
//{
//	public:
//		mock_intern (ccoms::subject* sub) :
//				ccoms::iobserver(std::vector<ccoms::subject*>{sub}) {}
//		mock_intern (ccoms::subject* sub1, ccoms::subject* sub2) :
//			ccoms::iobserver(std::vector<ccoms::subject*>{sub1, sub2}) {}
//		~mock_intern (void) {}
//
//		MOCK_METHOD1(merge_leaves, void(std::unordered_set<ccoms::subject*>&));
//		MOCK_METHOD1(leaves_collect, void(std::function<void(ccoms::subject*)>));
//		MOCK_METHOD1(update, void(ccoms::subject*));
//};

class div_subject : public ccoms::subject {
private:
	size_t value_;

public:
	div_subject (void) : value_(0) {}

	size_t get_val (void) {
		return value_;
	}

	void set_val (size_t value) {
		value_ = value;
		this->notify();
	}
};

class div_observer: public ccoms::iobserver {
private:
	size_t div_;
	size_t out_;

public:
	div_observer (div_subject* mod, int div) :
			ccoms::iobserver({mod}), div_(div) {}

	size_t get_out (void) { return out_; }

	void update (ccoms::subject* caller) {
		div_subject* sub = dynamic_cast<div_subject*>(this->dependencies_[0]);
		assert(sub);
		size_t v = sub->get_val();
		out_ = v / div_;
	}
};

class mod_observer: public ccoms::iobserver {
private:
	size_t div_;
	size_t out_;

public:
	mod_observer (div_subject* mod, int div) :
			ccoms::iobserver({mod}), div_(div) {}

	size_t get_out (void) { return out_; }

	void update (ccoms::subject* caller) {
		div_subject* sub = dynamic_cast<div_subject*>(this->dependencies_[0]);
		assert(sub);
		size_t v = sub->get_val();
		out_ = v % div_;
	}
};

#endif //ROCNNET_MOCK_CCOMS_H
