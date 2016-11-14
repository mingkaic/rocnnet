
#include "gtest/gtest.h"
#include "graph/ccoms/iobserver.hpp"
#include "graph/ccoms/subject.hpp"

// test classes

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

TEST(COMS, observer) {
	div_subject subj;
	div_observer div_obs1(&subj, 4);
	div_observer div_obs2(&subj, 3);
	mod_observer mod_obs3(&subj, 3);
	subj.set_val(14);
	ASSERT_EQ(14/4, div_obs1.get_out());
	ASSERT_EQ(14/3, div_obs2.get_out());
	ASSERT_EQ(14%3, mod_obs3.get_out());
}
