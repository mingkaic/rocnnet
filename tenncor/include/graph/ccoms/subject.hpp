//
//  subject.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2016-11-08
//  Copyright © 2016 Mingkai Chen. All rights reserved.
//

#include <experimental/optional>
#include "ireactive_node.hpp"

#pragma once
#ifndef subject_hpp
#define subject_hpp

namespace ccoms
{

class iobserver;
class subject;
class subject_owner;

struct update_message
{
	// preferrably we should move REVERSE_MODE to caller_info,
	// since REVERSE_MODE is deactivated once the gradient node is grabbed
	// but caller_info isn't accessible from gradient,
	// this should be reset to null after every update to prevent propogating up the graph (TODO: consider moving to caller)

	enum COMMAND
	{
		REVERSE_MODE, // todo use reverse mode instead of grad_
		REMOVE_ARG
	};
	std::experimental::optional<COMMAND> cmd_; // no cmd_ means forward mode
};

struct caller_info
{
	subject* caller_;
	size_t caller_idx_ = 0;
	caller_info (subject* caller = nullptr) : caller_(caller) {}
};

// subject retains control over all its observers,
// once destroyed, all observers are flagged for deletion

class subject : public ireactive_node
{
	private:
		// content (does not own... meaning memory leak/corruption if we delete this without variable; will occur for constant)
		subject_owner* var_ = nullptr;
		// dependents
		std::unordered_map<iobserver*, std::vector<size_t> > audience_;

	protected:
		// must explicitly destroy using delete
		virtual bool suicidal (void) { return false; }
		void attach (iobserver* viewer, size_t idx);
		// if suicidal, safe_destroy when detaching last audience
		virtual void detach (iobserver* viewer);

		friend class iobserver;
		
	public:
		subject (subject_owner* owner);
		virtual ~subject (void);
		// prevent audience from being copied over
		subject (const subject& other, subject_owner* owner);
	
		void notify (update_message msg = update_message());
		bool no_audience (void) const;
		// override ireactive_node's safe_destroy to kill var_ should subject become suicidal
		// killing var_ is essentially the same as suicide, since var_ will in turn kill this
		virtual bool safe_destroy (void);
		subject_owner*& get_owner (void);
};

}

#endif /* subject_hpp */
