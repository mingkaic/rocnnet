//
//  comm_record.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-01-30.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#include <unordered_set>
#include <fstream>
#include "edgeinfo/ptrinfo.hpp"
#include "graph/ccoms/subject.hpp"
#include "graph/ivariable.hpp"
#include "graph/iconnector.hpp"

#pragma once
#ifndef comm_record_hpp
#define comm_record_hpp

namespace rocnnet_record
{

class edge_record
{
	public:
		edge_record (std::string fname) : outname_(fname) {}

		void edge_capture (
				ccoms::iobserver* observer,
				ccoms::subject_owner* subject,
				size_t idx);

		void edge_release (
				ccoms::iobserver* observer,
				ccoms::subject_owner* subject,
				size_t idx);

		template <typename T>
		void to_csv (void)
		{
			std::ofstream ofile;
			ofile.open(outname_);
			if (ofile.is_open())
			{
				for (subinfo e : edges_)
				{
					nnet::iconnector<T>* ob = dynamic_cast<nnet::iconnector <T> *>(e.obs_);
				 	nnet::ivariable<T>* sb = dynamic_cast<nnet::ivariable <T> *>(e.sub_);

				 	// TODO: check ob and sb has_value

					ofile << ob->get_name() << ","
						  << *(edge_record::prec_.get_hash(e.obs_)) << ","
						  << sb->get_name() << ","
						  << *(edge_record::prec_.get_hash(e.sub_)) << ","
						  << e.sid_ << "\n";
				}
			}
			ofile.close();
		}

	private:
		static ptr_record prec_;

		struct subinfo
		{
			subinfo (ccoms::iobserver* obs,
					 ccoms::subject_owner* sub,
					 size_t sid) : obs_(obs), sub_(sub), sid_(sid) {}

			// we want to ensure:
			// 1. pair<obs_, sid_> uniquely maps to sub_
			// 2. sub_ uniquely maps to pair<obs_, sid_>
			ccoms::iobserver* obs_;
			ccoms::subject_owner* sub_;
			size_t sid_;
		};

		struct subinfo_hash {
			size_t operator () (const subinfo& info) const
			{
				std::experimental::optional<size_t> oid = edge_record::prec_.get_hash(info.obs_);
				std::experimental::optional<size_t> sid = edge_record::prec_.get_hash(info.sub_);

				// TODO: check if ids has_value for non-experimental optional

				size_t ptrmax = edge_record::prec_.get_max_id();
				return info.sid_ * ptrmax * ptrmax + *oid * ptrmax + *sid;
			}
		};

		friend bool operator == (edge_record::subinfo const& lhs, edge_record::subinfo const& rhs);

		std::string outname_;
		std::unordered_set<subinfo, subinfo_hash> edges_;
};

#ifdef EDGE_RCD
struct erec
{
	static edge_record rec;
};
#endif

}

#endif /* comm_record_hpp */
