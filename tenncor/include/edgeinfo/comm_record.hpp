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
#include "graph/react/subject.hpp"
#include "graph/inode.hpp"
#include "graph/connector/iconnector.hpp"

#pragma once
#ifdef EDGE_RCD

#ifndef comm_record_hpp
#define comm_record_hpp

namespace rocnnet_record
{

class edge_record
{
	public:
		edge_record (std::string fname) : outname_(fname) {}

		void edge_capture (
				iobserver* observer,
				subject_owner* subject,
				size_t idx);

		void edge_release (
				iobserver* observer,
				subject_owner* subject,
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
				 	nnet::inode<T>* sb = dynamic_cast<nnet::inode <T> *>(e.sub_);

					std::string obname = ob->get_name();
					std::string sbname = sb->get_name();

					std::replace(obname.begin(), obname.end(), ',', '|');
					std::replace(sbname.begin(), sbname.end(), ',', '|');

					ofile << obname << ","
						  << sbname << ","
						  << e.sid_ << "\n";
				}
			}
			ofile.close();
		}

	private:
		static ptr_record prec_;

		struct subinfo
		{
			subinfo (iobserver* obs,
					 subject_owner* sub,
					 size_t sid) : obs_(obs), sub_(sub), sid_(sid) {}

			// we want to ensure:
			// 1. pair<obs_, sid_> uniquely maps to sub_
			// 2. sub_ uniquely maps to pair<obs_, sid_>
			iobserver* obs_;
			subject_owner* sub_;
			size_t sid_;
		};

		struct subinfo_hash {
			size_t operator () (const subinfo& info) const
			{
				size_t oid = edge_record::prec_.get_hash(info.obs_);
				size_t sid = edge_record::prec_.get_hash(info.sub_);

				// TODO: check if ids has_value for non-experimental optional

				size_t ptrmax = edge_record::prec_.get_max_id();
				return info.sid_ * ptrmax * ptrmax + oid * ptrmax + sid;
			}
		};

		friend bool operator == (edge_record::subinfo const& lhs, edge_record::subinfo const& rhs);

		std::string outname_;
		std::unordered_set<subinfo, subinfo_hash> edges_;
};

struct erec
{
	static edge_record rec;
};

}

#endif /* comm_record_hpp */

#endif /* EDGE_RCD */
