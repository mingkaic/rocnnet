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

	void edge_capture (nnet::iobserver* obs, nnet::subject* sub, size_t idx);

	void edge_release (nnet::iobserver* obs, nnet::subject* sub, size_t idx);

	template <typename T>
	void to_csv (void)
	{
		std::ofstream ofile;
		ofile.open(outname_);
		size_t num_des = 0;
		std::unordered_map<nnet::inode<T>*, size_t> num_corres;
		if (ofile.is_open())
		{
			for (subinfo e : edges_)
			{
				nnet::iconnector<T>* ob = dynamic_cast<nnet::iconnector<T>*>(e.obs_);
				nnet::inode<T>* sb = dynamic_cast<nnet::inode<T>*>(e.sub_);

				std::stringstream obstrm;
				std::stringstream sbstrm;

				if (verbose_)
				{
					obstrm << ob->get_name();
					sbstrm << sb->get_name();
				}
				else
				{
					auto oit = num_corres.find(ob);
					auto sit = num_corres.find(sb);

					size_t obidx;
					if (num_corres.end() == oit)
					{
						num_corres[ob] = obidx = num_des++;
					}
					else
					{
						obidx = oit->second;
					}
					size_t sbidx;
					if (num_corres.end() == sit)
					{
						num_corres[sb] = sbidx = num_des++;
					}
					else
					{
						sbidx = sit->second;
					}

					obstrm << '[' << obidx << ']' << ob->get_label();
					sbstrm << '[' << sbidx << ']' << sb->get_label();
				}
				if (display_shape_)
				{
					obstrm << "(";
					sbstrm << "(";
					print_shape(ob->get_shape(), obstrm);
					print_shape(sb->get_shape(), sbstrm);
					obstrm << ")";
					sbstrm << ")";
				}

				ofile << obstrm.str() << "," << sbstrm.str() << "," << e.sid_ << "\n";
			}
		}
		ofile.close();
	}

	bool verbose_ = false;
	bool display_shape_ = true;

private:
	static ptr_record prec_;

	struct subinfo
	{
		subinfo (nnet::iobserver* obs, nnet::subject* sub, size_t sid) :
			obs_(obs), sub_(sub), sid_(sid) {}

		// we want to ensure:
		// 1. pair<obs_, sid_> uniquely maps to sub_
		// 2. sub_ uniquely maps to pair<obs_, sid_>
		nnet::iobserver* obs_;
		nnet::subject* sub_;
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
