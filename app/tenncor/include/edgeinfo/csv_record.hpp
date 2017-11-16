//
//  csv_record.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-01-30.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#include "edgeinfo/adjlist_record.hpp"
#include "graph/connector/iconnector.hpp"

#pragma once
#ifdef EDGE_RCD

#include <unordered_set>
#include <fstream>

#ifndef CSV_RECORD_HPP
#define CSV_RECORD_HPP

namespace rocnnet_record
{

class csv_record final : public adjlist_record
{
public:
	csv_record (std::string fname);

	template <typename T>
	void to_csv (const nnet::iconnector<T>* consider_graph = nullptr) const
	{
		std::ofstream ofile;
		ofile.open(outname_);
		size_t num_des = 0;
		std::unordered_map<const nnet::inode<T>*, size_t> num_corres;
		if (ofile.is_open())
		{
			for (auto sub2obs : this->subj_nodes)
			{
				const nnet::subject* sbs = sub2obs.first;
				const nnet::inode<T>* sub = dynamic_cast<const nnet::inode<T>*>(sbs);
				if (nullptr == sub)
				{
					continue; // skip if not inode of type T
				}
				auto& obs_infos = sub2obs.second;
				for (auto info : obs_infos)
				{
					const nnet::iconnector<T>* obs = static_cast<const nnet::iconnector<T>*>(info.obs_);
					if (consider_graph && !obs->is_same_graph(consider_graph))
					{
						continue;
					}

					std::stringstream obstrm;
					std::stringstream sbstrm;

					if (verbose_)
					{
						obstrm << obs->get_name();
						sbstrm << sub->get_name();
					}
					else
					{
						auto oit = num_corres.find(obs);
						auto sit = num_corres.find(sub);

						size_t obidx;
						if (num_corres.end() == oit)
						{
							num_corres[obs] = obidx = num_des++;
						}
						else
						{
							obidx = oit->second;
						}
						size_t sbidx;
						if (num_corres.end() == sit)
						{
							num_corres[sub] = sbidx = num_des++;
						}
						else
						{
							sbidx = sit->second;
						}

						obstrm << '[' << obidx << ']' << obs->get_label();
						sbstrm << '[' << sbidx << ']' << sub->get_label();
					}
					if (display_shape_)
					{
						obstrm << "(";
						sbstrm << "(";
						print_shape(obs->get_shape(), obstrm);
						print_shape(sub->get_shape(), sbstrm);
						obstrm << ")";
						sbstrm << ")";
					}

					ofile << obstrm.str() << "," << sbstrm.str() << "," << info.idx_ << "\n";
				}
			}
		}
		ofile.close();
	}

	void setVerbose (bool verbosity);

	void setDisplayShape (bool display);

private:
	bool verbose_ = false;

	bool display_shape_ = true;

	std::string outname_;
};

}

#endif /* CSV_RECORD_HPP */

#endif /* EDGE_RCD */
