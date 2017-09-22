//
//  comm_record.hpp
//  cnnet
//
//  Created by Mingkai Chen on 2017-01-30.
//  Copyright © 2017 Mingkai Chen. All rights reserved.
//

#include "graph/react/subject.hpp"
#include "graph/inode.hpp"
#include "graph/connector/iconnector.hpp"

#pragma once
#ifdef EDGE_RCD

#ifndef comm_record_hpp
#define comm_record_hpp

#include <unordered_set>
#include <fstream>

namespace rocnnet_record
{

class edge_record
{
public:
	edge_record (std::string fname) : outname_(fname) {}

	~edge_record (void);

	void edge_capture (nnet::iobserver* obs, nnet::subject* sub, size_t idx);

	void edge_release (nnet::iobserver* obs, nnet::subject* sub, size_t idx);

	void node_release (nnet::subject* sub);

	void node_release (nnet::iobserver* obs);

	template <typename T>
	void to_csv (const nnet::iconnector<T>* consider_graph = nullptr)
	{
		std::ofstream ofile;
		ofile.open(outname_);
		size_t num_des = 0;
		std::unordered_map<nnet::inode<T>*, size_t> num_corres;
		if (ofile.is_open())
		{
			for (auto ob2sub : edges_)
			{
				nnet::iconnector<T>* ob = dynamic_cast<nnet::iconnector<T>*>(ob2sub.first);
				if (nullptr == ob || (consider_graph && !ob->is_same_graph(consider_graph)))
				{
					continue; // skip non-connectors or connectors that aren't in the considered graph
				}
				std::vector<nnet::subject*>& sbs = ob2sub.second;
				for (size_t i = 0, n = sbs.size(); i < n; i++)
				{
					nnet::inode<T>* sb = static_cast<nnet::inode<T>*>(sbs[i]);

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
						nnet::tensorshape oshape;
						nnet::tensorshape sshape;
						// shapes can throw exceptions
						try
						{
							oshape = ob->get_shape();
							sshape = sb->get_shape();
						}
						catch(std::exception& e)
						{
							break;
						}

						obstrm << "(";
						sbstrm << "(";
						print_shape(oshape, obstrm);
						print_shape(sshape, sbstrm);
						obstrm << ")";
						sbstrm << ")";
					}

					ofile << obstrm.str() << "," << sbstrm.str() << "," << i << "\n";
				}
			}
		}
		ofile.close();
	}

	bool verbose_ = false;
	bool display_shape_ = true;

private:
	std::unordered_map<nnet::iobserver*,std::vector<nnet::subject*>> edges_; // grounded truth
	std::unordered_map<nnet::subject*,std::unordered_set<nnet::iobserver*>> subset_; // helper for subjects
	std::string outname_;
};

struct erec
{
	static edge_record rec;
	static bool rec_good;
};

}

#endif /* comm_record_hpp */

#endif /* EDGE_RCD */
