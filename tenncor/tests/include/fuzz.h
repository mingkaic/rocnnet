//
// Created by Mingkai Chen on 2017-03-09.
//

#include <random>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cstdlib>
#include <limits>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <queue>
#include <functional>

#include "utils/utils.hpp"

#ifndef TENNCOR_FUZZ_H
#define TENNCOR_FUZZ_H

namespace FUZZ
{

static const char* FUZZ_FILE = "fuzz.out";
static std::ofstream fuzzLogger(FUZZ_FILE);

void reset_logger (void);

std::vector<double> getDouble (size_t len,
	std::string purpose = "getDouble",
	std::pair<double,double> range={0,0});

std::vector<size_t> getInt (size_t len,
	std::string purpose = "getInteger",
	std::pair<size_t,size_t> range={0,0});

std::string getString (size_t len,
	std::string purpose = "getString",
	std::string alphanum =
	"0123456789!@#$%^&*"
	"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	"abcdefghijklmnopqrstuvwxyz");

template <typename T>
typename T::iterator rand_select (T container)
{
	size_t size = container.size();
	auto it = container.begin();
	for (size_t i = 0, n = getInt(1, "rand_select::n", {0, size-1})[0]; i < n; i++)
	{
		it++;
	}
	return it;
}

// generate a binary tree structure in serial form
std::string getBTree ();

template <typename N>
N* buildNTree (size_t n, size_t nnodes,
	std::function<N*()> buildleaf,
	std::function<N*(std::vector<N*>)> connect)
{
	std::vector<size_t> nstrlen = FUZZ::getInt(nnodes, "nstrlen", {14, 29});

	std::vector<size_t> preds = {0, 0};
	std::unordered_set<size_t> leaves = {1};
	std::vector<size_t> nargs = {1, 0};
	{
		// randomly generate predecessor nodes
		std::default_random_engine generator = nnutils::get_generator();
		std::vector<size_t> concount = {0, 1};

		fuzzLogger << "tree<";
		for (size_t i = 2; i < nnodes; i++)
		{
			std::uniform_int_distribution<size_t> dis(0, concount.size()-1);
			size_t pidx = dis(generator);

			auto it = concount.begin() + pidx;
			size_t p = *it;
			// remove nodes that are not leaves
			leaves.erase(p);

			// update the potential connectors list
			if (++nargs[p] >= n)
			{
				concount.erase(it);
			}

			concount.push_back(i);
			leaves.insert(i);
			nargs.push_back(0);
			// update predecessors
			preds.push_back(p);
			fuzzLogger << p << "," << std::endl;
		}
		fuzzLogger << ">" << std::endl;
	}
	std::unordered_map<size_t,std::vector<N*> > deps;
	std::queue<size_t> pq;
	for (size_t l : leaves)
	{
		N* nn = buildleaf();
		// preds
		size_t pp = preds[l];
		deps[pp].push_back(nn);
		if (deps[pp].size() == nargs[pp])
		{
			pq.push(pp);
		}
	}

	N* root = nullptr;
	while (pq.empty() == false)
	{
		size_t nidx = pq.front();
		pq.pop();
		root = connect(deps[nidx]);
		if (nidx)
		{
			// preds
			size_t pidx = preds[nidx];
			deps[pidx].push_back(root);
			if (deps[pidx].size() == nargs[pidx])
			{
				pq.push(pidx);
			}
		}
	}

	return root;
}

}

#endif //TENNCOR_FUZZ_H
