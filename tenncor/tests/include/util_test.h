//
// Created by Mingkai Chen on 2017-03-09.
//

#include <vector>
#include <iostream>

#include "tensor/tensorshape.hpp"


using namespace nnet;


bool tensorshape_equal (
	const tensorshape& ts1,
	const tensorshape& ts2);


bool tensorshape_equal (
	const tensorshape& ts1,
	std::vector<size_t>& ts2);


void print (std::vector<double> raw);


tensorshape make_partial (std::vector<size_t> shapelist);


tensorshape make_incompatible (std::vector<size_t> shapelist);


tensorshape padd(std::vector<size_t> shapelist, size_t nfront, size_t nback);


// for trouble shooting
//void print_tensor (nnet::tensor<double>* t)
//{
//	std::vector<size_t> s = t->get_shape().as_list();
//	size_t line_len = s.front();
//	size_t last = line_len;
//	char delim = '@';
//	// store the total number of elements to pass before entering passing an index
//	// and a unique symbol into tots
//	std::vector<std::pair<size_t, char> > tots;
//	for (auto it = ++s.begin(); it != s.end(); it++)
//	{
//		last *= *it;
//		tots.push_back(std::pair<size_t, char>(last, delim++));
//	}
//
//	std::vector<double> raw = nnet::expose<double>(t);
//	for (size_t i = 0; i < raw.size(); i++)
//	{
//		std::cout << raw[i] << " ";
//		// print a line of raws
//		if (0 == i % line_len)
//		{
//			std::cout << "\n";
//			// determine if we've incremented an index
//			for (size_t j = 0; j < tots.size(); j++)
//			{
//				size_t k = tots[j].first;
//				if (0 == i % k)
//				{
//					// for every new index, print a line of its symbols
//					char sym = tots[j].second;
//					for (size_t l = 0; l < line_len; l++)
//					{
//						std::cout << sym;
//					}
//					std::cout << "\n";
//				}
//			}
//		}
//	}
//}
