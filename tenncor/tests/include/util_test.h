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


// make partial full, but incompatible to comp
tensorshape make_full_incomp (std::vector<size_t> partial, std::vector<size_t> complete);


tensorshape padd(std::vector<size_t> shapelist, size_t nfront, size_t nback);


std::vector<std::vector<double> > doubleDArr(std::vector<double> v, std::vector<size_t> dimensions);
