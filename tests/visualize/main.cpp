//
// Created by Mingkai Chen on 2017-01-30.
//

#include "edgeinfo/comm_record.hpp"

#include "executor/varptr.hpp"
#include "executor/gradient.hpp"
#include "graph/functions.hpp"
#include "graph/operation/matmul.hpp"

using namespace nnet;

int main (int argc, char* argv[])
{
	tensorshape common = std::vector<size_t>{5, 5};
	random_uniform<double> rinit(-1, 1);
	session& sess = session::get_instance();

	// initializes a 5 by 5 matrix with uniformly distributed
	// doubles between -1 and 1
	varptr<double> A = new variable<double>(common, rinit, "a");
	placeptr<double> B = new placeholder<double>(common, "b");
	varptr<double> C = matmul<double>::build(A, B);
	varptr<double> D = sigmoid<double>(C);

	sess.initialize_all<double>();
	B = std::vector<double>(25, 2);

	gradient<double> grad(D);
	// prevent changes to B from cascading to gradient value
	grad.freeze();
	grad.execute();

	// forward accumulation
	tensor<double>* result = D->get_eval();
	// reverse accumulation
	tensor<double>* grad_result;
	grad.collect_grad(
		[&grad_result](ivariable<double>* key,
					   placeholder<double>* value)
		{
			grad_result = value->get_eval();
		});

#ifdef EDGE_RCD
rocnnet_record::erec::rec.to_csv<double>();
#endif

	delete A.get();
	delete B.get();
	return 0;
}