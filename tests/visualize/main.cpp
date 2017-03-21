//
// Created by Mingkai Chen on 2017-01-30.
//

#include "edgeinfo/comm_record.hpp"

#include "graph/varptr.hpp"
#include "executor/gradient.hpp"
#include "graph/functions.hpp"
#include "graph/operation/immutable/matmul.hpp"

using namespace nnet;

int main (int argc, char* argv[])
{
	random_uniform<double> rinit(-1, 1);
	session& sess = session::get_instance();

	placeptr<double> vin = new placeholder<double>(std::vector<size_t>{1, 10}, "vin");
	varptr<double> W1 = new variable<double>(std::vector<size_t>{10, 9}, rinit, "w1");
	varptr<double> B1 = new variable<double>(std::vector<size_t>{1, 9}, rinit, "b1");
	varptr<double> H1 = varptr<double>(matmul<double>::build(vin, W1)) + B1;
	varptr<double> vin2 = sigmoid<double>(H1);

	varptr<double> W2 = new variable<double>(std::vector<size_t>{9, 8}, rinit, "w2");
	varptr<double> B2 = new variable<double>(std::vector<size_t>{1, 8}, rinit, "b2");
	varptr<double> H2 = varptr<double>(matmul<double>::build(vin2, W2)) + B2;
	varptr<double> vout = sigmoid<double>(H2);

	sess.initialize_all<double>();
	vin = std::vector<double>(10, 1.2);

	gradient<double> grad(vout);
	// prevent changes to B from cascading to gradient value
	grad.freeze();
	grad.execute();

	// forward accumulation
	tensor<double>* result = vout->get_eval();
	// reverse accumulation
	tensor<double>* grad_result;
	grad.collect_grad(
		[&grad_result](inode<double>* key,
					   placeholder<double>* value)
		{
			grad_result = value->get_eval();
		});

#ifdef EDGE_RCD
rocnnet_record::erec::rec.to_csv<double>();
#endif

	delete vin.get();
	delete W1.get();
	delete B1.get();
	delete W2.get();
	delete B2.get();
	return 0;
}