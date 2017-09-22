//
// Created by Mingkai Chen on 2017-09-02.
//

#include "embedding.hpp"

#ifdef ROCBOOST_EMBEDDING_HPP

template<class T>
bp::list std_vector_to_py_list(const std::vector<T>& v)
{
	bp::list l;
	for(T e : v)
		l.append(e);
	return l;
}

// relative path from bin/bin to python
const bp::str pickle_script = "../../boostpy/python/pickle.py";

bp::tuple pickle (std::string data_path)
{
	// Retrieve the main module.
	bp::object main = bp::import("__main__");

	// Retrieve the main module's namespace
	bp::object global(main.attr("__dict__"));

	main.attr("__file__") = pickle_script;

	// Get pickle function in Python.
	bp::object result = bp::exec_file(pickle_script, global, global);

	// Create a reference to it.
	bp::object pickle = global["load_pickle"];

	// Call it.
	bp::object pickle_result = pickle(data_path);

	bp::tuple data = bp::extract<bp::tuple>(pickle_result);

	return data;
}

const bp::str imageout_script = "../../boostpy/python/imageout.py";

void mnist_imageout (std::vector<std::vector<double> >& imgdata,
	std::vector<size_t> inner_dims, size_t n_chains, size_t n_samples)
{
	bp::list imglist;
	for (std::vector<double>& imgvec : imgdata)
	{
		imglist.append(std_vector_to_py_list(imgvec));
	}

	// Retrieve the main module.
	bp::object main = bp::import("__main__");

	// Retrieve the main module's namespace
	bp::object global(main.attr("__dict__"));

	main.attr("__file__") = imageout_script;

	// Get imageout function in Python.
	bp::object result = bp::exec_file(imageout_script, global, global);

	// Create a reference to it.
	bp::object imgout = global["mnist_imageout"];

	// Call it.
	imgout(imglist, std_vector_to_py_list(inner_dims), n_chains, n_samples);
}

#endif
