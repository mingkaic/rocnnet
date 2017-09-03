//
// Created by Mingkai Chen on 2017-09-02.
//

#include "embedding.hpp"

#ifdef ROCBOOST_EMBEDDING_HPP

// relative path from bin/bin to python
const bp::str pickle_script = "../../boostpy/python/pickle.py";

bp::tuple pickle (std::string data_path)
{
	// Retrieve the main module.
	bp::object main = bp::import("__main__");

	// Retrieve the main module's namespace
	bp::object global(main.attr("__dict__"));

	main.attr("__file__") = pickle_script;

	// Define greet function in Python.
	bp::object result = bp::exec_file(pickle_script, global, global);

	// Create a reference to it.
	bp::object pickle = global["load_pickle"];

	// Call it.
	bp::object pickle_result = pickle(data_path);

	bp::tuple data = bp::extract<bp::tuple>(pickle_result);

	return data;
}

#endif
