%module dqn

%{
#define SWIG_FILE_WITH_INIT
#include "dqn_agent.hpp"
%}

%include "std_vector.i"
%include "std_string.i"

%template(IntVector) std::vector<unsigned int>;
%template(DoubleVector) std::vector<double>;

%include "dqn_agent.hpp"
