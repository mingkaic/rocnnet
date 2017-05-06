%module dqn

%{
#define SWIG_FILE_WITH_INIT
#include "dqn_agent.hpp"
%}

%include "std_vector.i"
%include "std_string.i"
namespace std {
   %template(IntVector) vector<unsigned int>;
   %template(DoubleVector) vector<double>;
};

%include "dqn_agent.hpp"
