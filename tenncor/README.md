# Building

Tenncor uses cmake (minimum 2.8). Compiling in a separate directory is advised.

    mkdir build
    cd build
    cmake <path/to/tenncor>
    make

# Testing

Set RunTest option to ON during cmake generation should generate a default `libtenncor-inst.a` in the `bin` directory

    cmake -DRunTest=ON <path/to/tenncor>
    
To see coverage, install or download and build [llvm](http://releases.llvm.org/download.html) and [coverage-profiler](https://github.com/mingkaic/coverage-profiler). 

Instrument by running `test.pl`. This script will use clang to generate llvm intermediate representation files in the `ll` directory, then instrument and link all the `ll/*.ll` files as `libtenncor-inst.a`.

    perl test.pl
    
Finally, relink `libtenncor-inst.a` to the test (manually for now), and run the test. Upon termination, coverage-profiler generates `coverage-profile.txt` in the `bin` directory which outlines the block coverage for each function.
    
Alternatively, feel free to use alternative coverage measuring methods.