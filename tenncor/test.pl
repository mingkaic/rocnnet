#!/usr/bin/perl
# this is basically a make file, except it's cross platform
# and I don't want to bother with cmake for llvm/clang integration
use strict;
use warnings;
use File::Find::Rule;
use File::Basename;
use Cwd;

my $CLANG = "/Users/cmk/llvm/bin/clang++";
my $OUTLL = "./ll";
my $OUTDIR = "./bin";

my $CP_PATH = "/Users/cmk/Developer/cworkspace/general/coverage-profiler/cmake-build-default/bin/coverageprofiler";
my $LIBNAME = "libtenncor-inst";

my $cwd = getcwd();

sub buildFileIndex {
    # File find rule
    my $excludeDirs = File::Find::Rule->directory
        ->name('ext', 'tests', 'cmake-build-debug') # Provide specific list of directories to *not* scan
        ->prune                                     # don't go into it
        ->discard;                                  # don't report it

    my $includeFiles = File::Find::Rule->file
        ->name('*.cpp');                            # search by file extensions

    return File::Find::Rule->or( $excludeDirs, $includeFiles )->in($cwd);
}

sub llvm {
    my @files = buildFileIndex();

    foreach my $farg (@files) {
        my $base = basename($farg, ".cpp");
        my $fout = "$OUTLL/$base.ll";
        system("$CLANG -std=c++14 -Iinclude -g -emit-llvm -S $farg -o $fout");
    }
}

sub bin {
    opendir(DIR, $OUTLL) or die $!;

    my $args = "";
    while (my $file = readdir(DIR)) {
        next if ($file =~ m/^\./);
        $args = "$args $OUTLL/$file";
    }

    closedir(DIR);
    system("$CP_PATH $args -o $OUTDIR/$LIBNAME");
}

my $num_args = $#ARGV + 1;
my $DIRECTIVE = "all";
if ($num_args > 0) {
    $DIRECTIVE = $ARGV[0];
}

if ($DIRECTIVE eq "all") {
    &llvm;
    &bin;
    print $OUTDIR/$LIBNAME;
}
elsif ($DIRECTIVE eq "llvm") {
    &llvm;
}
elsif ($DIRECTIVE eq "bin") {
    &bin;
}
elsif ($DIRECTIVE eq "clean") {
    unlink glob "$OUTLL/*.*";
}
else {
    print "usage: ['all', 'llvm', 'bin', 'clean']"
}