## nnet makefile
CC = g++ -std=c++11
DEBUG = -g
TESTFLAG = -lgtest
DEPLOY_FOLDER = bin/
FILES = main.cpp src/*/*.cpp src/*/*/*.cpp

## python and threading
PYFLAG = -framework Python
THREADFLAG = -pthread

all:
	$(CC) $(FILES) -o $(DEPLOY_FOLDER)nnet.exe

## build all, test everything
test:
	$(CC) $(TESTFLAG) $(FILES) -o $(DEPLOY_FOLDER)test.exe -D TESTFLAG
	./$(DEPLOY_FOLDER)test.exe

## build all, test tensor/graph component only
test_ten:
	$(CC) $(TESTFLAG) $(FILES) -o $(DEPLOY_FOLDER)test_ten.exe -D TENSOR_TEST
	./$(DEPLOY_FOLDER)test_ten.exe

## build all, test neural net component only
test_net:
	$(CC) $(TESTFLAG) $(FILES) -o $(DEPLOY_FOLDER)test_net.exe -D LAYER_TEST
	./$(DEPLOY_FOLDER)test_net.exe

clean:
	rm $(DEPLOY_FOLDER)nnet
	rm $(DEPLOY_FOLDER)test
