

CXXFLAGS=-std=c++11 -O3 -g

LIBFLAGS=-pthread 
INC=-I../tools/c++ 


all: bonsai_train  bonsai_predict

bonsai_train: helpers.o bonsai_train.cpp bonsai.cpp bonsai_update.cpp tron.cpp
	$(CXX) -o bonsai_train $(CXXFLAGS) $(INC) bonsai_train.cpp bonsai.cpp bonsai_update.cpp tron.cpp helpers.o $(LIBFLAGS)

bonsai_predict: bonsai_predict.cpp bonsai.cpp helpers.o tron.cpp
	$(CXX) -o bonsai_predict $(CXXFLAGS) $(INC) bonsai_predict.cpp bonsai.cpp tron.cpp helpers.o $(LIBFLAGS)

helpers.o: helpers.cpp
	$(CXX) -c $(CXXFLAGS) $(INC) helpers.cpp $(LIBFLAGS)

clean:
	rm -f bonsai_train bonsai_predict *.o

