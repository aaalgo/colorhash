CXXFLAGS += -std=c++11 -fopenmp -g
LDFLAGS += -std=c++11 -fopenmp -g
LDLIBS += -lkgraph -lfmt $(shell pkg-config --libs opencv) -lboost_program_options -lboost_timer -lrt

all:	extract	rank
