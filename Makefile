SRCDIR=./src/
INCLUDEDIR=./include/
OBJDIR=./obj/
BINDIR=./bin/

CXX=gcc

CFLAGS=-pedantic -Wall -std=gnu99 -O3 -I$(INCLUDEDIR) 
LIBS=-lm -lgsl -lgslcblas

DEPS = input.h outout.h network.h
SRCS = main.c input.c output.c network.c

OBJS := $(addprefix $(OBJDIR),$(SRCS:.c=.o))
SRCS := $(addprefix $(SRCDIR),$(SRCS))
DEPS := $(addprefix $(INCLUDEDIR),$(DEP))
	
ai: $(OBJS) 
	$(CXX) $(CFLAGS) $(LIBS) -o $(BINDIR)$@ $^  

$(OBJDIR)%.o: $(SRCDIR)%.c $(DEPS) dirs
	$(CXX) $(CFLAGS) -c -o $@ $< 

dirs:
	mkdir -p $(OBJDIR)
	mkdir -p $(BINDIR)

.PHONY: clean all ver


all: aiC
clean:
	rm -f $(OBJDIR)*
ver:
	$(CXX) --version

