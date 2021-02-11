#
#   AI implementation of handwritten digit recognition written in C with GSL_BLAS.
# 
#   Author: Aleksander Szpakiewicz-Szatan
# 
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.
# 
#   Contact: aleksander.szpakiewicz-szatan.dokt(a)pw.edu.pl
#

SRCDIR=./src/
INCLUDEDIR=./include/
OBJDIR=./obj/
BINDIR=./bin/

CXX=gcc

CFLAGS=-pedantic -Wall -std=gnu99 -O3 -I$(INCLUDEDIR) -DHAVE_INLINE
LIBS=-lm -lgsl  -lgslcblas

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

