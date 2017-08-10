# Time-stamp: <makefile 2017-06-01 08:24:19 Hidenori Kuwakado>

prog=matrixGenerator
srcs=$(wildcard *.c)
objdir=Obj.$(shell "$(hostname)" -s)
makefile=makefile

cc=/usr/bin/gcc
ccflags=$(ccgramaflags) $(ccwarningflags) $(ccoptflags)
ccgramaflags=-std=gnu99
ccoptflags=-O3
ccwarningflags=-Wall
dflags=
inc=-I. -I..
lib=-L. -lgmp

cmp=/usr/bin/cmp
cp=/bin/cp
echo=/bin/echo
hostname=/bin/hostname
ln=/bin/ln
mkdir=/bin/mkdir
rm=/bin/rm
expr=/usr/bin/expr

###
$(prog): $(objdir)/main.o
	$(cc) $^ -o $(objdir)/$@ $(ccflags) $(inc) $(lib) $(dflags)
	$(ln) --force --symbolic $(objdir)/$@ $@

$(objdir)/main.o: main.c
	@if [ ! -d $(objdir) ]; then $(mkdir) $(objdir); fi
	$(cc) $< -c -o $@ $(ccflags) $(inc) $(dflags)

###
generate: $(prog)
	./$(prog) --numRows 128 --numColumns $(numColumns) --matrixFile matrix.128x$(numColumns).h --name mat_h 

.PHONY: generateAll
generateAll:
	$(MAKE) --makefile=$(makefile) numColumns=$(shell $(expr) 1024 \*  4) generate
	$(MAKE) --makefile=$(makefile) numColumns=$(shell $(expr) 1024 \*  8) generate
	$(MAKE) --makefile=$(makefile) numColumns=$(shell $(expr) 1024 \* 16) generate
	$(MAKE) --makefile=$(makefile) numColumns=$(shell $(expr) 1024 \* 32) generate
	$(cp) --force matrix.*.h ..

###
.PHONY: clean
clean:
	-$(rm) --force --recursive $(prog) $(objdir) matrix.*.h

.PHONY: rebuild
rebuild:
	$(MAKE) --always-make --makefile=$(makefile) clean
	$(MAKE) --always-make --makefile=$(makefile) $(prog)

# end of file
