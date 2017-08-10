# Time-stamp: <makefile 2017-08-07 09:31:43 Hidenori Kuwakado>

# NVIDIA GeForce GTX Titan X, Maxwell 3072 CUDA Cores
nvccArch=sm_52
cudaArch=520
nvccMachine=64

# For other GPU macors
## NVIDIA GTX 690, Kepler 1536 CUDA Cores
# nvccArch=sm_30
# cudaArch=300
# nvccMachine=64
## NVIDIA Tegra K1, Kepler 192 CUDA Cores
# nvccArch=sm_32
# cudaArch=320
# nvccMachine=32
## NVIDIA Tegra X1, Maxwell 256 CUDA Cores
# nvccArch=sm_53
# cudaArch=530
# nvccMachine=64
## NVIDIA GeForce GTX 960M, Maxwell 640 CUDA Cores
# nvccArch=sm_50
# cudaArch=500
# nvccMachine=64
## NVIDIA GeForce TITAN X, Pascal 3584 CUDA Cores
# nvccArch=sm_61
# cudaArch=610
# nvccMachine=64


#------------------------------------------------------------
# CUDA Compiler
nvcc=/usr/local/cuda/bin/nvcc
nvccFlags=--optimize 3 --x cu --machine $(nvccMachine) -arch=$(nvccArch) --define-macro NUM_COLUMNS=$(numColumns) --define-macro __CUDA_ARCH__=$(cudaArch) --compiler-options "$(nvccXcompiler)" --ptxas-options=-v
nvccLib=-lcrypto
nvccInc=
nvccNoWarning=-Wunused-function
nvccXcompiler=-O3 -Wall

# Source files
headers=$(wildcard *.h)
makefile=makefile
objdir=Obj.$(numColumns).$(hname)
objs=$(patsubst %.cu,$(objdir)/%.o,$(srcs))
prog=sha512mpp
srcs=$(wildcard *.cu)

# Etc
echo=/bin/echo
hname=$(shell /bin/hostname -s)
mkdir=/bin/mkdir
rm=/bin/rm


#------------------------------------------------------------
# 4096 etc. are the number of columns of the AGGH fucntion
.PHONY: all
all: 4096 8192 16384 32768

.PHONY: 4096 8192 16384 32768
4096:
	$(MAKE) --makefile=$(makefile) numColumns=$@ $(prog).$@.$(hname)
8192:
	$(MAKE) --makefile=$(makefile) numColumns=$@ $(prog).$@.$(hname)
16384:
	$(MAKE) --makefile=$(makefile) numColumns=$@ $(prog).$@.$(hname)
32768:
	$(MAKE) --makefile=$(makefile) numColumns=$@ $(prog).$@.$(hname)

.PHONY: clean
clean:
	-$(rm) --force --recursive $(prog).*.$(hname) Obj.*.$(hname)

.PHONY: build
build:
	$(MAKE) --makefile=$(makefile) clean
	$(MAKE) --makefile=$(makefile) all


#------------------------------------------------------------
ifdef numColumns
$(prog).$(numColumns).$(hname): $(objs)
	$(nvcc) -o $@ $(objs) $(nvccLib)

$(objdir)/%.o: %.cu
	@if [ ! -d $(objdir) ]; then $(mkdir) $(objdir); fi
	$(nvcc) $< -c -o $@  $(nvccFlags) $(nvccInc)
endif


#------------------------------------------------------------
.PHONY: testRun
testRun:
	@$(echo) "SHA-512:";        ./$(prog).4096.$(hname)  --lifetime 3 --numMiBytes 4 --numInputData 2 --sha512
	@$(echo) "SHA512mp 4096:";  ./$(prog).4096.$(hname)  --lifetime 3 --numMiBytes 4 --numInputData 2
	@$(echo) "SHA512mp 8192:";  ./$(prog).8192.$(hname)  --lifetime 3 --numMiBytes 4 --numInputData 2
	@$(echo) "SHA512mp 16384:"; ./$(prog).16384.$(hname) --lifetime 3 --numMiBytes 4 --numInputData 2
	@$(echo) "SHA512mp 32768:"; ./$(prog).32768.$(hname) --lifetime 3 --numMiBytes 4 --numInputData 2

# end of file
