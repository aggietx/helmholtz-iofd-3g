.DEFAULT_GOAL := all
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CXX_FLAGS += -qopenmp-simd
all: solver
solver.o: compact_outer.hpp shell_scatter.hpp
solver: solver.o
	${CLINKER} -o $@ $^ ${PETSC_KSP_LIB}

