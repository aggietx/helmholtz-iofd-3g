# Attribution and external dependencies

The IOFD discretization and its numerical coefficient tables follow:

Christiaan C. Stolk, *A Dispersion Minimizing Scheme for the 3-D Helmholtz
Equation Based on Ray Theory*, Journal of Computational Physics 314 (2016),
618-646. DOI: 10.1016/j.jcp.2016.03.023.

The MIT license covers this repository's solver implementation. It does not
relicense external libraries, the cited article, or external velocity data.
PETSc, MPI, NVIDIA CUDA/cuBLAS, and optional Python packages are external
dependencies and are not vendored in this repository. Install them under
their respective license terms.

The SEG/EAGE Overthrust model is an external input and is not included.
Obtain it from an authorized source before running those benchmarks.

ChatGPT assisted with theoretical analysis, implementation, and language
editing. The authors are responsible for the paper and software contents.
