# Velocity input

The Overthrust input used for the paper has 185 x 801 x 801 nodal samples.
It is a headerless array of 32-bit floating-point velocities in metres/second,
in native byte order (little-endian on the reference systems). The first
index, depth, varies fastest. MATLAB `single(v(:))` ordering for an array
of size `[185,801,801]` gives this layout.

Reference SHA-256:
`e46ee28191a572e22271dd3503aa02c643d74cc836edb141b67a4ace229f202d`.

No PML samples are stored in the input. The solvers extend the model and
distribute coefficient evaluation/reading. Native spacing is 25 m;
refinement factors 2, 4, and 8 use 12.5, 6.25, and 3.125 m, respectively.
With eight PML intervals per side, the total mesh intervals in display
(horizontal, horizontal, depth) order are:

```
refine    displayed mesh          solver nodal dimensions
1         816 x 816 x 200         201 x 817 x 817
2         1616 x 1616 x 384       385 x 1617 x 1617
4         3216 x 3216 x 752       753 x 3217 x 3217
8         6416 x 6416 x 1488      1489 x 6417 x 6417
```

The paper source is `(500,2500,2500)` metres in solver coordinate order
(depth, horizontal, horizontal). Do not silently swap coordinate axes.
