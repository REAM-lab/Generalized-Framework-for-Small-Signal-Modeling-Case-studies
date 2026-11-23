# Make sure to run this script before running the main script to ensure that the required Julia packages are installed.
from juliacall import Main as jl

jl.seval('using Pkg')
jl.seval('Pkg.add(["ModelingToolkit", "OrdinaryDiffEq", "LinearAlgebra"])')

jl.seval('using ModelingToolkit')
jl.seval('using OrdinaryDiffEq')

print("ModelingToolkit successfully loaded!")