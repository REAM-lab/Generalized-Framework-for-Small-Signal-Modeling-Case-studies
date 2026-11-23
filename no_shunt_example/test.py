from juliacall import Main as jl

jl.seval('using Pkg')
jl.seval('Pkg.add(["ModelingToolkit", "OrdinaryDiffEq", "LinearAlgebra", "Plots"])')

jl.seval('using ModelingToolkit')
jl.seval('using OrdinaryDiffEq')

print("ModelingToolkit successfully loaded!")