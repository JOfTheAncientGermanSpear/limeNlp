using Base.Test
using PyCall
include("../src/mlHelpers.jl")

@pyimport numpy.random as nprand
@pyimport sklearn.metrics as skmet

a = nprand.randint(0, 3, 300)
b = nprand.randint(0, 3, 300)

@test_approx_eq skmet.f1_score(a, b, "weighted")  f1score(a, b)
