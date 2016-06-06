using Base.Test
using PyCall
include("../src/mlHelpers.jl")

@pyimport numpy.random as nprand
@pyimport sklearn.metrics as skmet

a = nprand.randint(0, 3, 300)
b = nprand.randint(0, 3, 300)

@test_approx_eq skmet.f1_score(a, b, "weighted")  f1score(a, b)

typealias Pairs Vector{Pair}

actual = _evalInputToModelStates(:a=>[1, 2], :b=>[3, 4])
ab(a,b) = Dict(:a=>a, :b=>b)
expected = Dict[ab(1,3), ab(1, 4), ab(2, 3), ab(2, 4)]
@test actual == expected


abc(a,b,c) = Dict(:a=>a, :b=>b, :c=>c)
expected = Dict[abc(1, 3, 5), abc(1, 3, 6),
                abc(1, 4, 5), abc(1, 4, 6),
                abc(2, 3, 5), abc(2, 3, 6),
                abc(2, 4, 5), abc(2, 4, 6)]
actual = _evalInputToModelStates(:a => [1, 2], :b=> [3, 4], :c=>[5, 6])
@test actual == expected
