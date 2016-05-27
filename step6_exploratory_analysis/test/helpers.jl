using Base.Test
using DataFrames
include("../src/helpers.jl")

df = DataFrame(had_stroke=[1, 2], had_stroke_1=[1, 2])
_removeStrokeDupeCols(df)
@test names(df) == [:had_stroke]
@test size(df) == (2, 1)
df_diff = DataFrame(had_stroke=[1, 2], had_stroke_1=[4, 3])
@test_throws AssertionError _removeStrokeDupeCols(df_diff)


df = DataFrame(one_four=@data([1, 2, 3, NA]), pure=@data([1, 2, 3, 4]))
function inThresh(col::Symbol, thresh::Float64)
  in(col, names(_removeNaCols(df, thresh)))
end
function inThresh(col::Symbol)
  in(col, names(_removeNaCols(df)))
end

@test !inThresh(:one_four, .76)
@test inThresh(:pure, .76)
@test inThresh(:one_four)
@test !inThresh(:pure, 1.0)


df = DataFrame(a=@data([1, 2, 3, NA]), b=@data([1, 2, 3, 4]), c=@data([2, 4, NA, NA]))
@test size(_removeNaRows(df)) == (2, 3)
@test size(_removeNaRows(df, .6)) == (3, 3)


df = DataFrame(a=@data([1, NA, 3]), b=@data([10, NA, 30]))
df = _fillna(df)
@test df[:a] == [1, 2, 3]
@test df[:b] == [10.0, 20.0, 30.0]
