using Base.Test
using DataFrames
include("../src/factor_analysis.jl")

df = DataFrame(has_aphasia=[1, 2], has_aphasia_1=[1, 2])
_remove_aphasia_dupe_cols(df)
@test names(df) == [:has_aphasia]
@test size(df) == (2, 1)
df_diff = DataFrame(has_aphasia=[1, 2], has_aphasia_1=[4, 3])
@test_throws AssertionError _remove_aphasia_dupe_cols(df_diff)


df = DataFrame(one_four=@data([1, 2, 3, NA]), pure=@data([1, 2, 3, 4]))
function in_thresh(col::Symbol, thresh::Float64)
  in(col, names(_remove_na_cols(df, thresh)))
end
function in_thresh(col::Symbol)
  in(col, names(_remove_na_cols(df)))
end

@test !in_thresh(:one_four, .76)
@test in_thresh(:pure, .76)
@test in_thresh(:one_four)
@test !in_thresh(:pure, 1.0)


df = DataFrame(a=@data([1, 2, 3, NA]), b=@data([1, 2, 3, 4]), c=@data([2, 4, NA, NA]))
@test size(_remove_na_rows(df)) == (2, 3)
@test size(_remove_na_rows(df, .6)) == (3, 3)


df = DataFrame(a=@data([1, NA, 3]), b=@data([10, NA, 30]))
df = _fillna(df)
@test df[:a] == [1, 2, 3]
@test df[:b] == [10.0, 20.0, 30.0]
