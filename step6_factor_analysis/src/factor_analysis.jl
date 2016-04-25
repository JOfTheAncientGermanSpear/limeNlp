using DataFrames
using Gadfly
using MultivariateStats


_data_dir = "../../data/"

_step5_data(f::AbstractString) = "$_data_dir/step5/$f.csv"

@enum Step5Data dependency lexical syntax

_step5_data(d::Step5Data) = begin
  f::AbstractString = d == dependency ? "dependencies" : "$d"
  _step5_data(f)
end

function _remove_aphasia_dupe_cols(df::DataFrame)
  for c in names(df)
    c_str = string(c)

    if(startswith(c_str, "has_aphasia") & (c != :has_aphasia))
      @assert df[:has_aphasia] == df[c]
      delete!(df, c)
    end
  end

end


function _remove_na_cols(df::DataFrame, thresh::Float64=.7)
  num_rows = size(df, 1)
  max_nas = (1 - thresh) * num_rows

  function within_limit(col::Symbol)
    num_nas = sum(isna(df[col]))
    ret::Bool = num_nas < max_nas
    ret
  end

  cols_to_keep = filter(within_limit, names(df))
  df[cols_to_keep]
end


function _remove_na_rows(df::DataFrame, thresh::Float64=.7)
  num_cols = size(df, 2)
  max_nas = (1 - thresh) * num_cols

  function nas_cnt(row::DataFrameRow)
    sum(isna([val for (sym, val) in row]))
  end

  rows_to_keep::Vector{Bool} = [nas_cnt(r) < max_nas for r in eachrow(df)]
  df[rows_to_keep, :]
end


function _fillna(df::DataFrame, val_fn::Function=mean)
  df = copy(df)
  for c in names(df)
    na_rows = isna(df[c])
    val = val_fn(df[!na_rows, c])
    df[na_rows, c] = val
  end
  df
end


function _is_aphasia_type_gen(aphasia_type::AbstractString)
  pat_class = readtable("$_data_dir/patient_classifications.csv")
  type_rows = pat_class[:aphasia_type] .== aphasia_type
  pat_ids = Set(pat_class[type_rows, :id])

  function select_ids(df::DataFrame)
    Bool[in(id, pat_ids) for id in df[:id]]
  end
end


function get_counts{T}(arr::AbstractArray{T})
  typealias CountMap Dict{T, Int64}
  reduce(CountMap(), arr) do acc::CountMap, e::T
    curr::Int64 = get(acc, e, 0)
    acc[e] = curr + 1
    acc
  end
end


function aphasia_count_filter_gen(cutoff::Int64)
  function filter_fn(df::DataFrame)
    counts::Dict{AbstractString, Int64} = filter(
      (k::AbstractString, v::Int64) -> v > cutoff,
      get_counts(df[:aphasia_type])
    )
    passing_types::Set{AbstractString} = Set(keys(counts))
    Bool[in(t, passing_types) for t in df[:aphasia_type]]
  end
end


function combine_fxns(T::Type, fn1::Function, fn2::Function, joinfn::Function)
  a -> joinfn(fn1(a), fn2(a))::T
end


function load_continuous(d::Step5Data,
                        post_thresh_filter::Function;
			col_thresh::Float64=.7,
                        row_thresh::Float64=.7,
                        fill_na_fn::Function=mean)

  f::AbstractString = _step5_data(d)
  ret = readtable(f)
  rename!(ret, :x, :id)

  valid_rows = !(isna(ret[:has_aphasia]))
  ret = ret[valid_rows, :]

  _remove_aphasia_dupe_cols(ret)

  ret = _remove_na_cols(ret, col_thresh)
  ret = _remove_na_rows(ret, row_thresh)
  ret = _fillna(ret, fill_na_fn)

  ret = add_aphasia_classifications(ret)

  ret[post_thresh_filter(ret), :]

end


typealias ThreshMap Dict{Step5Data, Float64}

function ThreshMap(d::Float64, l::Float64, s::Float64)
  Dict{Step5Data, Float64}(dependency=>d, lexical=>l, syntax=>s)
end

ThreshMap(f::Float64) = ThreshMap(f, f, f)

function load_continuous(col_thresh::ThreshMap=ThreshMap(.7),
                         row_thresh::ThreshMap=ThreshMap(.7),
                         fill_na_fn=mean;
			 post_thresh_filter::Function = combine_fxns(
			   AbstractVector{Bool},
			   df -> (df[:has_aphasia] .== 1)::AbstractVector{Bool},
			   aphasia_count_filter_gen(4),
			   &)
			 )
  typealias ColRowThreshes Tuple{Float64, Float64}

  rd(d::Step5Data, cr_threshes::ColRowThreshes) = begin
    c_thresh::Float64, r_thresh::Float64 = cr_threshes
    load_continuous(d, post_thresh_filter,
                    col_thresh=c_thresh, row_thresh=r_thresh,
                    fill_na_fn=fill_na_fn)
  end

  threshes(d::Step5Data) = (col_thresh[d], row_thresh[d])

  Dict{Step5Data,DataFrame}([d => rd(d, threshes(d)) for d in (dependency, lexical, syntax)])
end


function get_pca_input(df::DataFrame)
  valid_cols::AbstractVector{Symbol} = begin
    invalid_cols = Set{Symbol}([:has_aphasia, :id, :aphasia_type])
    valid_col_fn(c::Symbol) = !in(c, invalid_cols)
    filter(valid_col_fn, names(df))
  end

  Matrix(df[:, valid_cols])', valid_cols
end


function pca(df::DataFrame, maxoutdim=1)
  mat::Matrix{Float64}, _ = get_pca_input(df)
  pca_model = fit(PCA, mat; maxoutdim=maxoutdim)
  recon_data::Matrix{Float64} = transform(pca_model, mat)

  ret = DataFrame(id=df[:id])
  for i in 1:maxoutdim
    ret[symbol("recon_$i")] = recon_data[i, :][:]::Vector{Float64}
  end

  (ret, pca_model)
end


function add_aphasia_classifications(df::DataFrame,
                                     ignore_cols::Set{Symbol} = Set([:severity]))
  pat_class = readtable("$_data_dir/patient_classifications.csv")

  ret::DataFrame = join(df, pat_class, on=:id, kind=:inner)
  valid_cols::Vector{Bool} = Bool[!in(c, ignore_cols) for c in names(ret)]
  ret[:, valid_cols]
end


function pca(dfs::Dict{Step5Data, DataFrame},
             lname::Step5Data, rname::Step5Data)

  ((llower_dim, lmodel), (rlower_dim, rmodel)) = map([lname, rname]) do s::Step5Data
    df::DataFrame = dfs[s]
    data, model = pca(df)
    rename!(data, :recon_1, Symbol("$s"))
    (data, model)
  end

  (join(llower_dim, rlower_dim, kind=:inner, on=:id),
   (lmodel, rmodel))
end


function plot_(df::DataFrame, x_col::Symbol, y_col::Symbol)
  df_plot::DataFrame = add_aphasia_classifications(df)

  plot(df_plot, x=x_col, y=y_col, color=:aphasia_type,
       Guide.Title("$y_col vs $x_col"))
end

plot_(df::DataFrame, x::Step5Data, y::Step5Data) = plot_(df, symbol(x), symbol(y))
