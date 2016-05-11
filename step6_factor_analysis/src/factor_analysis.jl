using DataFrames
using Gadfly
using MultivariateStats


_data_dir = "../../data/"

_step5Data(f::AbstractString) = "$_data_dir/step5/$f.csv"

@enum Step5Data dependency lexical syntax

_step5Data(d::Step5Data) = begin
  f::AbstractString = d == dependency ? "dependencies" : "$d"
  _step5Data(f)
end


function _removeStrokeDupeCols(df::DataFrame)
  for c in names(df)
    c_str = string(c)

    if(startswith(c_str, "had_stroke") & (c != :had_stroke))
      @assert df[:had_stroke] == df[c]
      delete!(df, c)
    end
  end

end


function _removeNaCols(df::DataFrame, thresh::Float64=.7)
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


function _removeNaRows(df::DataFrame, thresh::Float64=.7)
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


function loadContinuous(d::Step5Data,
                        post_thresh_filter::Function;
                        col_thresh::Float64=.7,
                        row_thresh::Float64=.7,
                        fill_na_fn::Function=mean)

  f::AbstractString = _step5Data(d)
  ret = readtable(f)
  rename!(ret, :x, :id)

  valid_rows = !(isna(ret[:had_stroke]))
  ret = ret[valid_rows, :]

  _removeStrokeDupeCols(ret)

  ret = _removeNaCols(ret, col_thresh)
  ret = _removeNaRows(ret, row_thresh)
  ret = _fillna(ret, fill_na_fn)

  ret = addAphasiaClassifications(ret)

  ret[post_thresh_filter(ret), :]

end


typealias ThreshMap Dict{Step5Data, Float64}

function ThreshMap(d::Float64, l::Float64, s::Float64)
  Dict{Step5Data, Float64}(dependency=>d, lexical=>l, syntax=>s)
end

ThreshMap(f::Float64) = ThreshMap(f, f, f)

function loadContinuous(col_thresh::ThreshMap=ThreshMap(.7),
                        row_thresh::ThreshMap=ThreshMap(.7),
                        fill_na_fn=mean;
                        post_thresh_filter::Function = combine_fxns(
                            AbstractVector{Bool},
                            df -> (df[:had_stroke] .== 1)::AbstractVector{Bool},
                            aphasia_count_filter_gen(4),
                         &)
                       )
  typealias ColRowThreshes Tuple{Float64, Float64}

  rd(d::Step5Data, cr_threshes::ColRowThreshes) = begin
    c_thresh::Float64, r_thresh::Float64 = cr_threshes
    loadContinuous(d, post_thresh_filter,
                    col_thresh=c_thresh, row_thresh=r_thresh,
                    fill_na_fn=fill_na_fn)
  end

  threshes(d::Step5Data) = (col_thresh[d], row_thresh[d])

  Dict{Step5Data,DataFrame}([d => rd(d, threshes(d)) for d in (dependency, lexical, syntax)])
end


function getPcaInput(df::DataFrame)
  valid_cols::AbstractVector{Symbol} = begin
    invalid_cols = Set{Symbol}([:had_stroke, :id, :aphasia_type])
    valid_col_fn(c::Symbol) = !in(c, invalid_cols)
    filter(valid_col_fn, names(df))
  end

  Matrix(df[:, valid_cols])', valid_cols
end


function pca(df::DataFrame, maxoutdim=1)
  mat::Matrix{Float64}, _ = getPcaInput(df)
  pca_model = fit(PCA, mat; maxoutdim=maxoutdim)
  recon_data::Matrix{Float64} = transform(pca_model, mat)

  ret = DataFrame(id=df[:id])
  for i in 1:maxoutdim
    ret[symbol("recon_$i")] = recon_data[i, :][:]::Vector{Float64}
  end

  (ret, pca_model)
end


function addAphasiaClassifications(df::DataFrame,
                                     ignore_cols::Set{Symbol} = Set([:severity]))
  pat_class = readtable("$_data_dir/patient_classifications.csv")

  ret::DataFrame = join(df, pat_class, on=:id, kind=:inner)
  valid_cols::Vector{Bool} = Bool[!in(c, ignore_cols) for c in names(ret)]
  ret[:, valid_cols]
end


typealias Continuous Dict{Step5Data, DataFrame}

function pca(dfs::Continuous,
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


typealias Step5s Tuple{Step5Data, Step5Data}
function plot_(pca_df::DataFrame,
               leftright::Nullable{Step5s}=Nullable{Step5s}())

  left, right = isnull(leftright) ? names(pca_df)[2:3] : map(symbol, get(leftright))

  df_plot::DataFrame = addAphasiaClassifications(pca_df)

  plot(df_plot, y=left, x=right, color=:aphasia_type,
       Guide.Title("$left vs $right"))
end

plot_(pca_df::DataFrame, left::Step5Data, right::Step5Data) = plot_(pca_df,
  Nullable((left, right)))



function plot_(continuous::Continuous, left::Step5Data, right::Step5Data)
  pca_df::DataFrame, _ = pca(continuous, left, right)
  plot_(pca_df, left, right)
end


function filterForRecCols(dfs::Continuous)
  ret = Continuous()

  rec_cols::Vector{Symbol} = begin
    cols = readtable("../../data/step5/rec_cols.csv")[:col_name]
    Symbol[symbol(n) for n in cols]
  end
  
  for step5_data::Step5Data in keys(dfs)
    df::DataFrame = dfs[step5_data]
    valid_cols::Vector{Symbol} = intersect(rec_cols, names(df))
    ret[step5_data] = df[[:id; :aphasia_type; valid_cols]]
  end
  
  ret[lexical][:speech_rate] = dfs[lexical][:word_count]/6
  
  ret
end


function filterOutAphasiaType(dfs::Continuous, aphasia_type::AbstractString="none")
  ret = Continuous()

  for step5_data::Step5Data in keys(dfs)
    df::DataFrame = dfs[step5_data]
    valid_rows::Vector{Bool} = df[:aphasia_type] .!= aphasia_type
    ret[step5_data] = df[valid_rows, :]
  end
  
  ret
end