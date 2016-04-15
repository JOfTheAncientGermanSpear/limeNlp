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


function load_continuous(d::Step5Data;
                        col_thresh::Float64=.7,
                        row_thresh::Float64=.7,
                        id_filter::Function=df -> df[:id] .> 1000,
                        fill_na_fn::Function=mean)

    f::AbstractString = _step5_data(d)
    ret = readtable(f)
    rename!(ret, :x, :id)

    ret = ret[id_filter(ret), :]

    valid_rows = !(isna(ret[:has_aphasia]))
    ret = ret[valid_rows, :]

    _remove_aphasia_dupe_cols(ret)

    ret = _remove_na_cols(ret, col_thresh)
    ret = _remove_na_rows(ret, row_thresh)
    ret = _fillna(ret, fill_na_fn)

    ret
end


typealias ThreshMap Dict{Step5Data, Float64}
ThreshMap(f::Float64) = Dict{Step5Data, Float64}(dependency=>f, lexical=>f, syntax=>f)

function load_continuous(id_filter=df -> df[:id] .> 1000,
                         col_thresh::ThreshMap=ThreshMap(.7),
                         row_thresh::ThreshMap=ThreshMap(.7),
                         fill_na_fn=mean)

  typealias ColRowThreshes Tuple{Float64, Float64}

  rd(d::Step5Data, cr_threshes::ColRowThreshes) = begin
      c_thresh::Float64, r_thresh::Float64 = cr_threshes
      load_continuous(d, col_thresh=c_thresh, row_thresh=r_thresh,
            id_filter=id_filter, fill_na_fn=fill_na_fn)
  end

  threshes(d::Step5Data) = (col_thresh[d], row_thresh[d])

  Dict{Step5Data,DataFrame}([d => rd(d, threshes(d)) for d in (dependency, lexical, syntax)])
end


function get_pca_input(df::DataFrame)
    valid_cols::AbstractVector{Symbol} = begin
	invalid_cols = Set{Symbol}([:has_aphasia, :id])
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


function pca(dfs::Dict{Step5Data, DataFrame},
             lname::Step5Data, rname::Step5Data)

  ((llower_dim, lmodel), (rlower_dim, rmodel)) = map([lname, rname]) do s
      df::DataFrame = dfs[s]
      data, model = pca(df)
      rename!(data, :recon_1, Symbol("$s"))
      data, model
  end

  (join(llower_dim, rlower_dim, kind=:inner, on=:id),
   (lmodel, rmodel))
end


function add_aphasia_classifications(df::DataFrame)
  pat_class = readtable("$_data_dir/patient_classifications.csv")
  join(df, pat_class, on=:id, kind=:inner)
end


function plot_(df::DataFrame, x_col::Symbol, y_col::Symbol)

  df_plot = add_aphasia_classifications(df)
  plot(df_plot, x=x_col, y=y_col, color=:aphasia_type,
       Guide.Title("$y_col vs $x_col"))
end
