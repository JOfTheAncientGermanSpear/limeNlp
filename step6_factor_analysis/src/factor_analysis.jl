using DataFrames
using Gadfly
using MultivariateStats


_data_dir = "../../data/"

_step5_data(f) = "$_data_dir/step5/$f.csv"


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


function load_continuous(id_filter=df -> df[:id] .> 1000,
                         dep_path=_step5_data("dependencies"),
                         lex_path=_step5_data("lexical"),
                         syn_path=_step5_data("syntax"),
                         col_thresh=Dict(:dep=>.7, :lex=>.7, :syn=>.7),
                         row_thresh=Dict(:dep=>.7, :lex=>.7, :syn=>.7),
                         fill_na_fn=mean)

  function rd(f, cr_threshes)
    ret = readtable(f)
    rename!(ret, :x, :id)

    ret = ret[id_filter(ret), :]

    valid_rows = !(isna(ret[:has_aphasia]))
    ret = ret[valid_rows, :]

    _remove_aphasia_dupe_cols(ret)

    c_thresh, r_thresh = cr_threshes

    ret = _remove_na_cols(ret, c_thresh)
    ret = _remove_na_rows(ret, r_thresh)
    ret = _fillna(ret, fill_na_fn)

    ret
  end

  function threshes(key)
    (col_thresh[key], row_thresh[key])
  end

  Dict(:dep => rd(dep_path, threshes(:dep)),
       :lex => rd(lex_path, threshes(:lex)),
       :syn => rd(syn_path, threshes(:syn)))
end


function pca(dfs::Dict{Symbol, DataFrame},
             lname::Symbol, rname::Symbol)
  ((llower_dim, lmodel), (rlower_dim, rmodel)) = map([lname, rname]) do sym
    df = dfs[sym]
    mat::Matrix{Float64} = Matrix(df[:, 2:end])'
    pca_model = fit(PCA, mat; maxoutdim=1)
    recon_data::Vector{Float64} = transform(pca_model, mat)[:]
    (DataFrame(recon=recon_data, id=df[:id]), pca_model)
  end
  rename!(llower_dim, :recon, lname)
  rename!(rlower_dim, :recon, rname)

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
