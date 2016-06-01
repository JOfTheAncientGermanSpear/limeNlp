using DataFrames
using Gadfly
using MultivariateStats

include("helpers.jl")


function pca(df::DataFrame, maxoutdim=1)
  mat::Matrix{Float64} = getDataMat(df)[1]'
  pca_model = fit(PCA, mat; maxoutdim=maxoutdim)
  recon_data::Matrix{Float64} = transform(pca_model, mat)

  ret = DataFrame(id=df[:id])
  for i in 1:maxoutdim
    ret[symbol("recon_$i")] = recon_data[i, :][:]::Vector{Float64}
  end

  (ret, pca_model)
end


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

  pca_df_copy = copy(pca_df)
  pca_df_copy[:had_stroke] = Int64[i < 300 ? 0 : 1 for i in pca_df[:id]]
  df_plot::DataFrame = addAphasiaClassifications(pca_df_copy)

  plot(df_plot, y=left, x=right, color=:aphasia_type_general,
       Guide.Title("$left vs $right"))
end

plot_(pca_df::DataFrame, left::Step5Data, right::Step5Data) = plot_(pca_df,
  Nullable((left, right)))



function plot_(continuous::Continuous, left::Step5Data, right::Step5Data)
  pca_df::DataFrame, _ = pca(continuous, left, right)
  plot_(pca_df, left, right)
end
