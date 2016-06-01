using MLBase
using MultivariateStats
using PyCall

include("helpers.jl")
include("factorAnalysis.jl")
include("mlHelpers.jl")

LinSVC = begin
  @pyimport sklearn.svm as SVM
  SVM.LinearSVC
end

function pipelineGen(initial_state::Dict{Symbol, Any} = Dict{Symbol, Any}())

  lex = loadContinuous()[lexical]

  pca_ref::Dict{Symbol, Any} = Dict(:maxoutdim=>1)

  svc = LinSVC()

  typealias XY Tuple{Matrix{Float64}, Vector{ASCIIString}}

  function getXy(ixs::AbstractVector{Int64})
    X::Matrix{Float64} = getDataMat(lex[ixs, :])[1]
    y::Vector{ASCIIString} = [i for i in lex[ixs, :aphasia_type_general]]
    X, y
  end

  function pcaFit!(Xy::XY)
    X, y = Xy
    pca_ref[:pca] = fit(PCA, X'; maxoutdim=pca_ref[:maxoutdim])
    pcaTransform((X, y)), y
  end

  svcFit!(Xy::XY) = svc[:fit](Xy[1], Xy[2])

  pcaTransform(Xy::XY) = MultivariateStats.transform(pca_ref[:pca], Xy[1]')'

  svcPredict(X::Matrix) = ASCIIString[p for p in svc[:predict](X)]

  fits = Function[getXy, pcaFit!, svcFit!]
  predicts = Function[getXy, pcaTransform, svcPredict]

  truths = begin
    num_rows = size(lex, 1)
    ASCIIString[t for t in getXy(1:num_rows)[2]]
  end

  modelState!(s::Symbol, val) = @switch s begin
    :svc_C; svc[:C] = val;
    :pca_dim; pca_ref[:maxoutdim] = val;
  end

  modelState(s::Symbol) = @switch s begin
    :svc_C; svc[:C];
    :pca_dim; pca_ref[:maxoutdim];
  end

  for (s, v) in initial_state
    modelState!(s, v)
  end

  Pipeline(fits, predicts, f1score, truths, modelState, modelState!)

end
