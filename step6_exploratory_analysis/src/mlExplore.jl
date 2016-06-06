using MLBase
using MultivariateStats
using PyCall
using StatsBase

include("helpers.jl")
include("factorAnalysis.jl")
include("mlHelpers.jl")

LinSVC = begin
  @pyimport sklearn.svm as SVM
  SVM.LinearSVC
end


typealias XY{yT} Tuple{Matrix{Float64}, Vector{yT}}
function getXy(ixs::AbstractVector{Int64}, df::DataFrame,
    yT::Type, y_col::Symbol)
  X::Matrix{Float64} = getDataMat(df[ixs, :])[1]
  y::Vector{yT} = [i for i in df[ixs, y_col]]
  X, y
end

###Fit Functions

function keepVariantColsFit!(Xy::XY, state::Dict)
  X, y = Xy
  variant_cols = filter(c -> var(X[:, c]) > eps(Float64), 1:size(X, 2))
  state[:variant_cols] = variant_cols
  X[:, variant_cols], y
end

function standardizeFit!(Xy::XY, state::Dict)
  X, y = Xy
  state[:mean], state[:std] = mean_and_std(X, 1)
  zscore(X, 1), y
end

function svcFit!(Xy::XY, svc::PyObject,
                 params...)
  for (p::Symbol, v) in params
    svc[p] = v
  end
  svc[:fit](Xy[1], Xy[2])
end

function defaultFitGen(df::DataFrame,
    model_state::Dict,
    svc::PyObject;
    y_col::Symbol=:is_aphasiac,
    yT::Type = Bool)

  getXyLocal(ixs) = getXy(ixs, df, yT, y_col)

  keepVariantColsFitLocal!(Xy::XY) = keepVariantColsFit!(Xy, model_state)

  standardizeFitLocal!(Xy::XY) = standardizeFit!(Xy, model_state)

  function svcFitLocal!(Xy::XY)
    svcFit!(Xy, svc, :C => model_state[:svc_C])
    model_state[:svc_coef] = svc[:coef_]
  end

  Function[getXyLocal,
    keepVariantColsFitLocal!,
    standardizeFitLocal!,
    svcFitLocal!]
end
####


####Transforms
function defaultPredictGen(df::DataFrame,
    model_state::Dict,
    svc::PyObject)

  getX(ixs) = getXy(ixs, df, Bool, :is_aphasiac)[1]

  keepVariantColsTransform(X::Matrix) = X[:, model_state[:variant_cols]]

  standardizeTransform(X::Matrix) = zscore(X, model_state[:mean], model_state[:std])

  #svcPredict(X::Matrix) = yT[p for p in svc[:predict](X)]
  dotCoefs(X::Matrix) = (X * model_state[:svc_coef]')[:]

  Function[getX,
    keepVariantColsTransform,
    standardizeTransform,
    dotCoefs]
end
####

f1scoreMargin(truths::AbstractVector{Bool},
            margin_dists::AbstractVector) = f1score(truths, margin_dists .> 0)

typealias Functions Vector{Function}

function pipelineGen(step5::Step5Data = lexical,
    model_state::ModelState = ModelState(:svc_C=>1e-4))

  df = loadContinuous()[step5]

  svc = LinSVC()

  fits::Functions = defaultFitGen(df, model_state, svc)

  predicts::Functions = defaultPredictGen(df, model_state, svc)

  truths = begin
    num_rows = size(df, 1)
    getXy = fits[1]
    Bool[t for t in getXy(1:num_rows)[2]]
  end

  Pipeline(fits, predicts, f1scoreMargin, truths, model_state)
end

generateCvg(stratum::AbstractVector, num_iterations) = StratifiedRandomSub(
  stratum, round(Int64, length(stratum)*.8), num_iterations)

generateCvg(dfs::Continuous, s::Step5Data, num_iterations) = generateCvg(
  dfs[s][:is_aphasiac], num_iterations)

generateCvg(s::Step5Data, num_iterations) = generateCvg(
  loadContinuous(), s, num_iterations)


function ensemble(truths::AbstractVector, pipelines...)

  fitLocal(ixs::AbstractVector{Int64}) = for p in pipelines
    pipeFit!(p, ixs)
  end

  predictLocal(ixs::AbstractVector{Int64}) = @>> pipelines begin
    map(p::Pipeline -> pipePredict(p, ixs))
    sum
  end

  Pipeline([fitLocal], [predictLocal], f1scoreMargin, truths, ModelState())

end

function allPipelines()
  truths = Bool[t for t in loadContinuous()[lexical][:is_aphasiac]]

  p_dep, p_lex, p_syn = begin
    p_map::Dict{Step5Data, Pipeline} = allDs(s -> pipelineGen(s), Pipeline)
    (p_map[dependency], p_map[lexical], p_map[syntax])
  end

  p_ens = ensemble(truths, p_dep, p_lex, p_syn)

  p_dep, p_lex, p_syn, p_ens
end
