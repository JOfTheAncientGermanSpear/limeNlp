using MLBase

typealias Functions Vector{Function}

type Pipeline
  fits::Functions
  predicts::Functions
  score_fn::Function
  truths::AbstractVector
  model_state::Function
  model_state!::Function
end


typealias IXs AbstractVector{Int64}

_runFns(p::Pipeline, f::Symbol, ixs::IXs) = foldl(ixs, p.(f)) do prev_out, fn::Function
  fn(prev_out)
end


modelState!(pipe::Pipeline, s::Symbol, value) = pipe.model_state!(s, value)
modelState!(pipe::Pipeline, d::Dict{Symbol, Any}) = for (s, v) in d
  modelState!(pipe, s, v)
end

modelState(pipe::Pipeline, s::Symbol) = pipe.model_state(s)


pipeFit(pipe::Pipeline, ixs::IXs) = _runFns(pipe, :fits, ixs)


pipePredict(pipe::Pipeline, ixs::IXs) = _runFns(pipe, :predicts, ixs)


function pipeTest(pipe::Pipeline, ixs::IXs)
  preds = pipePredict(pipe, ixs)
  truths = pipe.truths[ixs]
  pipe.score_fn(preds, truths)
end


function evalModel(pipe::Pipeline, cvg::CrossValGenerator, num_samples::Int64)

  num_iterations = length(cvg)

  train_scores = zeros(Float64, num_iterations)
  fit_call = 0

  function fit(ixs::IXs)
    fit_call += 1
    pipeFit(pipe, ixs)
    train_scores[fit_call] = pipeTest(pipe, ixs)
  end

  test(_, ixs::IXs) = pipeTest(pipe, ixs)

  test_scores = cross_validate(fit, test, num_samples, cvg)
  train_scores, test_scores
end


typealias ModelState Dict{Symbol, Any}
typealias ModelStates AbstractVector{ModelState}
function evalModelParallel(pipeGen::Function, cvg::CrossValGenerator, num_samples::Int64,
  states::ModelStates)
  scores = map(states) do s::ModelState
    pipe = pipeGen()
    modelState!(pipe, s)
    train_scores, test_scores = evalModel(pipe, cvg, num_samples)
    mean(train_scores), mean(test_scores)
  end
  [t[1] for t in scores], [t[2] for t in scores]
end


function evalModel(pipe::Pipeline, cvg::CrossValGenerator, num_samples::Int64,
  states::ModelStates)
  scores = map(states) do s::ModelState
    modelState!(pipe, s)
    train_scores, test_scores = evalModel(pipe, cvg, num_samples)
    mean(train_scores), mean(test_scores)
  end
  [t[1] for t in scores], [t[2] for t in scores]
end


sqDist(x, y) = norm( (y - x).^2, 1)

function r2score{T <: Real}(y_true::AbstractVector{T}, y_pred::AbstractVector{T})

  dist_from_pred::Float64 = sqDist(y_true, y_pred)
  dist_from_mean::Float64 = sqDist(y_true, mean(y_true))

  1 - dist_from_pred/dist_from_mean
end


function precisionScore(y_true::AbstractVector{Bool}, y_pred::AbstractVector{Bool})
  true_pos = sum(y_pred & y_true)
  false_pos = sum(y_pred & !y_true)

  true_pos/(true_pos + false_pos)
end


function recallScore(y_true::AbstractVector{Bool}, y_pred::AbstractVector{Bool})
  true_pos = sum(y_pred & y_true)
  false_neg = sum(!y_pred & y_true)

  true_pos/(true_pos + false_neg)
end


function MLBase.f1score(y_true::AbstractVector{Bool}, y_pred::AbstractVector{Bool})
  precision = precisionScore(y_true, y_pred)
  recall = recallScore(y_true, y_pred)

  2 * precision * recall / (precision + recall)
end


function MLBase.f1score{T}(y_true::AbstractVector{T}, y_pred::AbstractVector{T})
  num_samples = length(y_true)
  labels = unique(union(y_true, y_pred))

  if length(labels) < 3
    return f1score(y_true .== labels[1], y_pred .== labels[1])
  end

  reduce(0., labels) do acc::Float64, l::T
    truths = y_true .== l
    score = begin
      preds = y_pred .== l
      true_pos = sum(truths & preds)
      true_pos > 0. ? f1score(truths, preds) : 0.
    end
    weight = sum(truths)/num_samples
    acc + score * weight
  end
end
