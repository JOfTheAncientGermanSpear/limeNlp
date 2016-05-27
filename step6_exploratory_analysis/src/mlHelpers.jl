using MLBase


type Pipeline{T}
  fits::Vector{Function}
  predicts::Vector{Function}
  score_fn::Function
  truths::AbstractVector{T}
end

run(p::Pipeline, field::Symbol, ixs::Vector{Int64}) = foldl(ixs, p.(field)) do prev_out, fn::Function
  fn(prev_out)
end


fit(pipe::Pipeline, ixs::Vector{Int64}) = run(pipe, :fits, ixs)


predict(pipe::Pipeline, ixs::Vector{Int64}) = run(pipe, :predicts, ixs)


function test(pipe::Pipeline, ixs::Vector{Int64})
  preds = predict(pipe, ixs)
  truths = pipe.truths[ixs]
  pipe.score_fn(preds, truths)
end


function evalModel(pipe::Pipeline, cvg::CrossValGenerator,
    num_iterations::Int64, num_samples::Int64)

    train_scores = zeros(Float64, num_iterations)
    fit_call = 0

    function fit(ixs::Int64)
      fit_call += 1
      fit(pipeline, ixs)
      train_scores[fit_call] = test(pipe, ixs)
    end

    test(ixs::Int64) = test(pipe, ixs)

    test_scores = cross_validate(fit, test, num_samples, cvg)
    train_scores, test_scores
end
