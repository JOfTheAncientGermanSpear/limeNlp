using DataFrames

include("mlHelpers.jl")


function plotPredScores1D(pred_scores::AbstractVector,
    truths::AbstractVector)
  plot(x=pred_scores, color=truths, Geom.histogram)
end


function plotPredScores2D(pred_scores1::AbstractVector,
    pred_scores2::AbstractVector, truths::AbstractVector)
  plot(x=pred_scores1, y=pred_scores2, color=truths)
end


function plotPredScores3DPlus{T <: AbstractVector, G}(pred_scores::Vector{T},
    xs::AbstractVector, groups::AbstractVector{G}, truths::AbstractVector)

  @assert length(groups) == length(pred_scores)

  for scores in pred_scores
    @assert length(scores) == length(truths)
  end

  plot_df::DataFrame = @>> zip(groups, pred_scores) begin
    map( (g::G, scrs::AbstractVector) -> DataFrame(score=scrs, group=g, x=xs, truth=truths))
    vcat
  end

  plot(plot_df, ygroup="group", y="score", x="xs", truth="truths",
    Geom.subplot_grid(Geom.point))
end
