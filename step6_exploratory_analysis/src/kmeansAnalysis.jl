using Clustering
using Gadfly

include("helpers.jl")

getKmeansInput(continuous::Continuous, s5data::Step5Data) = getDataMat(
  continuous[s5data])[1]'


function runKmeans(continuous::Continuous, s5data::Step5Data, k::Int64=4)
  input::Matrix{Float64} = getKmeansInput(continuous, s5data)
  kmeans(input, k)
end


function plotKmeans(continuous::Continuous, s5data::Step5Data, k::Int64=4;
                    color_col::Symbol=:aphasia_type)
  kres = runKmeans(continuous, s5data, k)
  assignments::Vector{Int64} = kres.assignments
  assignment_orders::Vector{Int64} = sortperm(assignments)

  plot(Geom.histogram, x=assignments,
    color=continuous[s5data][color_col])
end
