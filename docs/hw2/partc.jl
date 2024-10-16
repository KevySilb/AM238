using GLMakie
using QuadGK
function pX(x::Float64)
    return 1.0 / (π*(1+x^2))
end

function F⁻¹(x::Float64)
    return tan(π*(x - 0.5))
end

function partc(N::Integer)
    r = LinRange(-10.0, 10.0, 1000)
    samples = Vector{Float64}(undef, N)
    for i in eachindex(samples)
        samples[i] = F⁻¹(rand())
    end
    Xbar = sum(samples) / N
    fig = Figure()
    grid = fig[1, 1] = GridLayout()
    ax1 = Axis(grid[1, 1], title = "sample mean = $Xbar")
    linePlot = lines!(ax1, r, pX.(r), color = :red)
    histPlot = hist!(ax1, samples, normalization = :pdf, bins = 80)
    leg = Legend(fig, [linePlot, histPlot], ["p(x)", "$N samples"])
    grid[1, 2] = leg
    xlims!(ax1, -10.0, 10.0)
    fig
end
