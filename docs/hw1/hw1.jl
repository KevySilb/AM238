using GLMakie
using FresnelIntegrals
using QuadGK

function p(x::Real)
    (0 ≤ x ≤ 2.0) ? (2.0 * x * cos(x^2.0) + 5.0) / (10.0 + sin(4.0)) : 0.0
end

function E1(x::Real)
    return x * p(x)
end

function E2(x::Real)
    return x^2 * p(x)
end

function plotfunction(f::Function)
    fig = Figure(); display(fig)
    ax = Axis(fig[1, 1],
        xlabel = "x",
        ylabel = "f(x)")
    x_min, x_max, Δx = (-0.1, 2.1, 0.01)
    xrange = x_min:Δx:x_max
    lines!(ax, xrange, f.(xrange))
    xlims!(ax, x_min, x_max)
    return fig
end

function cumsumtrap(f, x)
    y = f.(x)
    N = length(x)
    dx = x[2:N] .- x[1:N-1]
    meanY = (y[2:N] .+ y[1:N-1]) ./ 2
    integral = cumsum(dx .* meanY)

    return [0; integral]
end

function plotCDF()
    fig = Figure();
    ax1 = Axis(fig[1,1],
        title = "CDF",
        xlabel = "x",
        ylabel = "F(x)"
    )
    r = -0.1:0.01:2.1
    lines!(ax1, r, cumsumtrap(p, r))
    xlims!(ax1, -0.1, 2.1)
    ylims!(ax1, -0.1, 1.1)
    display(fig)
end
