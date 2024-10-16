using Base: undef_ref_str
using GLMakie
using QuadGK

function cumsumtrap(f::Function, x)
    y = f.(x)
    N = length(x)
    x1 = @view x[1:N-1]
    x2 = @view x[2:N]
    y1 = @view y[1:N-1]
    y2 = @view y[2:N]
    integral = cumsum(((x2.-x1).*(y1.+y2))./2.0)
    integral ./= integral[end]
    return [0; integral]
end

function sampleInverseCDF(x::Float64, points::Matrix{Float64})
    idx = findfirst(points[:, 1] .> x)
    if idx === nothing
        p1 = points[end-1, :]
        p2 = points[end, :]
    elseif idx == 1
        p1 = points[1, :]
        p2 = points[2, :]
    else
        p1 = points[idx-1, :]
        p2 = points[idx, :]
    end
    liy(x, p1, p2)        
end

function liy(x::Float64, p1::Vector{Float64}, p2::Vector{Float64})
    x1, y1 = p1
    x2, y2 = p2
    if isapprox(x1, x2, atol = 1e-12)
        return (y1 + y2) / 2.0
    end
    return y1 + (x - x1)*(y2 - y1)/(x2 - x1)
end

function Si(x::Float64)
    return x == 0.0 ? 0.0 : quadgk(t -> sin(t)/t, 0.0, x, rtol=1e-3)[1]
end

function p(x::Float64, y::Float64)
    if x < 0.0 || y < 0.0
        return 0.0
    end
    return ((40.0)/(Si(20.0) + 20.0))*cos(10.0*x*y)*cos(10.0*x*y)
end

function pxGy(x::Float64, y::Float64)
    denom = (20.0*y+sin(20.0*y))
    if abs(denom) < 1e-6
        return 0.0
    end
    return (40.0*y*cos(10.0*x*y)*cos(10.0*x*y))/denom
end

function pyGx(x::Float64, y::Float64)
    denom = (20.0*x+sin(20.0*x))
    if abs(denom) < 1e-6
        return 0.0
    end
    return (40.0*x*cos(10.0*x*y)*cos(10.0*x*y))/denom
end

function getSurface()
    xs = LinRange(0, 1, 100)
    ys = LinRange(0, 1, 100)
    zs = [p(x, y) for x in xs, y in ys]
    return xs, ys, zs
end

function plotSurface()
    xs, ys, zs = getSurface()
    fig = Figure()
    ax = Axis3(fig[1,1])
    surface!(ax, xs, ys, zs)
    return fig, ax
end

function plotContourf()
    xs, ys, zs = getSurface()
    fig = Figure()
    ax = Axis(fig[1,1])
    contourf!(ax, xs, ys, zs)
    return fig, ax
end

function plotConditionals()
    xs, ys, zs = getSurface()
    fig = Figure()
    grid = fig[1, 1] = GridLayout(tellwidth = false, tellheight = false)
    
    ax1 = Axis(grid[1, 1])
    ax2 = Axis(grid[1, 2])
    ax3 = Axis(grid[2, 1])
    r = LinRange(0.0, 1.0, 1000)
    yG = rand()
    xG = rand()
    contourf!(ax1, xs, ys, zs)
    hlines!(ax1, yG, color = :blue)
    vlines!(ax1, xG, color = :red)
    lines!(ax2, r, pxGy.(r, yG), color = :blue)
    lines!(ax3, r, pyGx.(xG, r), color = :red)
    fig
end

function plotCDF()
    fig = Figure()
    grid = fig[1, 1] = GridLayout()
    ax1 = Axis(grid[1, 1])
    ax2 = Axis(grid[1, 2])
    r = LinRange(0.0, 1.0, 1000)
    x1 = rand()
    y1 = cumsumtrap(y -> pyGx(x1, y), r)

    lines!(ax1, r, pyGx.(x1, r))

    samples = Vector{Float64}(undef, 5000)
    for i in eachindex(samples)
        samples[i] = sampleInverseCDF(rand(), hcat(y1, r))
    end

    hist!(ax2, samples, normalization = :pdf, bins = 80)
    fig
end

function gibbsSample(N::Integer)
    samples = Matrix{Float64}(undef, N, 2)
    r = LinRange(0.0, 1.0, 1000)
    # initialize random x1
    y0 = rand()
    samples[1, 1] = sampleInverseCDF(rand(), hcat(cumsumtrap(x -> pxGy(x, y0), r), r))
    for i=2:N
        samples[i-1, 2] = sampleInverseCDF(rand(), hcat(cumsumtrap(y -> pyGx(samples[i-1, 1], y), r), r))
        samples[i, 1] = sampleInverseCDF(rand(), hcat(cumsumtrap(x -> pxGy(x, samples[i-1, 2]), r), r))
    end
    samples
end

function partb(N::Integer)
    fig = Figure()
    ax = Axis(fig[1,1])
    xs, ys, zs = getSurface()
    contourf!(ax, xs, ys, zs)
    samples = gibbsSample(N)
    scatter!(ax, samples[:, 1], samples[:, 2], markersize = 3, color = :red)
    display(fig)
end

function f(y::Float64, X1::Float64, X2::Float64)
    return sin(4*π*X1*y) + cos(4*π*X2*y)
end

function partc()
    N = Int64(1e3)
    M = Int64(500)
    r = LinRange(0.0, 1.0, M)
    X1, X2 = gibbsSample(2)[:, 1]
    fig = Figure(size = (650,800))
    grid = fig[1, 1] = GridLayout()
    ax1 = Axis(grid[1, 1],
        title = "f(y;X1=$(round(X1, digits=3)), X2=$(round(X2, digits = 3)))")
    ax2 = Axis(grid[2, 1], title = "y by sample mean")
    ax3 = Axis(grid[3, 1], title = "y by standard deviation")
    samples = Matrix{Float64}(undef, N, M)
    cdf = hcat(cumsumtrap(y -> f(y, X1, X2), r), r)
    for idx in eachindex(samples)
        samples[idx] = sampleInverseCDF(rand(), cdf)
    end
    μᵢ = vec(sum(j -> j, samples, dims=1) ./ N)
    μₜ = sum(μᵢ) / M
    σᵢ = vec(sqrt.(sum((samples .- μᵢ') .^ 2, dims = 1) ./ (N - 1)))
    lines!(ax1, r, f.(r, X1, X2), color = :purple)
    lines!(ax2, r, μᵢ, color = :blue)
    lines!(ax3, r, σᵢ, color = :red)
    fig
end

