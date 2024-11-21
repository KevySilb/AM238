using GLMakie
using QuadGK
using StaticArrays
using DifferentialEquations
using Statistics

function cumsumtrap(f::Function, x::LinRange{Float64, Int64})
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

μ(x) = exp(-x) * ^((exp(1.0) - ^(exp(1.0), -1.0)), -1.0)
integ(x::Function, sup::SVector{2}) = quadgk(x, sup[1], sup[2]; atol=1e-8, rtol=1e-8)[1]
sup = SVector{2}(-1.0, 1.0)
function stieltjes()
    M = 8
    n = 2
    π = Vector{Function}(undef, M)
    π[n-1] = x -> 0.0 * x^0.0  # π₀(x) = 0
    π[n] = x -> 1.0 * x^0.0    # π₁(x) = 1
    α = Vector{Float64}(undef, M)
    β = Vector{Float64}(undef, M)
    β[n-1] = 0.0
    β[n] = 0.0  
    α[n] = integ(x -> x * π[n](x) * π[n](x) * μ(x), sup) / integ(x -> π[n](x) * π[n](x) * μ(x), sup)
    π[n+1] = x -> (x - α[n]) * π[n](x) - β[n] * π[n-1](x)
    for n in 3:M-1
        α[n] = integ(x -> x * π[n](x) * π[n](x) * μ(x), sup) / integ(x -> π[n](x) * π[n](x) * μ(x), sup)
        β[n] = integ(x -> π[n](x) * π[n](x) * μ(x), sup) / integ(x -> π[n-1](x) * π[n-1](x) * μ(x), sup)
        π[n+1] = π[n+1] = x -> (x - α[n]) * π[n](x) - β[n] * π[n-1](x)
    end
    return π[2:8]
end

function question3a()
    # get polynomials
    πn = stieltjes()

    # simulation initial and final time
    t0, tf = SVector{2}(0.0, 3.0)

    # initial conditions
    u0 = SVector{7}(1, 0, 0, 0, 0, 0, 0)

    # highest polynomial order
    M = size(πn)[1]
    
    # precompute E{Pj}
    Epj1 = Vector{Float64}(undef, M)
    for k in eachindex(πn)
        Epj1[k] = integ(x -> πn[k](x) * μ(x), sup)
    end
    
    # precompute E{Pj^2}
    Epj2 = Vector{Float64}(undef, M)
    for j in eachindex(πn)
            Epj2[j] = integ(x -> πn[j](x) * πn[j](x) * μ(x), sup)
    end

    # precompute E{P₁PkPj}
    Ep1pjpk = Matrix{Float64}(undef, M, M)
    for idx in CartesianIndices(Ep1pjpk)
        (j, k) = idx.I
        ele = integ(x -> πn[2](x) * πn[j](x) * πn[k](x) * μ(x), sup)
        Ep1pjpk[j, k] = ele < 1e-15 ? 0.0 : ele
    end
    #=
    # precompute the constant multiplying xhatj
    β = (2.0 / (exp(1.0)^2 + 1.0))

    # define the system of odes
    function ode(u, p, t)
        return SVector{M}(
            ((-sum(u[k] * Ep1pjpk[j, k] for k in 1:M))/Epj2[j]) + β*u[j] + (cos(4*t)*Epj1[j]/Epj2[j]) for j in 1:M
        )
    end

    # solve the ode problem using Tsitouras 5/4 Runge-Kutta method
    prob = ODEProblem(ode, u0, (t0, tf))
    sol = solve(prob, Tsit5(), saveat=0.01, abstol=1e-8, reltol=1e-8)

    # find the variance
    var = sum([(sol[k, :].^2) .* Epj2[k] for k in 2:M], dims=1)[1]

    # plot the mean and variance
    fig = Figure(size = (800, 400));display(fig)
    ax1 = Axis(fig[1, 1], title = "mean of x(t;ω)",
        xlabel = "time")
    ax2 = Axis(fig[1, 2], title = "variance of x(t;ω)",
        xlabel = "time")    
    lines!(ax1, sol.t, sol[1, :])
    lines!(ax2, sol.t, var)
    save("question3a.png", fig)=#
    Ep1pjpk
end

function question3b()
    # get orthogonal polynomials
    πn = stieltjes()
    
    # get gPC modes
    sol = question3a()
    
    # generate samples of ξ
    r = LinRange(-1, 1, 1000)
    ys = hcat(cumsumtrap(μ, r), r)
    ξ_samples = Vector{Float64}(undef, 50000)
    for i in eachindex(ξ_samples)
        ξ_samples[i] = sampleInverseCDF(rand(), ys)
    end

    # define t*
    times = SVector{4}(0.5, 1.0, 2.0, 3.0)

    # number of solutions (gPC modes)
    M = size(sol)[1]

    # define figure
    fig = Figure(size = (800, 800))
    ax = SVector{4}(Axis(fig[1, 1]), Axis(fig[1, 2]), Axis(fig[2, 1]), Axis(fig[2, 2]))
    
    # plot histogram of samples of x(tstar;ω)
    for (i, tstar) in enumerate(times)
        hist!(ax[i], sum(sol(tstar)[j] .* πn[j].(ξ_samples) for j in 1:M), bins = 80, normalization = :pdf)
        ax[i].title = "t = $tstar"
        ax[i].ylabel = "normalized frequency"
    end
    Label(fig[0, :], "PDF of various times, for x(t;ω)", fontsize=20)
    save("question3b.png", fig)
end


function validation(N::Int64)
    # generate samples of ξ
    r = LinRange(-1, 1, 1000)
    ys = hcat(cumsumtrap(μ, r), r)
    ξ_samples = Vector{Float64}(undef, N)
    for i in eachindex(ξ_samples)
        ξ_samples[i] = sampleInverseCDF(rand(), ys)
    end

    # define t*
    times = SVector{4}(0.5, 1.0, 2.0, 3.0)

    # define the ODEProblem
    function ode(u, p, t)
    	return SVector{1}(-p * u[1] + cos(4*t))
    end

    # initial condition
    u0 = SVector{1}(1)

    # solve once to get size of time vector
    prob = ODEProblem(ode, u0, (0.0, 3.0), -1.0)
    sol = solve(prob, Tsit5(), saveat=0.01, abstol=1e-8, reltol=1e-8)
    
    # init matrix to store solutions
    solTstar = [Float64[] for _ in 1:length(times)]
    solutions = Matrix{Float64}(undef, N, size(sol.t)[1])

    # solve the ode for each sample and extract sample paths
    for (i, ξ) in enumerate(ξ_samples)
        prob = ODEProblem(ode, u0, (0.0, 3.0), ξ)
        sol = solve(prob, Tsit5(), saveat=0.01, abstol=1e-8, reltol=1e-8)
        solutions[i, :] = Float64[u[1] for u in sol.u]
        for (j, tstar) in enumerate(times)
            push!(solTstar[j], sol(tstar)[1])
        end
    end

    # calculate sample path mean and variance
    meanPath = mean(solutions, dims=1)
    variancePath = var(solutions, mean=meanPath, dims=1)

    # plot results
    fig1 = Figure(size = (800, 800))
    ax1 = [Axis(fig1[i, j]) for i in 1:2, j in 1:2]
    fig2 = Figure(size = (800, 400))
    ax2 = [Axis(fig2[1, i]) for i in 1:2]
    for (i, tstar) in enumerate(times)
        hist!(ax1[i], solTstar[i], bins = 80, normalization = :pdf)
        ax1[i].title = "t = $tstar"
        ax1[i].ylabel = "Normalized Frequency"
    end
    Label(fig1[0, :], "PDF of Various Times Using MC Method", fontsize=20)
    lines!(ax2[1], sol.t, vec(meanPath), label="Mean Path")
    ax2[1].title = "Mean Path"
    ax2[1].xlabel = "Time"
    lines!(ax2[2], sol.t, vec(variancePath), label="Variance")
    ax2[2].title = "Variance over Time"
    ax2[2].xlabel = "Time"
    save("validationPDF.png", fig1)
    save("validationmeanvar.png", fig2)
end
