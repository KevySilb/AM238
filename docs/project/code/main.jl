using GLMakie
using StaticArrays
using OrdinaryDiffEq
using DynamicalSystems
using LinearAlgebra

global forcing = Observable(0.3)

function reset()
    forcing[] = 0.3
end

function windSDE(forcing, Δt)
    μ, θ, σ = SVector{3}(
        0.3, # Mean forcing
        0.5, # Mean reversion rate
        0.2, # Standard dev
    )
    forcing[] = θ * (μ - forcing[]) * Δt + σ * sqrt(Δt) * randn()
end

function PDdrone(u0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; params)
    diffeq = (alg = Tsit5(), abstol = 1e-6, reltol = 1e-6)
    return CoupledODEs(droneRule, u0, params; diffeq)
end

function trajectory(t)
    y = 0.2
    dy = 0.0
    ddy = 0.0
    z = 1.0
    dz = 0.0
    ddz = 0.0
    return SVector{6}(y, dy, ddy, z, dz, ddz)
end

function controller(x, p, t)
    yd, dyd, ddyd, zd, dzd, ddzd = trajectory(t)
    g, m, Ixx, L, T_max = p
    Kpy, Kdy, Kpz, Kdz, Kpϕ, Kdϕ  = SVector{6}(0.4, 1.0, 0.4, 1.0, 18.0, 15.0)
    
    # define PD controller
    ϕd = -(1.0/g) * (ddyd + Kpy*(yd - x[1]) + Kdy*(dyd - x[4]))
    F = (m * g) + (m * (ddzd + Kpz*(zd - x[2]) + Kdz*(dzd - x[5])))
    τ = Ixx * (Kpϕ*(ϕd - x[3]) + Kdϕ*(-x[6]))

    # clamp motors to max throttle
    TL = clamp(0.5 * (F - τ/L), 0.0, T_max)
    TR = clamp(0.5 * (F + τ/L), 0.0, T_max)
    return TL + TR, (TR - TL)*L
end

function droneRule(u, p, t)
    # parameters
    g, m, Ixx, _, _ = p

    # states
    _, _, x3, x4, x5, x6 = u
    
    # inputs
    F, τ = controller(u, p, t)

    # system
    dx1 = x4
    dx2 = x5
    dx3 = x6
    dx4 = -(F * sin(x3))/m
    dx5 = ((F * cos(x3))/m) - g
    dx6 = (τ / Ixx)
    return SVector{6}(dx1, dx2, dx3, dx4, dx5, dx6)
end

function animstep!(integ, ϵ) 
    step!(integ)
    Δt = integ.dt
    windSDE(forcing, Δt)
    ϵ[] = [integ[1], integ[2], integ[3]]
end

function main()
    # mass (kg) drone + battery
    #m = 0.61 + 0.48
    m = 0.18
    # Arm length (m)
    #L = 0.25
    L = 0.086
    # max thrust
    #T_max = 1.332
    T_max = 1.7658
    
    # initial conditions
    u0 = [
        0.0,    # initial y position
        0.0,    # initial z position
        0.0,    # initial tilt position
        0.0,    # initial velocity in y
        0.0,    # initial velocity in z
        0.0     # initial velocity in ϕ
    ]
    
    # parameters 
    params = [9.81, m, m*L^2, L, T_max]
    
    fig, integ, ϵ = makefig(u0, params)

    run = Button(fig[2, 1]; label = "run", tellwidth = false)
    isrunning = Observable(false)
    on(run.clicks) do clicks; isrunning[] = !isrunning[]; end
    on(run.clicks) do clicks
        @async while isrunning[]
            isopen(fig.scene) || break 
            animstep!(integ, ϵ)
            sleep(0.02)
        end
    end

    ax = content(fig[1, 1])
    Makie.deactivate_interaction!(ax, :rectanglezoom)
end

function makefig(u0, params)
    L = params[4]
    h = L * 0.1
    pd = PDdrone(u0; params)
    integ = pd.integ

    # Observables
    ϵ = Observable([integ[1], integ[2], integ[3]])
    points = [-L -h; L -h; L h; -L h; -L -h]'

    # Listeners
    drone = lift(ϵ) do p
        translated_points = (
            [cos(p[3]) -sin(p[3]); sin(p[3]) cos(p[3])] * copy(points) .+ [p[1], p[2]]
        )
        map(col -> Point2f(col...), eachcol(translated_points))
    end

    fig = Figure(); display(fig)
    ax = Axis(fig[1, 1])
    lines!(ax, drone, linewidth = 1, color = :black)
    ax.title = "2D drone PD controller"
    xlims!(ax, -1.5, 1.5)
    ylims!(ax, 0, 2.0)
    return fig, integ, ϵ
end
