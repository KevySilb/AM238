using GLMakie
function MRG32k3a(seed::Integer, N::Integer)
    m₁ = 2.0^32 - 209
    m₂ = 2.0^32 - 22853

    X = Vector{Float64}(undef, N+3)
    Y = Vector{Float64}(undef, N+3)

    for i = 1:3
        X[i] = Float64(seed)
        Y[i] = Float64(seed)
    end

    for j in 1:length(X) - 3
        X[j+3] = mod(1403580*X[j+1] - 810728*X[j], m₁)
        Y[j+3] = mod(527642*Y[j+2] - 1370589*Y[j], m₂)
    end
    
    transformX(Xk::Float64) = Xk ≥ 0 ? Xk : Xk - m₁*Xk
    transformY(Yk::Float64) = Yk ≥ 0 ? Yk : Yk - m₂*Yk
    
    Xhat = transformX.(X)
    Yhat = transformY.(Y)

    Uk = Vector{Float64}(undef, N)
    for i in eachindex(Xhat)
        if i > length(Xhat) - 3
            break
        end
        Uk[i] = mod(Xhat[i+3] - Yhat[i+3], m₁) / (m₁ + 1)
    end
    Uk
end

function question2()
    fig = Figure()
    ax = Axis(fig[1, 1],
        title = "Relative frequencies of Uⱼ for N = 10⁶",
        xlabel = "Uⱼ",
        ylabel = "frequency"
    )
    hist!(ax, MRG32k3a(111, 1000000), bins = 80)
    save("question2.png", fig)
end
