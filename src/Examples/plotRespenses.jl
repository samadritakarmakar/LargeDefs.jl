using LargeDefs, Plots

function plotResponses()
    E::Float64 = 1.0e1 #MPa
    ν::Float64 = 0.0
    λ = (E * ν) / ((1 + ν) * (1 - 2ν))
    μ = E/(2*(1+ν))
    modelParams  = (λ, μ)
    modelType = "Neo Hookean Compressible"
    if modelType == "Saint Venant"
        hyperModel = LargeDefs.saintVenant
    elseif modelType == "Neo Hookean Compressible"
        hyperModel = LargeDefs.neoHookeanCompressible
    else
        error("$modelType is unknown")
    end
    ∂u_∂X = zeros(3,3)
    ∂u_∂X_1 = Vector{Float64}()
    b_1 = Vector{Float64}()
    σ_1 = Vector{Float64}()
    ψ = Vector{Float64}()
    noOfSteps = 1000
    startDisp = -1.0
    lastDisp = 1.0
    stepDisp = (lastDisp - startDisp)/noOfSteps
    for i ∈ 0:noOfSteps 
        ∂u_∂X[1,1] = startDisp+i*stepDisp
        F = getDeformationGradient(∂u_∂X)
        b = getLeftCauchyTensor(F)
        push!(b_1, b[1])
        push!(∂u_∂X_1, ∂u_∂X[1])
        σ = LargeDefs.cauchyStress(hyperModel, F, modelParams)
        push!(σ_1, σ[1])
        𝐄 = getRightCauchyTensor(F)
        push!(ψ,hyperModel.strainEnergyDensity(𝐄, (λ, μ)))
    end
    plt1 = plot(b_1, σ_1, xlabel = "Left Cauchy Tensor b₁₁ = (F⋅Fᵀ)₁₁", ylabel = "Cauchy Stress σ₁₁ = J⁻¹ (∂ψ(b)/∂b)₁₁", label = "$modelType", legend = :bottomright)
    savefig(plt1, "$modelType.png")
    plt1 = plot(∂u_∂X_1, σ_1, xlabel = "Displacement Gradient ∂u_∂X₁₁", ylabel = "Cauchy Stress σ₁₁ = J⁻¹ (∂ψ(b)/∂b)₁₁", label = "$modelType", legend = :bottomright)
    savefig(plt1, "$(modelType)2.png")
    plt1 = plot(∂u_∂X_1, ψ, xlabel = "Displacement Gradient ∂u_∂X₁₁", ylabel = "Strain Energy ψ", label = "$modelType", legend = :bottomright)
    savefig(plt1, "$(modelType)3.png")
end
