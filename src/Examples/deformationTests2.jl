using LargeDefs, LinearAlgebra, Tensors, PyPlot

function hyperElasticTest()
    #∂u_∂X = zeros(9)
    #∂u_∂X[1] = 1e-4
    E::Float64 = 10 #MPa
    ν::Float64 = 0.3
    λ = (ν*E)/((1+ν)*(1-2*ν))
    μ = E/(2*(1+ν))
    #μ = 3.8
    #λ = 1.0
    λ_μ = (λ, μ)
    ∂u_∂X_array_max = zeros(3,3)
    ∂u_∂X_array_max[1,1] = 0.5
    ∂u_∂X_array_max[2,2] = 0.0
    ∂u_∂X_array_max[3,3] = 0.0

    ∂u_∂X_array_max *= 1e-4

    model = LargeDefs.neoHookeanCompressible
    ∂u_∂X = LargeDefs.get_∂u_∂X_Tensor(∂u_∂X_array_max)
    F = LargeDefs.getDeformationGradient(∂u_∂X)
    σ1 = cauchyStress(model, F, (λ, μ))
    b = LargeDefs.getLeftCauchyTensor(F)
    J = det(F)
    I2 = one(SymmetricTensor{2,3, Float64})
    σ2 = 1/J *(μ*(b-I2)+λ*(log(J))*I2)
    println(norm(σ1-σ2))

    𝕔1 = spatialTangentTensor(model, F, (λ, μ))
    I4 = one(SymmetricTensor{4,3, Float64})
    𝕔2 = (2*(μ - λ*log(J))*I4+λ*I2⊗I2)/J
    println(norm(𝕔1-𝕔2))
end
