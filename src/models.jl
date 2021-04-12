struct HyperElasticModel
    secondPiolaStress::Function
    materialTangentTensor::Function
end

function secondPiolaStressGreenLagrangeBased(model_ψ::Function, E::SymmetricTensor{2,dim,T}, parameters::Tuple) where {dim, T}
    ψ(e) = model_ψ(e, parameters)
    return gradient(ψ, E)
end

#=
function secondPiolaStressRightCauchyBased(model_ψ::Function, C::Tensor{2,dim,T}, parameters::Tuple) where {dim, T}
    ψ(c) = model_ψ(c, parameters)
    return 2*gradient(ψ, c)
end
=#

function materialTangentGreenLagrangeBased(model_ψ::Function, E::SymmetricTensor{2,dim,T}, parameters::Tuple) where {dim, T}
    ψ(e) = model_ψ(e, parameters)
    return hessian(ψ, E)
end

#=
function materialTangentRightCauchyBased(model_ψ::Function, C::Tensor{2,dim,T}, parameters::Tuple) where {dim, T}
    ψ(c) = model_ψ(c, parameters)
    return 4*hessian(ψ, c)
end
=#

##############Models are Defined Below################################################

#########################Saint Venant###################################################3
function saintVenant_ψ(E::SymmetricTensor{2,dim,T}, λ_μ::Tuple{Float64, Float64}) where {dim, T}
    λ = λ_μ[1]
    μ = λ_μ[2]
    C = 2*E + one(E)
    Ic = getFirstInvariant(C)
    IIc = getSecondInvariant(C)
    return 1.0/8.0*(λ + 2*μ)*(Ic - 3.0)^2 - μ/2.0 * (-2*Ic + IIc + 3)
end

function saintVenantSecondPiola(E::SymmetricTensor{2,dim,T}, parameters::Tuple{Float64, Float64}) where {dim, T}
    return secondPiolaStressGreenLagrangeBased(saintVenant_ψ, E, parameters)
end

function saintVenantMaterialTangent(E::SymmetricTensor{2,dim,T}, parameters::Tuple{Float64, Float64}) where {dim, T}
    return materialTangentGreenLagrangeBased(saintVenant_ψ, E, parameters)
end

const saintVenant = HyperElasticModel(saintVenantSecondPiola, saintVenantMaterialTangent)
#######################################################################################

##########################Neo Hookean Compressible#########################################
function neoHookeanCompressible_ψ(E::SymmetricTensor{2,dim,T}, λ_μ::Tuple{Float64, Float64}) where {dim, T}
    λ = λ_μ[1]
    μ = λ_μ[2]
    C = 2*E + one(E)
    Ic = getFirstInvariant(C)
    J = sqrt(det(C))
    return (λ / 2) * (log(J))^2 - μ * log(J) + (μ / 2) * (Ic - 3)
end

function neoHookeanCompressibleSecondPiola(E::SymmetricTensor{2,dim,T}, parameters::Tuple{Float64, Float64}) where {dim, T}
    return secondPiolaStressGreenLagrangeBased(neoHookeanCompressible_ψ, E, parameters)
end

function neoHookeanCompressibleMaterialTangent(E::SymmetricTensor{2,dim,T}, parameters::Tuple{Float64, Float64}) where {dim, T}
    return materialTangentGreenLagrangeBased(neoHookeanCompressible_ψ, E, parameters)
end

const neoHookeanCompressible = HyperElasticModel(neoHookeanCompressibleSecondPiola, neoHookeanCompressibleMaterialTangent)
#######################################################################################

##########################Neo Hookean#########################################
function neoHookean_ψ(E::SymmetricTensor{2,dim,T}, D1_μ::Tuple{Float64, Float64}) where {dim, T}
    D1 = D1_μ[1]
    μ = D1_μ[2]
    C = 2*E + one(E)
    Ic = getFirstInvariant(C)
    J = sqrt(det(C))
    Ī₁ = J^(-2/3)*Ic
    return (μ / 2) * (Ic - 3.0) #+  (1.0/D1) * (J - 1)^2
end

function neoHookeanSecondPiola(E::SymmetricTensor{2,dim,T}, parameters::Tuple{Float64, Float64}) where {dim, T}
    return secondPiolaStressGreenLagrangeBased(neoHookean_ψ, E, parameters)
end

function neoHookeanMaterialTangent(E::SymmetricTensor{2,dim,T}, parameters::Tuple{Float64, Float64}) where {dim, T}
    return materialTangentGreenLagrangeBased(neoHookean_ψ, E, parameters)
end

const neoHookean = HyperElasticModel(neoHookeanSecondPiola, neoHookeanMaterialTangent)
