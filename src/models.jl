struct HyperElasticModel
    secondPiolaStress::Function
    materialTangentTensor::Function
end

function secondPiolaStressGreenLagrangeBased(model_ψ::Function, E::Tensor{2,dim,T}, parameters::Tuple) where {dim, T}
    ψ(e) = model_ψ(e, parameters)
    return gradient(ψ, E)
end

#=
function secondPiolaStressRightCauchyBased(model_ψ::Function, C::Tensor{2,dim,T}, parameters::Tuple) where {dim, T}
    ψ(c) = model_ψ(c, parameters)
    return 2*gradient(ψ, c)
end
=#

function materialTangentGreenLagrangeBased(model_ψ::Function, E::Tensor{2,dim,T}, parameters::Tuple) where {dim, T}
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
function saintVenant_ψ(E::Tensor{2,dim,T}, λ_μ::Tuple{Float64, Float64}) where {dim, T}
    λ = λ_μ[1]
    μ = λ_μ[2]
    Ie = getFirstInvariant(E)
    IIe = getSecondInvariant(E)
    return 0.5*(λ + 2*μ)*Ie^2 - 2*μ*IIe
end

function saintVenantSecondPiola(E::Tensor{2,dim,T}, parameters::Tuple{Float64, Float64}) where {dim, T}
    return secondPiolaStressGreenLagrangeBased(saintVenant_ψ, E, parameters)
end

function saintVenantMaterialTangent(E::Tensor{2,dim,T}, parameters::Tuple{Float64, Float64}) where {dim, T}
    return materialTangentGreenLagrangeBased(saintVenant_ψ, E, parameters)
end

const saintVenant = HyperElasticModel(saintVenantSecondPiola, saintVenantMaterialTangent)
#######################################################################################

##########################Neo Hookean Compressible#########################################
function neoHookeanCompressible_ψ(E::Tensor{2,dim,T}, λ_μ::Tuple{Float64, Float64}) where {dim, T}
    λ = λ_μ[1]
    μ = λ_μ[2]
    C = 2*E + one(E)
    Ic = getFirstInvariant(C)
    J = sqrt(det(C))
    return μ / 2 * (Ic - 3) - μ * log(J) + λ / 2 * log(J)^2
end

function neoHookeanCompressibleSecondPiola(E::Tensor{2,dim,T}, parameters::Tuple{Float64, Float64}) where {dim, T}
    return secondPiolaStressGreenLagrangeBased(neoHookeanCompressible_ψ, E, parameters)
end

function neoHookeanCompressibleMaterialTangent(E::Tensor{2,dim,T}, parameters::Tuple{Float64, Float64}) where {dim, T}
    return materialTangentGreenLagrangeBased(neoHookeanCompressible_ψ, E, parameters)
end

const neoHookeanCompressible = HyperElasticModel(neoHookeanCompressibleSecondPiola, neoHookeanCompressibleMaterialTangent)
#######################################################################################

##########################Neo Hookean#########################################
function neoHookean_ψ(E::Tensor{2,dim,T}, λ_μ::Tuple{Float64, Float64}) where {dim, T}
    λ = λ_μ[1]
    μ = λ_μ[2]
    C = 2*E + one(E)
    Ic = getFirstInvariant(C)
    J = sqrt(det(C))
    return μ / 2 * (Ic - 3)
end

function neoHookeanSecondPiola(E::Tensor{2,dim,T}, parameters::Tuple{Float64, Float64}) where {dim, T}
    return secondPiolaStressGreenLagrangeBased(neoHookean_ψ, E, parameters)
end

function neoHookeanMaterialTangent(E::Tensor{2,dim,T}, parameters::Tuple{Float64, Float64}) where {dim, T}
    return materialTangentGreenLagrangeBased(neoHookean_ψ, E, parameters)
end

const neoHookean = HyperElasticModel(neoHookeanSecondPiola, neoHookeanMaterialTangent)
