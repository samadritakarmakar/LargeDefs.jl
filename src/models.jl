struct HyperElasticModel
    strainEnergyDensity::Function
    secondPiolaStress::Function
    materialTangentTensor::Function
end

function strainEnergyDensityGreenLagrangeBased(model_ψ::Function, E::SymmetricTensor{2,dim,T}, parameters::Tuple) where {dim, T}
    return model_ψ(E, parameters)
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

function cauchyStress(model::HyperElasticModel, F::Tensor{2,dim,T}, parameters::Tuple) where {dim, T}
    E = getGreenLagrangeStrain(F)
    S = model.secondPiolaStress(E, parameters)
    return 1/det(F)*F⋅S⋅F'
end

function spatialTangentTensor(model::HyperElasticModel, F::Tensor{2,dim,T}, parameters::Tuple) where {dim, T}
    E = getGreenLagrangeStrain(F)
    ℂ = model.materialTangentTensor(E, parameters)
    #F_front = otimesu(F, F)
    #F_back = otimesl(F, F)
    #return 1/det(F)*(F_front⊡C⊡F_back) 
    𝕔 = zeros(3,3,3,3)
    for L ∈ 1:3
        for K ∈ 1:3
            for J ∈ 1:3
                for I ∈ 1:3
                    ℂ_IJKL=ℂ[I,J,K,L]
                    for l ∈ 1:3
                        F_lL = F[l,L]
                        for k ∈ 1:3
                            F_kK =F[k,K]
                            for j ∈ 1:3
                                F_jJ = F[j,J]
                                for i ∈ 1:3
                                    𝕔[i,j,k,l] += F[i,I]*F_jJ*F_kK*F_lL*ℂ_IJKL
                                end
                            end
                        end
                    end
                end
            end
        end
    end 
    return 1/det(F)*Tensor{4,3, Float64}(𝕔)
end

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

function saintVenantStrainEnergyDensity(E::SymmetricTensor{2,dim,T}, parameters::Tuple{Float64, Float64}) where {dim, T}
    return strainEnergyDensityGreenLagrangeBased(saintVenant_ψ, E, parameters)
end

function saintVenantSecondPiola(E::SymmetricTensor{2,dim,T}, parameters::Tuple{Float64, Float64}) where {dim, T}
    return secondPiolaStressGreenLagrangeBased(saintVenant_ψ, E, parameters)
end

function saintVenantMaterialTangent(E::SymmetricTensor{2,dim,T}, parameters::Tuple{Float64, Float64}) where {dim, T}
    return materialTangentGreenLagrangeBased(saintVenant_ψ, E, parameters)
end

const saintVenant = HyperElasticModel(saintVenantStrainEnergyDensity, saintVenantSecondPiola, saintVenantMaterialTangent)
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

function neoHookeanCompressibleStrainEnergyDensity(E::SymmetricTensor{2,dim,T}, parameters::Tuple{Float64, Float64}) where {dim, T}
    return strainEnergyDensityGreenLagrangeBased(neoHookeanCompressible_ψ, E, parameters)
end

function neoHookeanCompressibleSecondPiola(E::SymmetricTensor{2,dim,T}, parameters::Tuple{Float64, Float64}) where {dim, T}
    return secondPiolaStressGreenLagrangeBased(neoHookeanCompressible_ψ, E, parameters)
end

function neoHookeanCompressibleMaterialTangent(E::SymmetricTensor{2,dim,T}, parameters::Tuple{Float64, Float64}) where {dim, T}
    return materialTangentGreenLagrangeBased(neoHookeanCompressible_ψ, E, parameters)
end

const neoHookeanCompressible = HyperElasticModel(neoHookeanCompressibleStrainEnergyDensity, neoHookeanCompressibleSecondPiola, neoHookeanCompressibleMaterialTangent)
#######################################################################################

##########################Neo Hookean#########################################
function neoHookean_ψ(E::SymmetricTensor{2,dim,T}, D1_μ::Tuple{Float64, Float64}) where {dim, T}
    D1 = D1_μ[1]
    μ = D1_μ[2]
    C = 2*E + one(E)
    Ic = getFirstInvariant(C)
    J = sqrt(det(C))
    Ī₁ = J^(-2/3)*Ic
    return (μ / 2) * (Ī₁ - 3.0) +  (1.0/D1) * (J - 1)^2
end

function neoHookeanStrainEnergyDensity(E::SymmetricTensor{2,dim,T}, parameters::Tuple{Float64, Float64}) where {dim, T}
    return strainEnergyDensityGreenLagrangeBased(neoHookean_ψ, E, parameters)
end

function neoHookeanSecondPiola(E::SymmetricTensor{2,dim,T}, parameters::Tuple{Float64, Float64}) where {dim, T}
    return secondPiolaStressGreenLagrangeBased(neoHookean_ψ, E, parameters)
end

function neoHookeanMaterialTangent(E::SymmetricTensor{2,dim,T}, parameters::Tuple{Float64, Float64}) where {dim, T}
    return materialTangentGreenLagrangeBased(neoHookean_ψ, E, parameters)
end

const neoHookean = HyperElasticModel(neoHookeanStrainEnergyDensity, neoHookeanSecondPiola, neoHookeanMaterialTangent)

