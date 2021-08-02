module LargeDefs
using Tensors, LinearAlgebra, ForwardDiff
include("utils.jl")
include("models.jl")

#utils.jl
export get2DTensor, get_∂u_∂X_Tensor, getDeformationGradient, getJacobianDeformationGradient
export getRightCauchyTensor, getLeftCauchyTensor, getGreenLagrangeStrain
export getCauchyTensor, getAlmansiStrain
export getFirstInvariant, getSecondInvariant, getThirdInvariant
export getPrincipalStretches

#models

export HyperElasticModel
export strainEnergyDensityGreenLagrangeBased, secondPiolaStressGreenLagrangeBased, materialTangentGreenLagrangeBased, cauchyStress, spatialTangentTensor
export saintVenant_ψ, saintVenantSecondPiola, saintVenantMaterialTangent, saintVenant
export neoHookeanCompressible_ψ, neoHookeanCompressibleSecondPiola, neoHookeanCompressibleMaterialTangent, neoHookeanCompressible

end # module
