module LargeDefs
include("utils.jl")
include("models.jl")

#utils.jl
export get2DTensor, get_∂u_∂X_Tensor, getDeformationGradient, getJacobianDeformationGradient
export getRightCauchyTensor, getLeftCauchyTensor, getGreenLagrangeStrain
export getCauchyTensor, getAlmansiStrain
export getFirstInvariant, getSecondInvariant, getThirdInvariant

#models

export HyperElasticModel
export secondPiolaStressGreenLagrangeBased, materialTangentGreenLagrangeBased
export saintVenant_ψ, saintVenantSecondPiola, saintVenantMaterialTangent, saintVenant
export neoHookeanCompressible_ψ, neoHookeanCompressibleSecondPiola, neoHookeanCompressibleMaterialTangent, neoHookeanCompressible

end # module
