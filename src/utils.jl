using Tensors, ForwardDiff

function get1DTensor(array::Union{Array{T, N}, Adjoint{T,Array{T,N}}}, dim::Int64 = 3) where {T, N}
    return reinterpret(Tensor{1,dim, T, dim}, vec(array))
end


function get2DTensor(array::Union{Array{T, N}, Adjoint{T,Array{T,N}}},  dim::Int64 = 3) where {T, N}
    return reinterpret(Tensor{2,dim, T, dim^2}, vec(array))
end

function get4DTensor(array::Union{Array{T, N}, Adjoint{T,Array{T,N}}},  dim::Int64 = 3) where {T, N}
    return reinterpret(Tensor{2,dim, T, dim^4}, vec(array))
end

function get_∂u_∂X_Tensor(array::Union{Array{T, N}, Adjoint{T,Array{T,N}}},  dim::Int64 = 3) where {T, N}
    get2DTensor(array, dim)[1]
end

function getDeformationGradient(∂u_∂X::Array{T, N}) where {T,N}
    ∂u_∂X_Tensor = get_∂u_∂X_Tensor(∂u_∂X)
    return getDeformationGradient(∂u_∂X_Tensor)
end

function getDeformationGradient(∂u_∂X::T) where T
    return one(∂u_∂X) + ∂u_∂X
end

function getDeformationGradientFromCurrent(∂u_∂x::Array{T, N}) where {T,N}
    ∂u_∂x_Tensor = get_∂u_∂X_Tensor(∂u_∂x)
    return getDeformationGradientFromCurrent(∂u_∂x_Tensor)
end

function getDeformationGradientFromCurrent(∂u_∂x::T) where T
    return inv(one(∂u_∂x) - ∂u_∂x)
end

function getJacobianDeformationGradient(F::Array{Tensor{2,3,T,9},1}) where T
    return det(reshape(reinterpret(T, F), (3, 3)))
end


getJacobianDeformationGradient(F::T) where T = det(F)

function getRightCauchyTensor(F::T) where T
    #return F'⋅F
    return tdot(F)
end

function getLeftCauchyTensor(F::T) where T
    #return F⋅F'
    return dott(F)
end

function getGreenLagrangeStrain(F::T) where T
    C = getRightCauchyTensor(F)
    return 0.5*(C - one(C))
end

function getCauchyTensor(F::T) where T
    return inv(getLeftCauchyTensor(F))
end

function getAlmansiStrain(F::T) where T
    c = getCauchyTensor(F)
    return 0.5*(one(F) - c)
end

getFirstInvariant(tensor::T) where T = tr(tensor)

getSecondInvariant(tensor::T) where T = 0.5*(tr(tensor)^2 - tr(tensor^2))

getThirdInvariant(tensor::T) where T = det(tensor)

function getPrincipalStretches(F::T) where T
    return sqrt.(eigvals(getRightCauchyTensor(F)))
end
