using Tensors, ForwardDiff

function get1DTensor(array::Array{T, N}) where {T, N}
    return reinterpret(Tensor{1,length(array), Float64}, vec(array))
end


function get2DTensor(array::Array{T, N}, dim::Int64 = 3) where {T, N}
    return reinterpret(Tensor{2,dim, T, dim^2}, vec(array))
end

function get4DTensor(array::Array{T, N}, dim::Int64 = 3) where {T, N}
    return reinterpret(Tensor{2,dim, T, dim^4}, vec(array))
end

function get_∂u_∂X_Tensor(array::Array{T, 2}, dim::Int64 = 3) where T
    get2DTensor(array::Array{T, 2}, dim)
end

function getDeformationGradient(∂u_∂X::T) where T
    return reinterpret(Tensor{2,3,Float64, 9} ,vec([1.0 0 0; 0 1.0 0; 0 0 1.0])) + ∂u_∂X
end

function getJacobianDeformationGradient(F::Array{Tensor{2,3,T,9},1}) where T
    return det(reshape(reinterpret(T, F), (3, 3)))
end


getJacobianDeformationGradient(F::T) where T = det(F)

function getRightCauchyTensor(F::T) where T
    return F'⋅F
end

function getLeftCauchyTensor(F::T) where T
    return F⋅F'
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
