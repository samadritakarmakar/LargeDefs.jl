using Tensors, ForwardDiff

function get2DTensor(array::Array{T, 2}, dim::Int64 = 3) where T
    return Tensor{2,dim,T}((i,j) -> array[i,j])
end

function get_∂u_∂X_Tensor(array::Array{T, 2}, dim::Int64 = 3) where T
    get2DTensor(array::Array{T, 2}, dim)
end

function getDeformationGradient(∂u_∂X::Tensor{2,dim,T}) where {dim, T}
    return one(∂u_∂X) + ∂u_∂X
end

getJacobianDeformationGradient(F) = det(F)

function getRightCauchyTensor(F::Tensor{2,dim,T}) where {dim, T}
    return F'⋅F
end

function getLeftCauchyTensor(F::Tensor{2,dim,T}) where {dim, T}
    return F⋅F'
end

function getGreenLagrangeStrain(F::Tensor{2,dim,T}) where {dim, T}
    return 0.5*(getRightCauchyTensor(F) - one(F))
end

function getCauchyTensor(F::Tensor{2,dim,T}) where {dim, T}
    return inv(getLeftCauchyTensor(F))
end

function getAlmansiStrain(F::Tensor{2,dim,T}) where {dim, T}
    c = getCauchyTensor(F)
    return 0.5*(one(F) - c)
end

getFirstInvariant(tensor::Tensor{2,dim,T}) where {dim, T} = tr(tensor)

getSecondInvariant(tensor::Tensor{2,dim,T}) where {dim, T} = 0.5*(tr(tensor)^2 - tr(tensor^2))

getThirdInvariant(tensor::Tensor{2,dim,T}) where {dim, T} = det(tensor)
