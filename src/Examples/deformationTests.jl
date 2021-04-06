using LargeDefs, LinearAlgebra, Tensors, PyPlot

function hyperElasticTest()
    #∂u_∂X = zeros(9)
    #∂u_∂X[1] = 1e-4
    E::Float64 = 10 #MPa
    ν::Float64 = 0.3
    λ = (ν*E)/((1+ν)*(1-2*ν))
    μ = E/(2*(1+ν))
    #μ = 2.0e3
    #λ = 1.5
    λ_μ = (λ, μ)
    ∂u_∂X_array_max = zeros(3,3)
    ∂u_∂X_array_max[1,1] = 0.0
    ∂u_∂X_array_max[2,2] = 0.0
    ∂u_∂X_array_max[3,3] = 0.0

    ∂u_∂X_array_max[1,2] = 0.5

    ∂u_∂X_array_min = zeros(3,3)
    ∂u_∂X_array_min[1,1] = 0.0
    ∂u_∂X_array_min[2,2] = 0.0
    ∂u_∂X_array_min[3,3] = 0.0
    #∂u_∂X_array = [0.5000000000000002 4.163336342344337e-17 -1.3877787807814457e-17; -2.0816681711721685e-17 -0.11982540819326219 -1.5959455978986625e-16; 2.0816681711721685e-17 1.3877787807814457e-17 -0.11982540819326216]
    ∂u_∂X_T_max = get_∂u_∂X_Tensor(∂u_∂X_array_max)
    ∂u_∂X_T_min = get_∂u_∂X_Tensor(∂u_∂X_array_min)
    totalSteps = 20
    𝔼_lastStep = zero(Tensor{2,3, Float64})
    S_check2 = zero(Tensor{2,3, Float64})
    S_hyd_array = zeros(totalSteps+1)
    S_eff_array = zeros(totalSteps+1)
    𝔼_array = zeros(totalSteps+1)
    σ_array = Array{SymmetricTensor{2,3,Float64, 6}, 1}(undef, totalSteps+1)
    σ₁₁_array = zeros(totalSteps+1)
    inv_C_array = zeros(totalSteps+1)
    ψ_array = zeros(totalSteps+1)
    λ₁ = zeros(totalSteps+1)
    Δ∂u_∂X = (∂u_∂X_T_max - ∂u_∂X_T_min)/totalSteps
    println("Δ∂u_∂X = ", Δ∂u_∂X)
    ∂u_∂X = deepcopy(∂u_∂X_T_min)
    #hyperModel = LargeDefs.saintVenant
    hyperModel = LargeDefs.neoHookeanCompressible
    #hyperModel = LargeDefs.neoHookean
    for step ∈ 0:totalSteps
        F = LargeDefs.getDeformationGradient(∂u_∂X)
        println("F = ", F)
        Jacobian = LargeDefs.getJacobianDeformationGradient(F)
        𝔼_step = LargeDefs.getGreenLagrangeStrain(F)
        println("E = ", 𝔼_step)

        ##############################
        S_check1 = hyperModel.secondPiolaStress(𝔼_step, λ_μ)
        println("S = ", S_check1)
        #############################
        ℂ = hyperModel.materialTangentTensor(𝔼_step, λ_μ)
        ############################
        #println("ℂ = ", ℂ)

        if step == 0 || step == 1
            S_check2 = deepcopy(S_check1)
        else
            S_check2 += ℂ⊡(𝔼_step-𝔼_lastStep)

        end
        ###########################;
        println("Second Piola Stress Check 1 :", norm(S_check2- S_check1))
        C = LargeDefs.getRightCauchyTensor(F)
        if hyperModel == LargeDefs.saintVenant
            S_check3 = λ*tr(𝔼_step)*one(𝔼_step)+ 2*μ*𝔼_step
            ℂ_check = λ*(one(𝔼_step) ⊗ one(𝔼_step)) + 2 * μ * one(SymmetricTensor{4, 3})
        elseif hyperModel == LargeDefs.neoHookeanCompressible
            invC = inv(C)
            S_check3 = μ*(one(C)-invC) + λ*(log(Jacobian))*invC
            ℂ_check = (μ - λ*log(Jacobian))*(otimesu(invC, invC) + otimesl(invC, invC) ) + λ * invC ⊗ invC
        end

        println("Second Piola Stress Check 2 :", norm(S_check3 - S_check1))
        println("Material Tangent Check :", norm(ℂ_check- ℂ))

        𝔼_lastStep = deepcopy(𝔼_step)
        S_hyd_array[step+1] = tr(S_check1)
        S_eff_array[step+1] = norm(S_check1 - 1/3*tr(S_check1)*one(S_check1))
        𝔼_array[step+1] = 𝔼_step[1]

        ########Find Cauchy Stress################
        τ = F⋅S_check1⋅F'
        #σ_array[step+1] = 1/det(F)*τ
        #println("Cauchy Stress = ", σ_array[step+1])
        #σ₁₁_array[step+1] = σ_array[step+1][1,1]
        σ₁₁_array[step+1] = S_check1[1,2]
        principalStretch = getPrincipalStretches(F)
        #if minimum(principalStretch) < 1
        #    λ₁[step+1] = minimum(principalStretch)
        #else
        #    λ₁[step+1] = maximum(principalStretch)
        #end
        λ₁[step+1] = 0.5*(∂u_∂X+∂u_∂X')[1,1]*100

        C1 = 2*𝔼_step + one(𝔼_step)
        Ic = getFirstInvariant(C1)
        IIc = getSecondInvariant(C1)
        println("Jacobian  =", Jacobian )
        ψ_array[step+1] = λ / 2 * log(Jacobian)^2 - μ * log(Jacobian) + μ / 2 * (Ic - 3)

        ∂u_∂X = ∂u_∂X + Δ∂u_∂X
    end
    plot(𝔼_array, σ₁₁_array)
    #plot(𝔼_array, S_hyd_array, label = "Hydrostatic Stress")
    #plot(𝔼_array, S_eff_array, label = "Norm Deviatoric Stress")
end
