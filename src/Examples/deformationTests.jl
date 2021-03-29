using LargeDefs, LinearAlgebra, Tensors, PyPlot

function hyperElasticTest()
    #∂u_∂X = zeros(9)
    #∂u_∂X[1] = 1e-4
    E::Float64 = 10 #MPa
    ν::Float64 = 0.3
    λ = (ν*E)/((1+ν)*(1-2*ν))
    μ = E/(2*(1+ν))
    λ_μ = (λ, μ)
    ∂u_∂X_array = zeros(3,3)
    ∂u_∂X_array[1,1] = 0.5
    ∂u_∂X_array[2,2] = -.12
    ∂u_∂X_array[3,3] = -.12
    #∂u_∂X_array = [0.5000000000000002 4.163336342344337e-17 -1.3877787807814457e-17; -2.0816681711721685e-17 -0.11982540819326219 -1.5959455978986625e-16; 2.0816681711721685e-17 1.3877787807814457e-17 -0.11982540819326216]
    ∂u_∂X_total = get_∂u_∂X_Tensor(∂u_∂X_array)
    totalSteps = 2
    𝔼_lastStep = zero(Tensor{2,3, Float64})
    S_check2 = zero(Tensor{2,3, Float64})
    S_hyd_array = zeros(totalSteps)
    S_eff_array = zeros(totalSteps)
    𝔼_array = zeros(totalSteps)
    for step ∈ 1:totalSteps
        ∂u_∂X = (step/totalSteps)*∂u_∂X_total
        F = LargeDefs.getDeformationGradient(∂u_∂X)
        println("F = ", F)
        Jacobian = LargeDefs.getJacobianDeformationGradient(F)
        𝔼_step = LargeDefs.getGreenLagrangeStrain(F)
        println("E = ", 𝔼_step)
        #hyperModel = LargeDefs.saintVenant
        hyperModel = LargeDefs.neoHookeanCompressible
        #hyperModel = LargeDefs.neoHookean
        ##############################
        S_check1 = hyperModel.secondPiolaStress(𝔼_step, λ_μ)
        println("S = ", S_check1)
        #############################
        ℂ = hyperModel.materialTangentTensor(𝔼_step, λ_μ)
        ############################
        #println("ℂ = ", ℂ)

        if step == 1
            S_check2 = deepcopy(S_check1)
        else
            S_check2 += ℂ⊡(𝔼_step-𝔼_lastStep)

        end
        ###########################;
        println("Second Piola Stress Check 1 :", norm(S_check2- S_check1))
        if hyperModel == LargeDefs.saintVenant
            S_check3 = λ*tr(𝔼_step)*one(𝔼_step)+ 2*μ*𝔼_step
        elseif hyperModel == LargeDefs.neoHookeanCompressible
            J = det(F)
            C = LargeDefs.getRightCauchyTensor(F)
            invC = inv(C)
            S_check3 = 2 * ( μ/2 * one(𝔼_step)) + 2 *(λ/2 * log(J) *invC - μ/2 * invC)
        end

        println("Second Piola Stress Check 2 :", norm(S_check3- S_check1))

        𝔼_lastStep = deepcopy(𝔼_step)
        S_hyd_array[step] = tr(S_check1)
        S_eff_array[step] = norm(S_check1 - 1/3*tr(S_check1)*one(S_check1))
        𝔼_array[step] = 𝔼_step[1]
        #σ_check = LargeDeformations.convert2DTensorToMandel(1/J*F_tensor*S*F_tensor')
        #𝕔_mandel = zeros(9,9)
        #hyperModel.spatialTangentTensor!(𝕔_mandel, F, λ_μ)
        #println(𝕔_mandel)
        #𝕔_tensor = zeros(3,3,3,3)
        #hyperModel.spatialTangentTensor!(𝕔_tensor, F_tensor, λ_μ)
        #println(LargeDeformations.convert4DTensorToMandel(𝕔_tensor))

        #println("Spatial Tangent Tensor Check ", norm(𝕔_mandel - LargeDefs.convert4DTensorToMandel(𝕔_tensor))<1e-9)
    end
    #plot(𝔼_array, S_hyd_array, label = "Hydrostatic Stress")
    #plot(𝔼_array, S_eff_array, label = "Norm Deviatoric Stress")
end
