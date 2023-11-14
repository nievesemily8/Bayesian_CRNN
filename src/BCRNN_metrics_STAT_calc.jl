#########################
#Functions and data to make figures 

using OrdinaryDiffEq, DiffEqDevTools, Sundials, ParameterizedFunctions, Plots, ODE, ODEInterfaceDiffEq, LSODA, ModelingToolkit
gr()
using LinearAlgebra
#LinearAlgebra.BLAS.set_num_threads(1)
using Plots.PlotMeasures, JLD2
using Dates

using OrdinaryDiffEq, Flux, Optim, Random, Plots
using Zygote
using ForwardDiff
using LinearAlgebra, Statistics
using ProgressBars, Printf
using Flux.Optimise: update!, ExpDecay
using Flux.Losses: mae, mse, msle
using BSON: @save, @load
using Measures
Random.seed!(1234);
using JLD2
#using TransformVariables

par = load("params.jld2")

#33 rxns
ns=7#old 22
nr=6 #33 minus 11 so far
n_exp_train=200 #100
n_exp_test=20 #10
n_exp= n_exp_train+ n_exp_test
datasize=200; #50; #100;
tstep= 0.01; #0.01;
noise=0.0
n_epoch = 50000; #50000
n_plot = 100;
maxiters = 10000;

lb = 1.f-5;
ub = 1.f1;
alg= QNDF(); #Rosenbrock23();

atol = 1e-5;
rtol = 1e-2;

function sbml_model!(du, u, p, t)
    #reaction_mwa67e40c1_693d_4214_adc8_b2f2b71cef12 = p["reaction_mwa67e40c1_693d_4214_adc8_b2f2b71cef12_mw575f7f49_3663_47f1_b492_5b92c1c4345d"] * u[1] * u[2]
    #reaction_mwa67e40c1_693d_4214_adc8_b2f2b71cef12_part2= - p["reaction_mwa67e40c1_693d_4214_adc8_b2f2b71cef12_mw53c64fd3_9a1c_4947_a734_74a73554964c"] * u[3]
    #reaction_mw47dee769_daa0_4af4_978a_5ab17e504c2f = p["reaction_mw47dee769_daa0_4af4_978a_5ab17e504c2f_mwe49ede89_014e_40f2_acfd_0d1a0cd11fe7"] * u[6]
    reaction_mw877cd1e3_b48b_42e8_ab23_682dd893fd9d = p["reaction_mw877cd1e3_b48b_42e8_ab23_682dd893fd9d_mw8cfaf07f_dabe_45de_93cc_ef2c7fd31104"] * u[7] * u[7] 
    #reaction_mw877cd1e3_b48b_42e8_ab23_682dd893fd9d_part2= - p["reaction_mw877cd1e3_b48b_42e8_ab23_682dd893fd9d_mwab52aceb_4b19_4317_b2da_97ccbb973dab"] * u[4]



    reaction_mw413c6d45_ab23_4d3e_87b3_a8ed4629b923 = p["reaction_mw413c6d45_ab23_4d3e_87b3_a8ed4629b923_mw6b97a1ec_2cba_4bce_96f7_ec1d0fa2d16c"] * u[6]
    reaction_mwe8647e48_f4a9_40f4_9b32_f89ded572e01 = p["reaction_mwe8647e48_f4a9_40f4_9b32_f89ded572e01_mw50a0e884_a88c_46a7_b985_788868bc1029"] * u[1] * u[2] 
    reaction_mwe8647e48_f4a9_40f4_9b32_f89ded572e01_part2= - p["reaction_mwe8647e48_f4a9_40f4_9b32_f89ded572e01_mw2c88e0e2_e9c3_4e4c_bb2e_b0cd1f6420f4"] * u[3]
    reaction_mw65b9e026_bc6c_4c94_8b37_8b9acdf50c8a = p["reaction_mw65b9e026_bc6c_4c94_8b37_8b9acdf50c8a_mw95e2190d_8e39_419b_ad26_7cc141f7b87b"] * u[3]
    
    #reaction_mw1c9d29fa_bff4_4d2f_9d5f_f1791e4882a3 = p["reaction_mw1c9d29fa_bff4_4d2f_9d5f_f1791e4882a3_mw76d68ace_272d_4178_bba2_74dfdf260c70"] * u[3] * u[4] 
    #reaction_mw1c9d29fa_bff4_4d2f_9d5f_f1791e4882a3_part2= - p["reaction_mw1c9d29fa_bff4_4d2f_9d5f_f1791e4882a3_mwe37b936f_7781_4a01_b59b_96bd7db0c49e"] * u[5]
    #reaction_mw9544e67b_b6d0_4941_b7e0_ecd4f400a335 = p["reaction_mw9544e67b_b6d0_4941_b7e0_ecd4f400a335_mwb0744746_88a2_488e_a483_266747a044c6"] * u[10]
    #reaction_mwe9988e4a_083c_4f8e_b154_3e599c9307b0 = p["reaction_mwe9988e4a_083c_4f8e_b154_3e599c9307b0_mw26164d03_adda_4a21_b5ac_59e1d5a8d8ab"] * u[12]
    #reaction_mw177fa7b0_f0be_4c3e_8b47_2ac4e13159a2 = p["reaction_mw177fa7b0_f0be_4c3e_8b47_2ac4e13159a2_mw9f1a7f64_0b37_42df_9dd5_e1a44efdcbba"] * u[7] * u[9] 
    #reaction_mw177fa7b0_f0be_4c3e_8b47_2ac4e13159a2_part2= - p["reaction_mw177fa7b0_f0be_4c3e_8b47_2ac4e13159a2_mw366e6f17_4081_4cdc_9fa5_0aeb354d692c"] * u[15]
    #reaction_mwd189238c_e8f9_40be_b4ea_18a42bba1b4f = p["reaction_mwd189238c_e8f9_40be_b4ea_18a42bba1b4f_mw31eb851a_c381_419d_b694_f158b7f5cfb6"] * u[21]
    #reaction_mwad97bd5a_3dae_49d9_990b_2e6574740618 = p["reaction_mwad97bd5a_3dae_49d9_990b_2e6574740618_mwb6701ead_d3f2_4eb3_8b08_341cea49a4b2"] * u[9] * u[11] 
    #reaction_mwad97bd5a_3dae_49d9_990b_2e6574740618_part2= - p["reaction_mwad97bd5a_3dae_49d9_990b_2e6574740618_mwa5016035_3f9f_44fc_9f69_1d7a0155eb36"] * u[12]
    reaction_mwf8bacf1a_6c1a_49b6_b344_2d3bd404a735 = p["reaction_mwf8bacf1a_6c1a_49b6_b344_2d3bd404a735_mw9fe16c2b_7271_4e4f_b6de_c149721a3198"] * u[4] * u[4] 
    #reaction_mwf8bacf1a_6c1a_49b6_b344_2d3bd404a735_part2= - p["reaction_mwf8bacf1a_6c1a_49b6_b344_2d3bd404a735_mw74ea5b55_ead0_4b6f_8da0_fd1dcf7e231d"] * u[14]
    
    #reaction_mwc9b945cf_3a14_4bd9_b253_7064498c75e2 = p["reaction_mwc9b945cf_3a14_4bd9_b253_7064498c75e2_mw8cbe6595_6f16_4704_afe2_0dd043a175fa"] * u[14] * u[11] 
    #reaction_mwc9b945cf_3a14_4bd9_b253_7064498c75e2_part2=- p["reaction_mwc9b945cf_3a14_4bd9_b253_7064498c75e2_mw21d22acd_ddd4_4794_9700_52201984f75b"] * u[13]
    #reaction_mw75c6078f_fb76_4ca9_9fdd_e221e3ba57ad = p["reaction_mw75c6078f_fb76_4ca9_9fdd_e221e3ba57ad_mw81384973_14a0_4498_ab21_f70666d46d7f"] * u[13]
    #reaction_mwec4127b5_6bcf_4128_aff4_a6b3c470f690 = p["reaction_mwec4127b5_6bcf_4128_aff4_a6b3c470f690_mw1df2caba_8e41_4fe5_a1b5_7777eb98ed1c"] * u[14]
    #reaction_mw5c806b00_59a1_491e_99a1_2c932b2d5d7a = p["reaction_mw5c806b00_59a1_491e_99a1_2c932b2d5d7a_mw5a798f7a_b4eb_4a27_b413_4ff3956b90e9"] * u[17] * u[17] 
    #reaction_mw5c806b00_59a1_491e_99a1_2c932b2d5d7a_part2= - p["reaction_mw5c806b00_59a1_491e_99a1_2c932b2d5d7a_mw54178365_18c1_47e0_94ee_6b96582c52ef"] * u[16]
    #reaction_mw26fdabae_323b_4a78_b134_4c2eb70ea6a7 = p["reaction_mw26fdabae_323b_4a78_b134_4c2eb70ea6a7_mw1ff4e75e_fce5_4a7a_907b_05df4981f80b"] * u[16] * u[18] 
    #reaction_mw26fdabae_323b_4a78_b134_4c2eb70ea6a7_part2= - p["reaction_mw26fdabae_323b_4a78_b134_4c2eb70ea6a7_mw8b269d52_eda9_4dd1_8616_ebcf29c971fa"] * u[19]
    #reaction_mwc38a99c8_74cf_49f2_a16b_f6610ca1a0a7 = p["reaction_mwc38a99c8_74cf_49f2_a16b_f6610ca1a0a7_mwa0806e7a_a90d_4187_9c37_6d9ea569a447"] * u[21] * u[17] 
    #reaction_mwc38a99c8_74cf_49f2_a16b_f6610ca1a0a7_part2= - p["reaction_mwc38a99c8_74cf_49f2_a16b_f6610ca1a0a7_mw95cb9071_56e2_447d_b7c7_59ac96baa623"] * u[20]
    #reaction_mw45d92b79_0656_4795_87d0_7a465949ca43 = p["reaction_mw45d92b79_0656_4795_87d0_7a465949ca43_mwba545ecf_c7d4_4a6c_8c47_9e91f052d5a9"] * u[17] * u[18] 
    #reaction_mw45d92b79_0656_4795_87d0_7a465949ca43_part2= - p["reaction_mw45d92b79_0656_4795_87d0_7a465949ca43_mw01c5ceef_57a1_4baa_b2cd_fd39e9588a10"] * u[22]
    #reaction_mw3b0c171c_6d60_41ca_8193_83cd5e6c188c = p["reaction_mw3b0c171c_6d60_41ca_8193_83cd5e6c188c_mw90b25c4b_ad1a_4ee5_ae20_c60451484516"] * u[19]
    #reaction_mwb71945c2_03a8_4fad_a995_e1caeee98525 = p["reaction_mwb71945c2_03a8_4fad_a995_e1caeee98525_mw7aba6db3_c7ec_4192_bb5e_0ac4b466c1a5"] * u[22]


   
    # Species:   id = mwbfcf6773_1915_432c_b1d2_1f246094cc74; name = pEGF-EGFR2; affected by kineticLaw
    du[1] = (1 / (p["compartment_mw1637dd35_5f09_4a8d_bb7f_58717cdf1612"])) * ((-1.0 * reaction_mwe8647e48_f4a9_40f4_9b32_f89ded572e01) +(-1.0 * reaction_mwe8647e48_f4a9_40f4_9b32_f89ded572e01_part2) + (1.0 * reaction_mw65b9e026_bc6c_4c94_8b37_8b9acdf50c8a)+(1.0 * reaction_mw413c6d45_ab23_4d3e_87b3_a8ed4629b923) )
    # Species:   id = mw13abe2a6_9905_40e5_8c23_3fc8834b572a; name = STAT3c; affected by kineticLaw
    du[2] = (1 / (p["compartment_mw1637dd35_5f09_4a8d_bb7f_58717cdf1612"])) * ((-1.0 * reaction_mwe8647e48_f4a9_40f4_9b32_f89ded572e01)+(-1.0 * reaction_mwe8647e48_f4a9_40f4_9b32_f89ded572e01_part2))

    # Species:   id = mw2fd710a6_7fe2_4484_bca6_59c187bade8b; name = pEGF-EGFR2-STAT3c; affected by kineticLaw
    du[3] = (1 / (p["compartment_mw1637dd35_5f09_4a8d_bb7f_58717cdf1612"])) * ((1.0 * reaction_mwe8647e48_f4a9_40f4_9b32_f89ded572e01)+(1.0 * reaction_mwe8647e48_f4a9_40f4_9b32_f89ded572e01_part2) + (-1.0 * reaction_mw65b9e026_bc6c_4c94_8b37_8b9acdf50c8a))

    # Species:   id = mwb6a9aa2c_62e7_410f_9c33_dbe36dfcc4af; name = pSTAT3c; affected by kineticLaw
    du[4] = (1 / (p["compartment_mw1637dd35_5f09_4a8d_bb7f_58717cdf1612"])) * ((1.0 * reaction_mw65b9e026_bc6c_4c94_8b37_8b9acdf50c8a)+(-1.0 * reaction_mwf8bacf1a_6c1a_49b6_b344_2d3bd404a735)+(-1.0 * reaction_mwf8bacf1a_6c1a_49b6_b344_2d3bd404a735 )) #+(-1.0 * reaction_mw1c9d29fa_bff4_4d2f_9d5f_f1791e4882a3) #+(-1.0 * reaction_mw1c9d29fa_bff4_4d2f_9d5f_f1791e4882a3_part2)) # + (-1.0 * reaction_mwad97bd5a_3dae_49d9_990b_2e6574740618)+(-1.0 * reaction_mwad97bd5a_3dae_49d9_990b_2e6574740618_part2) + (-1.0 * reaction_mwf8bacf1a_6c1a_49b6_b344_2d3bd404a735)+(-1.0 * reaction_mwf8bacf1a_6c1a_49b6_b344_2d3bd404a735_part2) + (-1.0 * reaction_mwf8bacf1a_6c1a_49b6_b344_2d3bd404a735)+ (-1.0 * reaction_mwf8bacf1a_6c1a_49b6_b344_2d3bd404a735_part2)+ (-1.0 * reaction_mw177fa7b0_f0be_4c3e_8b47_2ac4e13159a2)+(-1.0 * reaction_mw177fa7b0_f0be_4c3e_8b47_2ac4e13159a2_part2))

    # Species:   id = mw341082a0_8017_4cc7_9d00_b1211a196072; name = pEGF-EGFR2-pSTAT3c; affected by kineticLaw
    #du[5] = (1 / (p["compartment_mw1637dd35_5f09_4a8d_bb7f_58717cdf1612"])) * ((1.0 * reaction_mw1c9d29fa_bff4_4d2f_9d5f_f1791e4882a3))#+(1.0 * reaction_mw1c9d29fa_bff4_4d2f_9d5f_f1791e4882a3_part2))
    
    # Species:   id = mw4f575c55_7dff_45d7_94ad_cda9621d5b63; name = pSTAT3c-pSTAT3c; affected by kineticLaw
    du[5] = (1 / (p["compartment_mw1637dd35_5f09_4a8d_bb7f_58717cdf1612"])) * ((1.0 * reaction_mwf8bacf1a_6c1a_49b6_b344_2d3bd404a735))#+(1.0 * reaction_mwf8bacf1a_6c1a_49b6_b344_2d3bd404a735_part2) + (-1.0 * reaction_mwc9b945cf_3a14_4bd9_b253_7064498c75e2)+(-1.0 * reaction_mwc9b945cf_3a14_4bd9_b253_7064498c75e2_part2) + (-1.0 * reaction_mwec4127b5_6bcf_4128_aff4_a6b3c470f690))

    # Species:   id = mwa8f2e7b2_0927_4ab4_a817_dddc43bb4fa3; name = EGF-EGFR2; affected by kineticLaw
    du[6] = (1 / (p["compartment_mw1637dd35_5f09_4a8d_bb7f_58717cdf1612"])) * ((-1.0 * reaction_mw413c6d45_ab23_4d3e_87b3_a8ed4629b923)+ (1.0 * reaction_mw877cd1e3_b48b_42e8_ab23_682dd893fd9d))#((1.0 * reaction_mw877cd1e3_b48b_42e8_ab23_682dd893fd9d) + (1.0 * reaction_mw877cd1e3_b48b_42e8_ab23_682dd893fd9d_part2))

    # Species:   id = mw7eacabf9_d68c_491a_aba2_ec0809a8ecc8; name = EGF-EGFR; affected by kineticLaw

    du[7]= (1 / (p["compartment_mw1637dd35_5f09_4a8d_bb7f_58717cdf1612"]))*((-1.0 * reaction_mw877cd1e3_b48b_42e8_ab23_682dd893fd9d)+ (-1.0 * reaction_mw877cd1e3_b48b_42e8_ab23_682dd893fd9d))
end 

u0=zeros(ns)
u0[1] = 0.002##change this
u0[2] = 1.0
u0[3] = 0.0
u0[4]=0.0



tspan = Float64[0.0, datasize * tstep];

prob = ODEProblem(sbml_model!,u0,tspan, par)
#ode_data = Array(solve(prob, alg=QNDF(;autodiff=false), saveat=tsteps));
#sys = modelingtoolkitize(prob)

#sys = structural_simplify(sys)

#prob = ODEProblem(sys, [], (0,10.))

#sol = solve(prob, CVODE_BDF(), abstol=1 / 10^14, reltol=1 / 10^14)

#how to vary initial conditions for training examples 
#vary randomly 25% from original, but keep the ones that are very small, very small 
#match initial conditions to the examples in their paper, replicate results 

#issues: very small numbers on most species , identifiability 


######################## 
#Generate synthetic dataset with random initial conditions (for non-negative initial conditions)
#Maybe I need more variation in initial conditions 
u0_list= rand(Float64, (n_exp, ns));
#u0_list[:, 4].*= 1.f-2
#u0_list[:,2] .=1.f0
#u0_list[:, 3].= 0.f0

#u0_list[:, 19:22].= 0.f0

tspan = Float64[0.0, datasize * tstep];
tsteps = range(tspan[1], tspan[2], length=datasize);
ode_data_list = zeros(Float64, (n_exp, ns, datasize));
std_list = [];

function max_min(ode_data)
    return maximum(ode_data, dims=2) .- minimum(ode_data, dims=2) .+ lb
end

for i in 1:n_exp
    u0= u0_list[i, :]
    prob_trueode = ODEProblem(sbml_model!,u0,tspan, par);
    ode_data = Array(solve(prob_trueode, alg=QNDF(;autodiff=false), saveat=tsteps));
    #ode_data += randn(size(ode_data)) .* ode_data .* noise
    ode_data_list[i, :, :] = ode_data
    push!(std_list, max_min(ode_data));
end 

y_std = maximum(hcat(std_list...), dims=2);



#########################
#CRNN

b0 = -10.0


function p2vec(p)
    w_b = p[1:nr] .+ b0;
    w_out = reshape(p[nr + 1:end], ns, nr);
    w_in = -w_out;
    w_in .= relu.(w_in);
    #w_out = clamp.(w_out, -2.5, 2.5);
    #w_in = -w_out; # 0, 2.5);
    return w_in, w_b, w_out
end



p_store= load("STAT3_EGFR_toEGFR2_add_v1")["p_store"]

nsample = 1000

w_in_react = zeros(nr, ns)
k1 = Float64[]
k2 = Float64[]
k3 = Float64[]
k4 = Float64[]
k5= Float64[]
k6= Float64[]



global w_in_coeff = zeros(nr, ns)

for i in 1:nsample
    p = p_store[:, end - i]
    w_in, w_b, w_out = p2vec(p)

    append!(k1, exp(w_b[1]))
    append!(k2, exp(w_b[2]))
    append!(k3, exp(w_b[3]))
    append!(k4, exp(w_b[4]))
    append!(k5, exp(w_b[5]))
    append!(k6, exp(w_b[6]))
    

    w_in = round.(w_in', digits=3)
    global w_in_coeff = w_in_coeff + w_in

   for j in 1:nr
       for k in 1:ns
           if w_in[j, k] >= 0.0001
               w_in_react[j ,k] +=1
           end
       end
   end

end

################
#Calculate metrics for STAT3 without noise 


k1_true=10.0
k1_AD= mean(@.abs(k1 .- k1_true))
k1_std= std(@.abs(k1 .- k1_true))/k1_true
k1_percent_AD= (k1_AD/k1_true)*100

k2_true=5.5
k2_AD= mean(@.abs(k2 .- k2_true))
k2_std= std(@.abs(k2 .- k2_true))/ k2_true
k2_percent_AD= (k2_AD/k2_true)*100

k3_true=0.4
k3_AD= mean(@.abs(k3 .- k3_true))
k3_std= std(@.abs(k3 .- k3_true))/k3_true
k3_percent_AD= (k3_AD/k3_true)*100

k4_true=11.74
k4_AD= mean(@.abs(k4 .- k4_true))
k4_std= std(@.abs(k4 .- k4_true))/k4_true
k4_percent_AD= (k4_AD/k4_true)*100

k5_true=20.0
k5_AD= mean(@.abs(k5 .- k5_true))
k5_std= std(@.abs(k5 .- k5_true))/k5_true
k5_percent_AD= (k5_AD/k5_true)*100

k6_true=2.014
k6_AD= mean(@.abs(k6 .- k6_true))
k6_std= std(@.abs(k6 .- k6_true))/k6_true
k6_percent_AD= (k6_AD/k6_true)*100

##############
#Noisy STAT3, 0.05

noisy_p_store= load("STAT3_noise0.05_v1")["p_store"]

w_in_react = zeros(nr, ns)
noisy_k1 = Float64[]
noisy_k2 = Float64[]
noisy_k3 = Float64[]
noisy_k4 = Float64[]
noisy_k5 = Float64[]
noisy_k6 = Float64[]



global w_in_coeff = zeros(nr, ns)

for i in 1:nsample
    p = noisy_p_store[:, end - i]
    w_in, w_b, w_out = p2vec(p)

    append!(noisy_k1, exp(w_b[1]))
    append!(noisy_k2, exp(w_b[2]))
    append!(noisy_k3, exp(w_b[3]))
    append!(noisy_k4, exp(w_b[4]))
    append!(noisy_k5, exp(w_b[5]))
    append!(noisy_k6, exp(w_b[6]))
    

    w_in = round.(w_in', digits=3)
    global w_in_coeff = w_in_coeff + w_in

   for j in 1:nr
       for k in 1:ns
           if w_in[j, k] >= 0.0001
               w_in_react[j ,k] +=1
           end
       end
   end

end

noisy_k1_true=10.0
noisy_k1_AD= mean(@.abs(noisy_k1 .- noisy_k1_true))
noisy_k1_std= std(@.abs(noisy_k1 .- noisy_k1_true))/noisy_k1_true
noisy_k1_percent_AD= (noisy_k1_AD/noisy_k1_true)*100

noisy_k2_true=11.74
noisy_k2_AD= mean(@.abs(noisy_k2 .- noisy_k2_true))
noisy_k2_std= std(@.abs(noisy_k2 .- noisy_k2_true))/ noisy_k2_true
noisy_k2_percent_AD= (noisy_k2_AD/noisy_k2_true)*100

noisy_k3_true=2.014
noisy_k3_AD= mean(@.abs(noisy_k3 .- noisy_k3_true))
noisy_k3_std= std(@.abs(noisy_k3 .- noisy_k3_true))/noisy_k3_true
noisy_k3_percent_AD= (noisy_k3_AD/noisy_k3_true)*100

noisy_k4_true=0.4
noisy_k4_AD= mean(@.abs(noisy_k4 .- noisy_k4_true))
noisy_k4_std= std(@.abs(noisy_k4 .- noisy_k4_true))/noisy_k4_true
noisy_k4_percent_AD= (noisy_k4_AD/noisy_k4_true)*100

noisy_k5_true=20.0
noisy_k5_AD= mean(@.abs(noisy_k5 .- noisy_k5_true))
noisy_k5_std= std(@.abs(noisy_k5 .- noisy_k5_true))/noisy_k5_true
noisy_k5_percent_AD= (noisy_k5_AD/noisy_k5_true)*100

noisy_k6_true=5.5
noisy_k6_AD= mean(@.abs(noisy_k6 .- noisy_k6_true))
noisy_k6_std= std(@.abs(noisy_k6 .- noisy_k6_true))/noisy_k6_true
noisy_k6_percent_AD= (noisy_k6_AD/noisy_k6_true)*100
