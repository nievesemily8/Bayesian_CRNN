
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

#####################
#Extrapolation accuracy 

pSGLD_case1_truth= load("pSGLD_only75_truth")["ode_data_list"]
pSGLD_case1_extrap= load("pSGLD_extrapolated")["extra_data_list"]

node_case1_truth= load("simp_rxn_nODE_100_truthv4")["ode_data_list"]
#only take 25 of 75 time points from below
node_case1_extra= load("simp_rxn_nODE25_extrav4")["extrap_ode_data"]

MAPE_list_pSGLD=[]
#calculate MAPE for each n_exp_train
for i in 1:100
    MAPE_loss= mean(abs.(pSGLD_case1_extrap[i, :, 76:100].- pSGLD_case1_truth[i, :, 76:100])./pSGLD_case1_truth[i, :, 76:100])
    push!(MAPE_list_pSGLD, MAPE_loss)
end 

MAPE_list_nODE=[]
#calculate MAPE for each n_exp_train
for i in 1:100
    MAPE_loss= mean(abs.(node_case1_extra[i, :, 1:25].- node_case1_truth[i, :, 76:100])./node_case1_truth[i, :, 76:100])
    push!(MAPE_list_nODE, MAPE_loss)
end 

pSGLD_mean= mean(MAPE_list_pSGLD)
pSGLD_std= std(MAPE_list_pSGLD)

node_mean= mean(MAPE_list_nODE)
node_std=std(MAPE_list_nODE)

#check not extrapolated values 
node_simp_training= load("simp_rxn_nODE_75trainingv4")["pred_ode_data"]
MAPE_list_nodetraining=[]
#calculate MAPE for each n_exp_train
for i in 1:100
    MAPE_loss= mean(abs.(node_simp_training[i, :, 1:75].- node_case1_truth[i, :, 1:75])) #./node_case1_truth[i, :, 1:75])
    push!(MAPE_list_nodetraining, MAPE_loss)
end 


training_loss_pSGLD=[]
for i in 1:100
    MAPE_loss= mean(abs.(pSGLD_case1_extrap[i, :, 1:75].- pSGLD_case1_truth[i, :, 1:75])) #./pSGLD_case1_truth[i, :,1:75])
    push!(training_loss_pSGLD, MAPE_loss)
end 



#Make extrapolation compare figure

species = ["A", "B", "C", "D", "E" ];
list_plt = []
ns=5

datasize=100
tstep = 0.4;
tspan = Float32[0.0, datasize * tstep];
tsteps = range(tspan[1], tspan[2], length=datasize);
n_exp_fig=2
for j in 1:ns
ode_data = node_case1_truth[n_exp_fig, :, :]
plt = Plots.scatter(tsteps, ode_data[j,:],
markercolor=:transparent,
label="Data",
framestyle=:box, color =:purple, alpha = 0.5)
if j == 1
        plot!(plt, legend=true, framealpha=0, dpi=300)
    else
        plot!(plt, legend=false, dpi=300)
end
pSGLD= pSGLD_case1_extrap[n_exp_fig, :, :]
node_training= node_simp_training[n_exp_fig, :, 1:75]
node_extra= node_case1_extra[n_exp_fig, :, 1:25]
node_combined= hcat(node_training, node_extra)
plot!(plt, tsteps, color = :purple, alpha = 1, pSGLD[j,:], label="CRNN-ODE", dpi=300)
plot!(plt, tsteps, color = :green, alpha = 1, node_combined[j,:], label="Neural-ODE", linewidth=2, dpi=300)
plot!([30.0], seriestype="vline", linestyle= :dash, label="Training Boundary", linewidth=2, color= :black, dpi=300)
plot!(xlabel="t (s)", ylabel="[" * species[j]* "]", dpi=300)


push!(list_plt, plt)

end

plt_all = plot(list_plt..., size=(850, 700), dpi=300)
display(plt_all)


#RNN Extrapolation 

rnn= load("simp_rxn_RNN_full100")["extrap_LSTM_data"]
rnn_true= load("simp_rxn_trueode_for_RNN")["ode_data_list"]

#calculate MAPE 
MAPE_list_RNN=[]

#just extrapolated time points
for i in 1:100
    MAPE_loss= mean(abs.(rnn[i, :, 76:100].- rnn_true[i, :, 76:100])./rnn_true[i, :, 76:100])
    push!(MAPE_list_RNN, MAPE_loss)
end 

RNN_mean= mean(MAPE_list_RNN) #90.93%
RNN_std= std(MAPE_list_RNN) #58.83

#Make Figure
species = ["A", "B", "C", "D", "E" ];
list_plt = []
ns=5
n_exp_fig=2
for j in 1:ns
    ode_data = rnn_true[n_exp_fig, :, :]
    plt = Plots.scatter(tsteps, ode_data[j,:],
    markercolor=:transparent,
    label="Data",
    framestyle=:box, color =:purple, alpha = 0.5)
    if j == 1
            plot!(plt, legend=true, legendfontsize=12, ylabel="Concentration", framealpha=0)
        else
            plot!(plt, legend=false)
    end
    pSGLD= rnn_true[n_exp_fig, :, :]
    rnn_data= rnn[n_exp_fig, :, :]
    
   
    plot!(plt, tsteps, color = :blue, alpha = 1, pSGLD[j,:], label="CRNN-ODE", linewidth=2, dpi=300)
    plot!(plt, tsteps, color = :red, alpha = 1, rnn_data[j,:], label="LSTM", linewidth=2, dpi=300)
    plot!([30.0], seriestype="vline", linestyle= :dash, label="Training Boundary", linewidth=2, color= :black, dpi=300)
    plot!(xlabel="t (s)", title="" * species[j]* "", dpi=300)
    
    
    push!(list_plt, plt)
    
end
plt_layout= @layout[a b c d e]
plt_all = plot(list_plt..., layout=plt_layout, size=(2000, 500), leftmargin=10mm, bottommargin=9mm,  dpi=300)
display(plt_all)
