
using Flux, Random, Plots
using DiffEqFlux
using DifferentialEquations
#using Zygote
#using ForwardDiff
using LinearAlgebra, Statistics
#using ProgressBars, Printf
using Flux.Optimise: update!, ExpDecay
using Flux.Losses: mae, mse
using BSON: @save, @load
using Measures
#using SciMLSensitivity 
using JLD2
Random.seed!(1234);

###################################
# Argments
is_restart = false;
p_cutoff = 0.0;
n_epoch = 60000;
n_plot = 100;
opt = ADAMW(0.001, (0.9, 0.999), 1.f-8);
datasize = 75;
tstep = 0.4;
n_exp_train = 100;
n_exp_test = 10;
n_exp = n_exp_train + n_exp_test;
noise = 0
#noise = 5.f-1;
ns = 5;
nr = 4;
k = Float32[0.1, 0.2, 0.13, 0.3];
alg = Tsit5();
atol = 1e-5;
rtol = 1e-2;

maxiters = 1000000;

lb = 1.f-5;
ub = 1.f1;
####################################

function trueODEfunc(dydt, y, k, t)
    dydt[1] = -2 * k[1] * y[1]^2 - k[2] * y[1];
    dydt[2] = k[1] * y[1]^2 - k[4] * y[2] * y[4];
    dydt[3] = k[2] * y[1] - k[3] * y[3];
    dydt[4] = k[3] * y[3] - k[4] * y[2] * y[4];
    dydt[5] = k[4] * y[2] * y[4];
end

# Generate data sets
u0_list = rand(Float32, (n_exp, ns));
u0_list[:, 1:2] .+= 2.f-1;
u0_list[:, 3:end] .= 0.f0;
tspan = Float32[0.0, datasize * tstep];
tsteps = range(tspan[1], tspan[2], length=datasize);
ode_data_list = zeros(Float32, (n_exp, ns, datasize));
std_list = [];

function max_min(ode_data)
    return maximum(ode_data, dims=2) .- minimum(ode_data, dims=2) .+ lb
end


for i in 1:n_exp
    u0 = u0_list[i, :];
    prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k);
    ode_data = Array(solve(prob_trueode, alg, saveat=tsteps));
    ode_data += randn(size(ode_data)) .* ode_data .* noise
    ode_data_list[i, :, :] = ode_data
    push!(std_list, max_min(ode_data));
end
y_std = maximum(hcat(std_list...), dims=2);
u0_list= u0_list'
#transform testing data into correct shape for neural ode 
ode_data_list=permutedims(ode_data_list, [2, 3, 1])

data = Flux.DataLoader((u0_list, ode_data_list), batchsize=10)

dudt = Chain(
             Dense(5,50,tanh),
             Dense(50, 50, tanh),
             Dense(50,5))
n_ode = NeuralODE(dudt,tspan,Tsit5(),saveat=tsteps) #removed tolerances?? 

#ps = Flux.params(n_ode)
#print(ps)
#function predict_n_ode(u0)
 #   n_ode(u0)
#end

#not including y_std for now
#maybe this should be loss for one batch??

#loss_n_ode() = mae(ode_data_list[:,:,:] , predict_n_ode()) #update this to be the same as the CRNN


function loss_n_ode(u0, y, n_ode)
    result=n_ode(u0)
    result= permutedims(result, [1, 3, 2])
    
    loss= mae(y, result)

return loss
end 

#data = Iterators.repeated((), 1000) ##what is this??
opt = ADAM(0.001)
cb = function () #callback function to observe training
  #display(loss_n_ode(data))
  # plot current prediction against data
  #cur_pred = predict_n_ode()
  #pl = scatter(t,ode_data[1,:],label="data")
  #scatter!(pl,t,cur_pred[1,:],label="prediction")
  #display(plot(pl))
end

# Display the ODE with the initial parameter values.
cb()
using ChainRules
loss_history= Float32[]
#for epoch in 1:10 
  #  Flux.train!(loss_n_ode, ps, data, opt, cb = cb)
#end 
opt_state=Flux.setup(Adam(0.001), n_ode)
for epoch in 1:1000
    #Flux.train!(Flux.params(n_ode), data, opt, cb = cb) do x, y 
    Flux.train!(n_ode, data, opt_state) do n_ode, x, y 
        err= loss_n_ode(x, y, n_ode)
        ChainRules.ignore_derivatives()do 
            push!(loss_history, err)
        end 
        @show err
        return err
    end
end 
#(x,y)-> loss_n_ode(n_ode, x, y),
#what to do with different inital conditions/ training examples??
#for epoch in 1:1000
    #Flux.train!(loss_n_ode, ps, data, opt, cb = cb) do x, y 
   # loss_n_ode(x, y)
#end 
#end

#JLD2.@save "NNode_nonoise_v2_300epoch_simplerxn" n_ode
#result.u[length(result.u)][[1], :]
#]test = Flux.params(n_ode)
#print(test)
#Plot results
prediction= n_ode(u0_list[:,1])
plot(prediction[1,:])
plot!(ode_data_list[1,:,1])


#Extrapolation 
pred_ode_data= zeros(Float64, (n_exp_train, ns, datasize));
extrap_ode_data = zeros(Float64, (n_exp_train, ns, datasize));

for i in 1:n_exp_train
    u0_i= u0_list[:,i]
    pred= Array(n_ode(u0_i))
    pred_ode_data[i, :, :]= pred
    u0_n= [pred[1, end],pred[2, end], pred[3, end], pred[4, end], pred[5,end]] 
    extra_pred=Array(n_ode(u0_n))
    extrap_ode_data[i, :, :]= extra_pred
end 

#=
datasize=100
tspan = Float32[0.0, datasize * tstep];
tsteps = range(tspan[1], tspan[2], length=datasize);
theta, nn= Flux.destructure(opt_state) #get trained parameters
extra_prob= ODEProblem((u,p ,t)->dudt(u), u0_list[:,1], tspan, theta) #not accepting any params??
extra_pred=solve(extra_prob, alg, saveat=tsteps)
=#
correct_order= permutedims(ode_data_list, [3, 1, 2])
#JLD2.@save "simple_rxn_nODE_truth" correct_order

JLD2.@save "simp_rxn_nODE_75trainingv4" pred_ode_data
JLD2.@save "simp_rxn_nODE25_extrav4" extrap_ode_data

datasize=100
tspan = Float32[0.0, datasize * tstep];
tsteps = range(tspan[1], tspan[2], length=datasize);
ode_data_list = zeros(Float32, (n_exp_train, ns, datasize));
std_list = [];



for i in 1:n_exp_train
    u0 = u0_list[:,i]
    prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k);
    ode_data = Array(solve(prob_trueode, alg, saveat=tsteps));
    #ode_data += randn(size(ode_data)) .* ode_data .* noise
    ode_data_list[i, :, :] = ode_data
    
end

JLD2.@save "simp_rxn_nODE_100_truthv4" ode_data_list