
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
# Arguments
is_restart = false;
p_cutoff = 0.0;
n_epoch = 60000;
n_plot = 100;
opt = ADAMW(0.001, (0.9, 0.999), 1.f-8);
datasize = 75;
tstep = 0.4;
n_exp_train = 200;
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
xi_list=zeros(Float32, (n_exp, ns, datasize));
std_list = [];

function max_min(ode_data)
    return maximum(ode_data, dims=2) .- minimum(ode_data, dims=2) .+ lb
end

tstep_matrix= zeros(Float32, (ns, datasize));
global counter=0
for i in tsteps
    counter+=1
    if counter!=1
        tstep_matrix[:, counter].= i
    end
end 

xi_list6=zeros(Float32, (n_exp, ns+1, datasize));
for i in 1:n_exp
    x_list=[]
    u0 = u0_list[i, :];
    tstep_matrix[:, 1]= u0
    init_cond_matrix= repeat(u0, 1, datasize)
    
    xi_list6[i, 1:ns, :]= init_cond_matrix
    xi_list6[i, ns+1, :]=tsteps
   
    prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k);
    ode_data = Array(solve(prob_trueode, alg, saveat=tsteps));
    ode_data += randn(size(ode_data)) .* ode_data .* noise
    ode_data_list[i, :, :] = ode_data
    push!(std_list, max_min(ode_data));
end
y_std = maximum(hcat(std_list...), dims=2);
u0_list= u0_list'
#transform testing data into correct shape for flux
ode_data_list=permutedims(ode_data_list, [2, 3, 1]) #should be instead a vector of seq length containing matrix (features, samples), use collect(eachslice(x, dims=3))

ode_data_vec= collect(eachslice(ode_data_list, dims=2))  #convert to vector of seq length containing matrix (features, samples)

xi_list= permutedims( xi_list,[2, 3, 1])
xi_vec= collect(eachslice(xi_list, dims=2))

xi_list6= permutedims( xi_list6,[2, 3, 1])
xi_vec6= collect(eachslice(xi_list6, dims=2))

#here x is the timesteps but first x is the initial condition of each species **
data = Flux.DataLoader((xi_vec6, ode_data_vec), batchsize=10)


#input size, hidden dimension size 
rnn=Flux.GRUCell(5,10)

#model= Chain(LSTM(5 => 50), LSTM(50=>50), Dense(50 => 50), Dense(50=>50, tanh), Dense(50=>5))
#could also add another input for timestep so NN goes from 6 to 5
#add in another LSTM layer 
model= Chain(LSTM(6 => 200), LSTM(200=>200), Dense(200=>5))  #middle layer is new 

function loss(model, x,y)
    Flux.reset!(model) #reset between batches 
    
    sum(mae(model(xi), yi) for (xi, yi) in zip(x, y))
end 

opt = ADAM(0.001)
cb = function () #callback function to observe training
  
end


cb()
using ChainRules
loss_history= Float32[]

opt_state=Flux.setup(Adam(0.001), model)

for epoch in 1:2000
    Flux.train!(model, data, opt_state) do model, x, y 
        err= loss(model, x, y)
        ChainRules.ignore_derivatives()do 
            push!(loss_history, err)
        end 
        @show err
        return err
    end
end 

#Predictions using model
function predict_timecourse(model, n_exp, datasize)
    pred_matrix= zeros(Float32, (ns, datasize));
    x=xi_list6[:,:,n_exp]
    Flux.reset!(model)
    [pred_matrix[:,i]= model(x[:,i]) for i in range(1,datasize)]
    return pred_matrix
end 

pm1= predict_timecourse(model, 1, 75)
pm2= predict_timecourse(model, 2, 75)

plot(pm2[4,:])
plot!(ode_data_list[4, :, 2])
plot!(pm1[4,:])
plot!(ode_data_list[4, :, 1])

plot(pm2[1,:])
plot!(ode_data_list[1, :, 2])
plot!(pm1[1,:])
plot!(ode_data_list[1, :, 1])

plot(pm2[5,:])
plot!(ode_data_list[5, :, 2])
plot!(pm1[5,:])
plot!(ode_data_list[5, :, 1])

plot(pm2[2,:])
plot!(ode_data_list[2, :, 2])
plot!(pm1[2,:])
plot!(ode_data_list[2, :, 1])


function extra_timecourse(model, n_exp, datasize)
    pred_matrix= zeros(Float32, (ns, datasize));
    x_matrix= zeros(Float32, (ns+1, datasize));
    #Make xi list using initial conditions from n_exp 
    u0= u0_list[:, n_exp]

    tspan_extra = Float32[0.0, datasize * tstep];
    tsteps_extra = range(tspan_extra[1], tspan_extra[2], length=datasize);
    init_cond_matrix= repeat(u0, 1, datasize)
    
    x_matrix[1:ns, :]= init_cond_matrix
    x_matrix[ns+1, :]=tsteps_extra

    #Input to RNN model
    Flux.reset!(model)
    [pred_matrix[:,i]= model(x_matrix[:,i]) for i in range(1,datasize)]
    return pred_matrix
end 

#Extrapolate past time points used for training for each experiment 
extrap_LSTM_data = zeros(Float64, (n_exp_train, ns, 100));
for i in 1:n_exp_train
    pred= extra_timecourse(model, i, 100)
    extrap_LSTM_data[i, :, :]=pred
end 
JLD2.@save "simp_rxn_RNN_full100" extrap_LSTM_data

#Save true ODE data with datasize=100
datasize=100
tspan = Float32[0.0, datasize * tstep];
tsteps = range(tspan[1], tspan[2], length=datasize);
ode_data_list = zeros(Float32, (n_exp_train, ns, datasize));

for i in 1:n_exp_train
    u0 = u0_list[:,i]
    prob_trueode = ODEProblem(trueODEfunc, u0, tspan, k);
    ode_data = Array(solve(prob_trueode, alg, saveat=tsteps));
    ode_data_list[i, :, :] = ode_data
    
end

JLD2.@save "simp_rxn_trueode_for_RNN" ode_data_list


