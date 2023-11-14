#cd("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural ODE")
#Pkg.activate(".")


using OrdinaryDiffEq, Flux, Optim, Random, Plots
using Zygote
using ForwardDiff
using LinearAlgebra, Statistics
using ProgressBars, Printf
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
datasize = 100;
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

b0 = -10.0

function p2vec(p)
    w_b = p[1:nr] .+ b0;
    w_out = reshape(p[nr + 1:end], ns, nr);
    # w_out = clamp.(w_out, -2.5, 2.5);
    w_in = clamp.(-w_out, 0, 2.5);
    return w_in, w_b, w_out
end

function crnn!(du, u, p, t)
    w_in_x = w_in' * @. log(clamp(u, lb, ub));
    du .= w_out * @. exp(w_in_x + w_b);
end

u0 = u0_list[1, :]
p = randn(Float32, nr * (ns + 1)) .* 1.f-1;
# p[1:nr] .+= b0;
@show p
prob = ODEProblem(crnn!, u0, tspan, saveat=tsteps) #atol=atol, rtol=rtol

function predict_neuralode(u0, p)
    global w_in, w_b, w_out = p2vec(p);
    pred = clamp.(Array(solve(prob, alg, u0=u0, p=p;
                  maxiters=maxiters)), -ub, ub)
    return pred
end
# predict_neuralode(u0, p);

function display_p(p)
    w_in, w_b, w_out = p2vec(p);
    println("species (column) reaction (row)")
    println("w_in")
    show(stdout, "text/plain", round.(w_in', digits=3))

    println("\nw_b")
    show(stdout, "text/plain", round.(exp.(w_b'), digits=3))

    println("\nw_out")
    show(stdout, "text/plain", round.(w_out', digits=3))
    println("\n\n")
end
display_p(p)

l2_coeff= 0.0001 #tune this parameter, should be between 0 and 0.1
function loss_neuralode(p, i_exp)
    pred = predict_neuralode(u0_list[i_exp, :], p)
    loss = mae(ode_data_list[i_exp, :, :] ./ y_std, pred ./ y_std) #why do we divide by y std here? 
    loss+= l2_coeff*dot(p,p) #L2 regularization 
    return loss
end
# loss_neuralode(p, 1)

# Callback function to observe training

species = ["A", "B", "C", "D", "E"];
cbi = function (p, i_exp)
    ode_data = ode_data_list[i_exp, :, :]
    pred = predict_neuralode(u0_list[i_exp, :], p)
    list_plt = []
    for i in 1:ns
        plt = Plots.scatter(tsteps, ode_data[i,:],
                      markercolor=:transparent,
                      label="Exp",
                      framestyle=:box)
        plot!(plt, tsteps, pred[i,:], label="CRNN-ODE")
        plot!(xlabel="Time", ylabel="Concentration of " * species[i])

        if i == 1
            plot!(plt, legend=true, framealpha=0)
        else
            plot!(plt, legend=false)
        end

        push!(list_plt, plt)
    end
    plt_all = plot(list_plt...)
    #png(plt_all, string("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/ChemicalReaction/figs/i_exp_", i_exp))
    return false
end

list_loss_train = []
list_loss_val = []
iter = 1
cb = function (p, loss_train, loss_val)

    global list_loss_train, list_loss_val, iter
    push!(list_loss_train, loss_train)
    push!(list_loss_val, loss_val)

    if iter % n_plot == 0
        display_p(p)

        @printf("min loss train %.4e val %.4e\n", minimum(list_loss_train), minimum(list_loss_val))

        list_exp = randperm(n_exp)[1:1];
        println("update plot for ", list_exp)
        for i_exp in list_exp
            cbi(p, i_exp)
        end

        plt_loss = plot(list_loss_train, xscale=:log10, yscale=:log10, label="train");
        plot!(plt_loss, list_loss_val, label="val");

        #png(plt_loss, "C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/ChemicalReaction/figs/loss");

        #@save "C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/ChemicalReaction//checkpoint/mymodel.bson" p opt list_loss_train list_loss_val iter
    end

    iter += 1;
end

#if is_restart
#    @load "C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/ChemicalReaction/checkpoint/mymodel.bson" p opt list_loss_train list_loss_val iter;
#    iter += 1;
#end

# opt = ADAMW(0.001, (0.9, 0.999), 1.f-5);

# pSLGD
beta = 0.9;#0.999; #0.9;
λ = 1e-6;#1e-3; #1e-8;
precond = zeros(length(p))
a = 0.0001; #0.0001; #0.001; try 0.0001 again see if params are smaller
b = 0.15;
γ = 0.005;

p_store = p


up_count = 1;

i_exp = 1;
epochs = ProgressBar(iter:n_epoch)
loss_epoch = zeros(Float32, n_exp);

function chemtrain(precond, p_store, up_count, noise)
    global p
for epoch in epochs
    #global p #is this resetting p?
    
    i_counter=1
    for i_exp in randperm(n_exp_train) #change this to be like a mini batch?

        grad = gradient(p) do x
            Zygote.forwarddiff(x) do x
                loss_neuralode(x, i_exp)
            end
        end

        #noise = sqrt(ϵ/m[i])*randn()
        weight_decay= 1/n_exp_train
        
        #∇L = grad[1] #[1]  #why this 1? Add a weight decay here???
        ∇L = grad[1]#+ (weight_decay.*p) #weight decay stops it from training 
        
        #Shouldn't be using momentum and RMSprop together??- RMSprop inherently has momentum 
        if epoch == 1 #should it be for epoch==1? or i_exp?
            precond[:] = ∇L .* ∇L
        else
            precond *= beta
            precond += (1 - beta) * (∇L .* ∇L)
        end
        m = λ .+ sqrt.(precond) #try without dot + or .sqrt
        ϵ = a * (b + up_count)^-γ
        
        #noise = ϵ * randn(length(p))
        #noise= sqrt.(ϵ./m).*randn(length(p))
        
        #noise=sqrt.(2*ϵ./m).*randn(length(p))/length(n_exp_train) #n_exp_train #produced instable ODe/ Nan results without division, still get instability  
        noise=sqrt.(2*ϵ./m).*randn(length(p))/n_exp_train #n_exp_train #produced instable ODe/ Nan results without division, still get instability  

        #p .= p - (0.5 * ϵ * ∇L ./ m +  noise)
        p .= p - (2.0* ϵ * ∇L ./ m +  noise) #changed this back to 0.5 from 2
        # update!(opt, p, grad[1])

        up_count = up_count + 1
        i_counter+=1
    end

    p_store = hcat(p_store, p)

    for i_exp in 1:n_exp
        loss_epoch[i_exp] = loss_neuralode(p, i_exp)
    end
    loss_train = mean(loss_epoch[1:n_exp_train]);
    loss_val = mean(loss_epoch[n_exp_train + 1:end]);
    set_description(epochs, string(@sprintf("Loss train %.4e val %.4e", loss_train, loss_val)))
    cb(p, loss_train, loss_val);
end

return p_store
end

p_store = chemtrain(precond, p_store, up_count, noise)


#JLD2.@save "C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/ChemicalReaction/Jul21Analysis/Noise0/WJ_Integrate_Chem_Data_SGLD_60000_noise_0.jld2" p_store w_in_react

#JLD2.@load "C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/ChemicalReaction/Jul21Analysis/Noise0/WJ_Integrate_Chem_Data_SGLD_60000_noise_0.jld2" p_store w_in_react

JLD2.@save "pSGLD_no_noise_pstorev1" p_store

p_store=load("pSGLD_no_noise_pstorev1")["p_store"]
############### PLOTS FOR NOISE: 0, 0.0005, 0.005 (low noise), 0.5
nsample = 1000

w_in_react = zeros(4, 5)
k1 = Float64[]
k2 = Float64[]
k3 = Float64[]
k4 = Float64[]

global w_in_coeff = zeros(4, 5)

for i in 1:nsample
    p = p_store[:, end - i]
    w_in, w_b, w_out = p2vec(p)

    append!(k1, exp(w_b[1]))
    append!(k2, exp(w_b[2]))
    append!(k3, exp(w_b[3]))
    append!(k4, exp(w_b[4]))

    w_in = round.(w_in', digits=3)
    global w_in_coeff = w_in_coeff + w_in

   for j in 1:4
       for k in 1:5
           if w_in[j, k] >= 0.0001
               w_in_react[j ,k] +=1
           end
       end
   end

end

w_in_coeff = (1/ nsample) * w_in_coeff

for j in 1:4
    for k in 1:5
        w_in_coeff[j, k] = w_in_coeff[j,k]/ (maximum(w_in_coeff[j,:]))
    end
end



############################ REACTANT PROBABILITY PLOTS############################

####################################REACTION 1##############################
Reaction3 = (1/nsample)* [w_in_react[1, 1], w_in_react[1, 2], w_in_react[1, 3], w_in_react[1, 4], w_in_react[1, 5]]
labels = ["Species A", "Species B", "Species C", "Species D", "Species E"]

bar(labels, Reaction3, left_margin = 5mm, top_margin = 5mm, right_margin = 5mm, color =:green, alpha = 0.5,  ylims = (0, 1.5),  framestyle = :box, ylabel = "Recovery Probability", legend= false, title = "Reaction 1")

f(x) = 1
p1_a = plot!(f, xlims = (0,5), color = :black, linewidth = 3, label = "", linestyle =:dash)



using StatsPlots, LaTeXStrings
pyplot()
####DENSITY PLOTS####
true_rate = 0.1

density(k1, linewidth = 3, top_margin = 5mm, right_margin = 5mm, ylims = (0, 250), label = "Recovered rate", ylabel="Density", color =:green,framestyle = :box, grid =:off, legend = :topright, left_margin = 5mm, bottom_margin = 5mm, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12, dpi=300 )

p2_a = plot!([abs(true_rate)-0.000001,abs(true_rate)+0.0000001],[0.0,150],lw=3,color=:black,label = string("True reaction rate"),linestyle = :dash, dpi=300)

l = @layout [a; b]

p3_a= plot(p1_a, p2_a,layout = l, dpi=300)

#savefig("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/ChemicalReaction/Jul21Analysis/Noise0/Noise_0_Reaction3_60000.pdf")

Score3 = w_in_coeff[1,:] .*  Reaction3

bar(labels, Score3, left_margin = 5mm, top_margin = 5mm, right_margin = 5mm, color =:green, alpha = 0.5,  ylims = (0, 1.5),  framestyle = :box, ylabel = "Score = Probability * Coeff Count", legend=false, title = "Reaction 1", dpi=300)

score_a= plot!(f, xlims = (0,5), color = :black, linewidth = 3, label = "Score = 1", linestyle =:dash, dpi=300)

#savefig("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/ChemicalReaction/Jul21Analysis/Noise0/Score_Noise_0_Reaction3_60000.pdf")


####################################REACTION 2##############################
Reaction4 = (1/nsample)* [w_in_react[2, 1], w_in_react[2, 2], w_in_react[2, 3], w_in_react[2, 4], w_in_react[2, 5]]
labels = ["Species A", "Species B", "Species C", "Species D", "Species E"]

bar(labels, Reaction4, left_margin = 5mm, top_margin = 5mm, right_margin = 5mm, color =:purple, alpha = 0.5,  ylims = (0, 1.5),  framestyle = :box,  ylabel = "Recovery Probability", legend=false, title = "Reaction 2")

f(x) = 1
p1_b = plot!(f, xlims = (0,5), color = :black, linewidth = 3, label = "", linestyle =:dash)


using StatsPlots, LaTeXStrings
pyplot()
####DENSITY PLOTS####
true_rate = 0.2

density(k2,linewidth = 3, top_margin = 5mm, right_margin = 5mm, ylims = (0, 250), label = "Recovered rate", ylabel="Density", color =:purple,framestyle = :box, grid =:off, legend = :topright, left_margin = 5mm, bottom_margin = 5mm, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12, dpi=300 )

p2_b = plot!([abs(true_rate)-0.000001,abs(true_rate)+0.0000001],[0.0, 150],lw=3,color=:black,label = string("True reaction rate"),linestyle = :dash, dpi=300)

l = @layout [a; b]

p3_b= plot(p1_b, p2_b, layout = l, dpi=300)

#savefig("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/ChemicalReaction/Jul21Analysis/Noise0/Noise_0_Reaction4_60000.pdf")

Score4 = w_in_coeff[2,:] .*  Reaction4

bar(labels, Score4, left_margin = 5mm, top_margin = 5mm, right_margin = 5mm, color =:purple, alpha = 0.5,  ylims = (0, 1.5),  framestyle = :box,  ylabel = "Score = Probability * Coeff Count", legend=false, title = "Reaction 2", dpi=300)

score_b= plot!(f, xlims = (0,5), color = :black, linewidth = 3, label = "Score = 1", linestyle =:dash, dpi=300)

#savefig("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/ChemicalReaction/Jul21Analysis/Noise0/Score_Noise_0_Reaction4_60000.pdf")

####################################REACTION 3##############################
Reaction2 = (1/nsample)* [w_in_react[3, 1], w_in_react[3, 2], w_in_react[3, 3], w_in_react[3, 4], w_in_react[3, 5]]
labels = ["Species A", "Species B", "Species C", "Species D", "Species E"]

bar(labels, Reaction2, left_margin = 5mm, top_margin = 5mm, right_margin = 5mm, color =:blue, alpha = 0.5,  ylims = (0, 1.5),  framestyle = :box, ylabel = "Recovery Probability", legend=false, title = "Reaction 3")

f(x) = 1
p1_c = plot!(f, xlims = (0,5), color = :black, linewidth = 3, label = "", linestyle =:dash)


using StatsPlots, LaTeXStrings
pyplot()
####DENSITY PLOTS####
true_rate = 0.13

density(k3, xlims = (0.12, 0.15), linewidth = 3, top_margin = 5mm, right_margin = 5mm, ylims = (0, 250), label = "Recovered rate", ylabel="Density", color =:blue,framestyle = :box, grid =:off, legend = :topright, left_margin = 5mm, bottom_margin = 5mm, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12, dpi=300 )

p2_c = plot!([abs(true_rate)-0.000001,abs(true_rate)+0.0000001],[0.0, 150],lw=3,color=:black,label = string("True reaction rate"),linestyle = :dash, dpi=300)

l = @layout [a; b]

p3_c= plot(p1_c, p2_c, layout = l, dpi=300)

#savefig("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/ChemicalReaction/Jul21Analysis/Noise0/Noise_0_Reaction2_60000.pdf")

Score2 = w_in_coeff[3,:] .*  Reaction2

bar(labels, Score2, left_margin = 5mm, top_margin = 5mm, right_margin = 5mm, color =:blue, alpha = 0.5,  ylims = (0, 1.5),  framestyle = :box, ylabel = "Score = Probability * Coeff Count", legend=false, title = "Reaction 3", dpi=300)

score_c= plot!(f, xlims = (0,5), color = :black, linewidth = 3, label = "Score = 1", linestyle =:dash, dpi=300)

#savefig("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/ChemicalReaction/Jul21Analysis/Noise0/Score_Noise_0_Reaction2_60000.pdf")

####################################REACTION 4##############################
Reaction1 = (1/nsample)* [w_in_react[4, 1], w_in_react[4, 2], w_in_react[4, 3], w_in_react[4, 4], w_in_react[4, 5]]
labels = ["Species A", "Species B", "Species C", "Species D", "Species E"]

bar(labels, Reaction1, left_margin = 5mm, top_margin = 5mm, right_margin = 5mm, color =:red, alpha = 0.5,  ylims = (0, 1.5),  framestyle = :box,  ylabel = "Recovery Probability", legend = false, title="Reaction 4")

f(x) = 1
p1_d = plot!(f, xlims = (0,5), color = :black, linewidth = 3, label = "", linestyle =:dash)


using StatsPlots, LaTeXStrings
pyplot()
####DENSITY PLOTS####
true_rate = 0.3

density(k4,linewidth = 3, top_margin = 5mm, right_margin = 5mm, ylims = (0, 250), label = "Recovered rate", ylabel="Density", color =:red,framestyle = :box, grid =:off, legend = :topright, left_margin = 5mm, bottom_margin = 5mm, foreground_color_legend = nothing, background_color_legend = nothing,  yguidefontsize = 14, xguidefontsize = 14,  xtickfont = font(12, "TimesNewRoman"), ytickfont = font(12, "TimesNewRoman"), legendfontsize = 12, dpi=300 )

p2_d = plot!([abs(true_rate)-0.000001,abs(true_rate)+0.0000001],[0.0, 150],lw=3,color=:black,label = string("True reaction rate"),linestyle = :dash, dpi=300)

l = @layout [a; b]

p3_d=plot(p1_d, p2_d, layout = l, dpi=300)

#savefig("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/ChemicalReaction/Jul21Analysis/Noise0/Noise_0_Reaction1_60000.pdf")

Score1 = w_in_coeff[4,:] .*  Reaction1

bar(labels, Score1, left_margin = 5mm, top_margin = 5mm, right_margin = 5mm, color =:red, alpha = 0.5,  ylims = (0, 1.5),  framestyle = :box, ylabel = "Score = Probability * Coeff Count", legend= false, title = "Reaction 4", dpi=300)

score_d= plot!(f, xlims = (0,5), color = :black, linewidth = 3, label = "Score = 1", linestyle =:dash, dpi=300)

#savefig("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/ChemicalReaction/Jul21Analysis/Noise0/Score_Noise_0_Reaction1_60000.pdf")

#####################RETRODICTED PLOTS##################
list_plt = []


for j in 1:ns
ode_data = ode_data_list[1, :, :]
plt = Plots.scatter(tsteps, ode_data[j,:],
markercolor=:transparent,
label="Exp",
framestyle=:box, color =:purple, alpha = 0.5)
if j == 1
           plot!(plt, legend=true, framealpha=0)
     else
           plot!(plt, legend=false)
     end

for i in 1:500

  if i == 1
    p_pred = p_store[:, end-i];
    pred = predict_neuralode(u0_list[1, :], p_pred)

    plot!(plt, tsteps, color = :purple, alpha = 1, pred[j,:], label="CRNN-ODE")
    plot!(xlabel="t", ylabel="Concentration of " * species[j])

else
    p_pred = p_store[:, end-i];
    pred = predict_neuralode(u0_list[1, :], p_pred)

    plot!(plt, tsteps, color = :purple, alpha = 0.02, pred[j,:], label="")
    plot!(xlabel="t", ylabel="Concentration of " * species[j])
end

end

push!(list_plt, plt)

end

plt_all = plot(list_plt..., dpi=300)
display(plt_all)

#savefig(plt_all, string("C:/Users/16174/OneDrive/Desktop/Julia Lab/Bayesian Neural UDE/PDE/ChemicalReaction/figs/Case1_AllPlot.pdf") )


######## Make all subplots into one 

# score plots all combined 
score_layout= @layout[a b; c d]
plot(p3_a, p3_b, p3_c, p3_d, layout= score_layout, size=(1200, 900), topmargin= 20mm,  dpi=300)


plot(score_a, score_b, score_c, score_d, layout=score_layout, size=(900, 600), topmargin=20mm, bottommargin=20mm, dpi=300)