# Recurrent inference machines for seismic imaging
# Same notation as Putzky and Welling paper (2017)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using JUDI.TimeModeling, JUDI4Flux, LinearAlgebra, InvertibleNetworks
using PyPlot, Random, JLD, Flux
import Flux.Optimise.update!

# Training data
D = load("/data/pwitte3/models/overthrust_images_train.jld")
ntrain = length(D["m"])

# Crop images and models to 400 x 120 (must be evenly dividable by 4)
for j=1:ntrain
    D["m"][j] = D["m"][j][1:400, 1:120]
    D["m0"][j] = D["m0"][j][1:400, 1:120]
    D["dm"][j] = D["dm"][j][1:400, 1:120]
end

# Use one sample to set up operators
m = D["m"][1]
m0 = D["m0"][1]
dm = vec(D["dm"][1])

# Set up model structure
n = size(m0)
d = (25., 25.)
o = (0., 0.)

# Setup info and model structure
nsrc = 1
model0 = Model(n, d, o, m0)

# Set up receiver geometry
nxrec = 301
xrec = range(400f0, stop=9600f0, length=nxrec)
yrec = 0f0
zrec = range(250f0, stop=250f0, length=nxrec)

# receiver sampling and recording time
time = 2000f0   # receiver recording time [ms]
dt = 4f0    # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dt, t=time, nsrc=nsrc)

# Set up source geometry (cell array with source locations for each shot)
num_simsource = 21
xsrc = range(500f0, stop=9500f0, length=num_simsource)
ysrc = range(0f0, stop=0f0, length=num_simsource)
zsrc = range(20f0, stop=20f0, length=num_simsource)

# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dt, t=time)

# setup wavelet
f0 = 0.015f0     # MHz
wavelet = zeros(Float32, srcGeometry.nt[1], num_simsource)
for j=1:num_simsource
    wavelet[:, j] = ricker_wavelet(time, dt, f0) * randn(1)[1]/sqrt(num_simsource)
end
q = judiVector(srcGeometry, wavelet)
q_sim = zeros(Float32, size(wavelet))

# Set up info structure for linear operators
ntComp = get_computational_nt(srcGeometry, recGeometry, model0)
info = Info(prod(n), nsrc, ntComp)

####################################################################################################

# Return data as julia array
opt = Options(return_array=true)

# Setup operators
F0 = judiModeling(info, model0; options=opt)
Pr = judiProjection(info, recGeometry)
Ps = judiProjection(info, srcGeometry)

# Modeling operator and Jacobian
F0 = Pr*F0*Ps'  # initial model
J = judiJacobian(F0, q)

####################################################################################################

# Dimensions
nx1 = n[1]
nx2 = n[2]
nx_in = 32
nx_hidden = 64
batchsize = 1

#ny1 = recGeometry.nt[1]
#ny2 = nxrec
#ny_in = 1
#ny_hidden = 64

maxiter = 8
Ψ(η) = identity(η)

SLIM = Array{AffineCouplingLayerSLIM}(undef, maxiter)
Params = Array{Parameter}(undef, 0)

# i-SLIM coupling layer
for j=1:maxiter
    SLIM[j] = AffineCouplingLayerSLIM(nx1, nx2, nx_in, nx_hidden, batchsize, Ψ; logdet=false, permute=true)
    #SLIM[j] = ConditionalLayerSLIM(nx1, nx2, nx_in, nx_hidden, ny1, ny2, ny_channel, ny_hidden, batchsize; type="affine")
    global Params = cat(Params, get_params(SLIM[j]); dims=1)
end


function forward(X, Y, J)
    for j=1:maxiter
        X = SLIM[j].forward(X, Y, J)
    end
    return X
end

function backward(dX, X, Y, J)
    for j=maxiter:-1:1
        dX, X = SLIM[j].backward(dX, X, Y, J)[[1,3]]
    end
    return dX
end

# Objective function 
function loss(Y, J, x_true)
    
    # Initiliaze w/ zeros
    X0 = zeros(Float32, nx1, nx2, nx_in, batchsize)

    # Forward pass
    X_ = forward(X0, Y, J)

    # Residual and function value
    dX = zeros(Float32, nx1, nx2, nx_in, batchsize)
    dX[:,:,1:1,:] = X_[:,:,1:1,:] - reshape(x_true, nx1, nx2, 1, batchsize)
    f = .5f0*norm(dX)^2

    # Backward pass (set gradients)
    backward(dX, X_, Y, J)

    return f
end

####################################################################################################

# Get network parameters and overwrite w/ saved values
iter_start = 1
if iter_start > 1
    P_save = load(join(["../data/network_islim_iter_8_params_iteration_", string(iter_start), ".jld"]))["P"]
    for j=1:length(Params)
        Params[j].data = P_save[j].data
    end
end

if iter_start > 1
    opt = load(join(["../data/network_islim_iter_8_params_iteration_", string(iter_start), ".jld"]))["opt"]
else
    opt = Flux.ADAM(1f-3)
end

# Optimization parameters
train_iter = 1000
indices = randperm(ntrain)

# Training loop
for j=iter_start+1:train_iter

    # Draw image + velocity from training data
    i = indices[j]
    print("Source no: ", i, "\n")
    x_true = D["dm"][i]
    m0 = D["m0"][i]

    # Draw random source
    for k=1:num_simsource
        q_sim[:,k] = ricker_wavelet(time, dt, f0) * randn(Float32, 1)[1]/sqrt(1f0*num_simsource)
    end

    # Generate observed data on the fly
    J.model.m = m0
    J.source[1] = q_sim
    Y = J*vec(x_true)

    # Evaluate objective and gradients
    @time f = loss(Y, J, x_true)
    print("Iteration: ", j, "; f(x) = ", f, "\n")

    # Update weights
    for p in Params
        update!(opt, p.data, p.grad)
    end
    for j=1:length(SLIM)
        clear_grad!(SLIM[j])
    end

    # Save intermediate results
    if mod(j, 100) == 0
        save(join(["../data/network_islim_iter_8_params_iteration_", string(j), ".jld"]), "P", Params, "opt", opt)
    end
end


