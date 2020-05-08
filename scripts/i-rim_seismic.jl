# Recurrent inference machines for seismic imaging
# Same notation as Putzky and Welling paper (2017)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using JUDI.TimeModeling, JUDI4Flux, LinearAlgebra, InvertibleNetworks
using PyPlot, Random, JLD, Flux
import Flux.Optimise.update!

# Training data
D = load("../data/overthrust_images_train.jld")
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
dt = 1f0    # receiver sampling interval [ms]

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
opt = Options(return_array=true, dt_comp=1f0)

# Setup operators
F0 = judiModeling(info, model0; options=opt)
Pr = judiProjection(info, recGeometry)
Ps = judiProjection(info, srcGeometry)

# Modeling operator and Jacobian
F0 = Pr*F0*Ps'  # initial model
J = judiJacobian(F0, q)

####################################################################################################

# Objective function 
function loss(L, J, d, η)
    
    # Initiliaze w/ zeros
    nx, ny = J.model.n
    η0 = zeros(Float32, nx, ny, 1, 1)
    s0 = zeros(Float32, nx, ny, L.L[1].C.k - 1, 1)

    # Forward pass
    η_, s_ = L.forward(η0, s0, d, J)

    # Residual and function value
    Δη = η_ - η
    f = .5f0*norm(Δη)^2

    # Backward pass (set gradients)
    L.backward(Δη, 0f0.*s0, η_, s_, d, J)

    return f
end

# Dimensions
n_in = 32
n_hidden = 64
batchsize = 1
maxiter = 2
Ψ(η) = identity(η)

# Unrolled loop
L = NetworkLoop(n[1], n[2], n_in, n_hidden, batchsize, maxiter, Ψ; type="HINT")

# Get network parameters and overwrite w/ saved values
iter_start = 1
P_curr = get_params(L)
# P_save = load(join(["network_params_iteration_", string(iter_start), ".jld"]))["P"]
# for j=1:length(P_curr)
#     P_curr[j].data = P_save[j].data
# end

# Optimization parameters
opt = Flux.ADAM(1f-3)
train_iter = 1000
indices = randperm(ntrain)

# Training loop
for j=iter_start+1:train_iter
    # Draw image + velocity from training data
    i = indices[j]
    print("Source no: ", i, "\n")
    η = D["dm"][i]
    m0 = D["m0"][i]

    # Draw random source
    for k=1:num_simsource
        q_sim[:,k] = ricker_wavelet(time, dt, f0) * randn(Float32, 1)[1]/sqrt(1f0*num_simsource)
    end

    # Generate observed data on the fly
    J.model.m = m0
    J.source[1] = q_sim
    d = J*vec(η)

    # Evaluate objective and gradients
    @time f = loss(L, J, d, η)
    print("Iteration: ", j, "; f(x) = ", f, "\n")

    # Update weights
    P = get_params(L)
    for p in P
        update!(opt, p.data, p.grad)
    end
    clear_grad!(L)

    # Save intermediate results
    if mod(j, 100) == 0
        P = get_params(L)
        save(join(["network_params_iteration_", string(j), ".jld"]), "P", P, "opt", opt)
    end
end

####################################################################################################
# Evaluate trained network for new data

# Step 1: Draw new m0, q_sim and d
i = randperm(ntrain)[1]
η = D["dm"][i]
m0 = D["m0"][i]
for k=1:num_simsource
    q_sim[:,k] = ricker_wavelet(time, dt, f0) * randn(Float32, 1)[1]/sqrt(1f0*num_simsource)
end

# Set up Jacobian
J.model.m = m0
J.source[1] = q_sim
d = J*vec(η)    # Observed data

# Step 2: Compute predicted image
η0 = zeros(Float32, n[1], n[2], 1, 1)   # zero initial guess
s0 = zeros(Float32, n[1], n[2], n_in-1, 1)
η_ = L.forward(η0, s0, d, J)[1]

# Comparison to RTM
rtm = J'*d

# Plot
η_ = η_ / norm(η_, 2)
η = η / norm(η, 2)
rtm = rtm / norm(rtm, 2)

figure(figsize=(6, 7))
subplot(3,1,1); imshow(reshape(η, model0.n)', cmap="gray", vmin=-2e-2, vmax=2e-2); title("True image")
subplot(3,1,2); imshow(reshape(rtm, model0.n)', cmap="gray", vmin=-2e-2, vmax=2e-2); title("RTM")
subplot(3,1,3); imshow(reshape(η_, model0.n)', cmap="gray", vmin=-2e-2, vmax=2e-2); title("i-RIM")
