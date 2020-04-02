# Recurrent inference machines for seismic imaging
# Same notation as Putzky and Welling paper (2017)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using JUDI.TimeModeling, LinearAlgebra, InvertibleNetworks
using PyPlot, Random, JLD, Flux

Random.seed!(11)

# Training data
D = load("/data/pwitte3/models/overthrust_images_test.jld")
ntest = length(D["m"])

# Crop images and models to 400 x 120 (must be evenly dividable by 4)
for j=1:ntest
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

maxiter = 4
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

# Load params
iter = 600
P_save = load(join(["../data/network_islim_params_iteration_", string(iter), ".jld"]))["P"]
for j=1:length(Params)
    Params[j].data = P_save[j].data
end


####################################################################################################
# Evaluate trained network for new data

# Step 1: Draw new m0, q_sim and d
i = 99 #randperm(ntest)[1]
x = D["dm"][i]
m0 = D["m0"][i]
for k=1:num_simsource
    q_sim[:,k] = ricker_wavelet(time, dt, f0) * randn(Float32, 1)[1]/sqrt(1f0*num_simsource)
end

# Set up Jacobian
J.model.m = m0
J.source[1] = q_sim
Y = J*vec(x)    # Observed data

# Step 2: Compute predicted image
X0 = zeros(Float32, nx1, nx2, nx_in, batchsize)
X = forward(X0, Y, J)
x_ = X[:,:,1,1]

# Comparison to RTM
rtm = J'*vec(Y)

# Plot
x_ = x_ / norm(x_, 2)
x = x / norm(x, 2)
rtm = rtm / norm(rtm, 2)

figure(figsize=(5, 7))
subplot(3,1,1); imshow(reshape(x, model0.n)', cmap="gray", vmin=-2e-2, vmax=2e-2); title("True image")
subplot(3,1,2); imshow(reshape(rtm, model0.n)', cmap="gray", vmin=-2e-2, vmax=2e-2); title("RTM")
subplot(3,1,3); imshow(reshape(x_, model0.n)', cmap="gray", vmin=-2e-2, vmax=2e-2); title("i-SLIM (affine)")
