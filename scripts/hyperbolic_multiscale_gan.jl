using InvertibleNetworks, LinearAlgebra, Test
using PyPlot, Flux, Random, JLD
import Flux.Optimise.update!

# Random seed
Random.seed!(100)

#############################################################################################################
# Utils

# Concatenate states Zi and final output
function cat_states(Z_save, X)
    Y = []
    for j=1:length(Z_save)
        Y = cat(Y, vec(Z_save[j]); dims=1)
    end
    Y = cat(Y, vec(X); dims=1)
    return Float32.(Y)  # convert to Array{Float32, 1}
end

# Split 1D vector in latent space back to states Zi
function split_states(Y, Z_dims)
    L = length(Z_dims)
    Z_save = Array{Array}(undef, L)
    count = 1
    for j=1:L
        Z_save[j] = reshape(Y[count: count + prod(Z_dims[j])-1], Z_dims[j])
        count += prod(Z_dims[j])
    end
    X = reshape(Y[count: count + prod(Z_dims[end])-1], Int.(Z_dims[end].*(.5, .5, 4, 1)))
    return Z_save, X
end

#############################################################################################################
# Data and network layers

# Load MNIST
X = load("/home/pwitte3/InvertibleNetworks/data/mnist_training.jld")["X"]

# Data
nx = 32
ny = 32
n_in = 1
batchsize = 10
k = 3   # kernel size
s = 1   # stride
p = 1   # padding
hidden_factor = 2
α = 0.1f0

# Network
steps_per_scale = 10

# Scale 1
H1 = Array{HyperbolicLayer}(undef, steps_per_scale)
for j=1:steps_per_scale
    H1[j] = HyperbolicLayer(16, 16, 2, batchsize, k, s, p; action="same", α=α, hidden_factor=hidden_factor)
end
Hs1 = HyperbolicLayer(16, 16, 2, batchsize, k, s, p; action="down", α=α, hidden_factor=hidden_factor)

# Scale 2
H2 = Array{HyperbolicLayer}(undef, steps_per_scale)
for j=1:steps_per_scale
    H2[j] = HyperbolicLayer(8, 8, 4, batchsize, k, s, p; action="same", α=α, hidden_factor=hidden_factor)
end
Hs2 = HyperbolicLayer(8, 8, 4, batchsize, k, s, p; action="down", α=α, hidden_factor=hidden_factor)

# Scale 3
H3 = Array{HyperbolicLayer}(undef, steps_per_scale)
for j=1:steps_per_scale
    H3[j] = HyperbolicLayer(4, 4, 8, batchsize, k, s, p; action="same", α=α, hidden_factor=hidden_factor)
end
Hs3 = HyperbolicLayer(4, 4, 8, batchsize, k, s, p; action="down", α=α, hidden_factor=hidden_factor)

# Save and dimensions
X_save = Array{Array}(undef, 3)
X_dims = Array{Tuple}(undef, 3)


#############################################################################################################
# Network passes

# Forward pass
function forward(X)

    # Initial transform to increase no. of channels
    logdet = 0f0
    X = wavelet_squeeze(X)
    X_prev, X_curr = tensor_split(X)

    # Scale 1
    for j=1:steps_per_scale
        X_prev, X_curr = H1[j].forward(X_prev, X_curr)
    end
    X_prev, X_curr = Hs1.forward(X_prev, X_curr)
    X_save[1] = X_prev
    X_dims[1] = size(X_prev)

    # Scale 2
    X_prev, X_curr = tensor_split(X_curr)
    for j=1:steps_per_scale
        X_prev, X_curr = H2[j].forward(X_prev, X_curr)
    end
    X_prev, X_curr = Hs2.forward(X_prev, X_curr)
    X_save[2] = X_prev
    X_dims[2] = size(X_prev)

    # Scale 3
    X_prev, X_curr = tensor_split(X_curr)
    for j=1:steps_per_scale
        X_prev, X_curr = H3[j].forward(X_prev, X_curr)
    end
    X_prev, X_curr = Hs3.forward(X_prev, X_curr)
    X_save[3] = X_prev
    X_dims[3] = size(X_prev)

    # Concatenate state from all scales
    X = cat_states(X_save, X_curr)
    return X, logdet
end

function inverse(X)
    X_save, X_curr = split_states(X, X_dims)

    # Scale 3
    X_prev, X_curr = Hs3.inverse(X_save[3], X_curr)
    for j=steps_per_scale:-1:1
        X_prev, X_curr = H3[j].inverse(X_prev, X_curr)
    end
    X_curr = tensor_cat(X_prev, X_curr)

    # Scale 2
    X_prev, X_curr = Hs2.inverse(X_save[2], X_curr)
    for j=steps_per_scale:-1:1
        X_prev, X_curr = H2[j].inverse(X_prev, X_curr)
    end
    X_curr = tensor_cat(X_prev, X_curr)

    # Scale 1
    X_prev, X_curr = Hs1.inverse(X_save[1], X_curr)
    for j=steps_per_scale:-1:1
        X_prev, X_curr = H1[j].inverse(X_prev, X_curr)
    end
    X_curr = tensor_cat(X_prev, X_curr)

    return X_curr
end

# Current sample
Xi = X[:,:,:,1:10]
Y, logdet = forward(Xi)
Xi_ = inverse(Y)

#############################################################################################################
# Training

# # Loss
# function loss(H, X)
#     Y, logdet = H.forward(X)
#     f = -log_likelihood(Y) - logdet
#     ΔY = -∇log_likelihood(Y)
#     H.backward(ΔY, Y)
#     return f
# end

# # Training
# ntrain = 60000
# maxiter = 1000
# opt = Flux.ADAM(1f-2)
# fval = zeros(Float32, maxiter)
# Params = get_params(H)

# for j=1:maxiter

#     # Evaluate objective and gradients
#     idx = randperm(ntrain)[1:batchsize]
#     fval[j] = loss(H, X[:,:,:,idx])
#     mod(j, 10) == 0 && (print("Iteration: ", j, "; f = ", fval[j], "\n"))
    
#     # Update params
#     for p in Params
#         update!(opt, p.data, p.grad)
#     end
#     clear_grad!(H)
# end

# # Test
# testsize = 1
# idx = randperm(ntrain)[1]
# Xi = X[:,:,:,idx:idx]
# Yi = H.forward(Xi)[1]

# Yi_ = randn(Float32, nx, ny, n_in, testsize)
# Xi_ = H.inverse(Yi_)

# figure(figsize=[8,8])
# i=1
# subplot(2,2,1); imshow(Xi[:,:,1,1]); title(L"Data space: $x \sim \hat{p}_X$")
# subplot(2,2,2); imshow(Yi[:,:,1,1]); title(L"Latent space: $z = f(x)$")
# subplot(2,2,3); imshow(Xi_[:,:,1,i]); title(L"Data space: $x = f^{-1}(z)$")
# subplot(2,2,4); imshow(Yi_[:,:,1,i]); title(L"Latent space: $z \sim \hat{p}_Z$")

