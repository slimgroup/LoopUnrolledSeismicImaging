using InvertibleNetworks, NNlib, LinearAlgebra, Test
using PyPlot, Flux, Random
import Flux.Optimise.update!

# Random seed
#Random.seed!(100)

# Data
nx = 1
ny = 1
n_in = 2
batchsize = 100
hidden_factor = 64   # use this times as many weights for the hidden layer
k = 1   # kernel size
s = 1   # stride
p = 0   # padding
depth = 40

###################################################################################################
# Hyperbolic layer

# Layer
HL = Array{HyperbolicLayer}(undef, depth)
AN = ActNorm(n_in; logdet=true)
Params = get_params(AN)
for j=1:depth
    HL[j] = HyperbolicLayer(nx, ny, Int(n_in/2), batchsize, k, s, p; action="same",  α=1f0, hidden_factor=4)
    global Params = cat(Params, get_params(HL[j]); dims=1)
end

# Forward pass
function forward(X)
    X, logdet = AN.forward(X)
    X_prev, X_curr = tensor_split(X)
    for j=1:depth
        X_prev, X_curr = HL[j].forward(X_prev, X_curr)
    end
    X = tensor_cat(X_prev, X_curr)
    return X, logdet
end

# Inverse pass
function inverse(X)
    X_curr, X_new = tensor_split(X)
    for j=depth:-1:1
        X_curr, X_new = HL[j].inverse(X_curr, X_new)
    end
    X = tensor_cat(X_curr, X_new)
    X = AN.inverse(X)
    return X
end

# Backward pass
function backward(ΔX, X)
    ΔX_curr, ΔX_new = tensor_split(ΔX)
    X_curr, X_new = tensor_split(X)
    for j=depth:-1:1
        ΔX_curr, ΔX_new, X_curr, X_new = HL[j].backward(ΔX_curr, ΔX_new, X_curr, X_new)
    end
    ΔX = tensor_cat(ΔX_curr, ΔX_new)
    X = tensor_cat(X_curr, X_new)
    ΔX, X = AN.backward(ΔX, X)
    return ΔX, X
end

####################################################################################################

# Loss
function loss(X)
    Y, logdet = forward(X)
    f = -log_likelihood(Y) - logdet
    ΔY = -∇log_likelihood(Y)
    backward(ΔY, Y)
    return f
end

# Training
maxiter = 4000
opt = Flux.ADAM(1f-2)
fval = zeros(Float32, maxiter)
for j=1:maxiter

    # Evaluate objective and gradients
    X = sample_swirl(batchsize)
    fval[j] = loss(X)
    mod(j, 100) == 0 && (print("Iteration: ", j, "; f = ", fval[j], "\n"))
    
    # Update params
    for p in Params
        update!(opt, p.data, p.grad)
    end
    clear_grad!(Params)
end


####################################################################################################


# Evaluate network
test_size = 2000
X = sample_swirl(test_size)
Y = forward(X)[1]
Y_ = randn(Float32, nx, ny, n_in, test_size)
X_ = inverse(Y_)

# Plot
figure(figsize=[8,8])
ax1 = subplot(2,2,1); plot(X[1, 1, 1, :], X[1, 1, 2, :], "."); title(L"Data space: $x \sim \hat{p}_X$")
ax1.set_xlim([-15,15]); ax1.set_ylim([-15,18])
ax2 = subplot(2,2,2); plot(Y[1, 1, 1, :], Y[1, 1, 2, :], "g."); title(L"Latent space: $z = f(x)$")
ax2.set_xlim([-5, 5]); ax2.set_ylim([-5, 5])
ax3 = subplot(2,2,3); plot(X_[1, 1, 1, :], X_[1, 1, 2, :], "g."); title(L"Data space: $x = f^{-1}(z)$")
ax3.set_xlim([-15,15]); ax3.set_ylim([-15,18])
ax4 = subplot(2,2,4); plot(Y_[1, 1, 1, :], Y_[1, 1, 2, :], "."); title(L"Latent space: $z \sim \hat{p}_Z$")
ax4.set_xlim([-5, 5]); ax4.set_ylim([-5, 5])