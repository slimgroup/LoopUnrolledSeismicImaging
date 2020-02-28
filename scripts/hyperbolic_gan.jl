using InvertibleNetworks, LinearAlgebra, Test
using PyPlot, Flux, Random, JLD
import Flux.Optimise.update!

# Random seed
Random.seed!(100)

# Load MNIST
X = load("../mnist_training.jld")["X"]

# Data
nx = 32
ny = 32
n_in = 1
batchsize = 10
k = 3   # kernel size
s = 1   # stride
p = 1   # padding

# Network
nscales = 2
steps_per_scale = 10
H = NetworkHyperbolic(nx, ny, n_in, batchsize, nscales, steps_per_scale; α=1f-1, hidden_factor=8, ncenter=10)

#############################################################################################################
# Training

# Loss
function loss(H, X)
    Y, logdet = H.forward(X)
    f = -log_likelihood(Y) - logdet
    ΔY = -∇log_likelihood(Y)
    H.backward(ΔY, Y)
    return f
end

# Training
ntrain = 60000
maxiter = 1000
opt = Flux.ADAM(1f-2)
fval = zeros(Float32, maxiter)
Params = get_params(H)

for j=1:maxiter

    # Evaluate objective and gradients
    idx = randperm(ntrain)[1:batchsize]
    fval[j] = loss(H, X[:,:,:,idx])
    mod(j, 10) == 0 && (print("Iteration: ", j, "; f = ", fval[j], "\n"))
    
    # Update params
    for p in Params
        update!(opt, p.data, p.grad)
    end
    clear_grad!(H)
end

# Test
testsize = 1
idx = randperm(ntrain)[1]
Xi = X[:,:,:,idx:idx]
Yi = H.forward(Xi)[1]

Yi_ = randn(Float32, nx, ny, n_in, testsize)
Xi_ = H.inverse(Yi_)

figure(figsize=[8,8])
i=1
subplot(2,2,1); imshow(Xi[:,:,1,1]); title(L"Data space: $x \sim \hat{p}_X$")
subplot(2,2,2); imshow(Yi[:,:,1,1]); title(L"Latent space: $z = f(x)$")
subplot(2,2,3); imshow(Xi_[:,:,1,i]); title(L"Data space: $x = f^{-1}(z)$")
subplot(2,2,4); imshow(Yi_[:,:,1,i]); title(L"Latent space: $z \sim \hat{p}_Z$")

