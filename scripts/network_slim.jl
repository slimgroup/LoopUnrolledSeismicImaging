# Example for HINT layer (Kruse et al, 2020)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

using InvertibleNetworks, LinearAlgebra, Test, Flux, Random
import Flux.Optimise.update!

# X dimensions
nx1 = 32
nx2 = 32
nx_channel = 32
nx_hidden = 64
batchsize = 2

# Y dimensions
ny1 = 128
ny2 = 64
ny_channel = 1
ny_hidden = 32

#######################################################################################################################
# Network

# Linear operator
Op = randn(Float32, ny1*ny2, nx1*nx2)
num_layer_operator = 2
num_layer_learned = 2
SLIM_A = Array{ConditionalLayerSLIM}(undef, num_layer_operator)
SLIM_B = Array{ConditionalLayerSLIM}(undef, num_layer_learned)
Params = Array{Parameter}(undef, 0)

# Conditional SLIM layer w/ modeling operator
for j=1:num_layer_operator
    SLIM_A[j] = ConditionalLayerSLIM(nx1, nx2, nx_channel, nx_hidden, ny1, ny2, ny_channel, ny_hidden, batchsize; type="affine")
    global Params = cat(Params, get_params(SLIM_A[j]); dims=1)
end

# Conditional SLIM layer w/ learned operator
for j=1:num_layer_learned
    SLIM_B[j] = ConditionalLayerSLIM(nx1, nx2, nx_channel, nx_hidden, ny1, ny2, ny_channel, ny_hidden, batchsize; type="learned")
    global Params = cat(Params, get_params(SLIM_B[j]); dims=1)
end

function forward(X, Y)
    logdet_cum = 0f0
    for j=1:num_layer_operator
        X, Y, logdet = SLIM_A[j].forward(X, Y, Op)
        logdet_cum += logdet
    end
    for j=1:num_layer_learned
        X, Y, logdet = SLIM_B[j].forward(X, Y, nothing)
        logdet_cum += logdet
    end
    return X, Y, logdet_cum
end

function inverse(X, Y)
    for j=num_layer_learned:-1:1
        X, Y = SLIM_B[j].inverse(X, Y, nothing)
    end
    for j=num_layer_operator:-1:1
        X, Y = SLIM_A[j].inverse(X, Y, Op)
    end
    return X, Y
end

function backward(ΔX, ΔY, X, Y)
    for j=num_layer_learned:-1:1
        ΔX, ΔY, X, Y = SLIM_B[j].backward(ΔX, ΔY, X, Y, nothing)
    end
    for j=num_layer_operator:-1:1
        ΔX, ΔY, X, Y = SLIM_A[j].backward(ΔX, ΔY, X, Y, Op)
    end
    return ΔX, ΔY, X, Y
end

function loss(X, Y)
    Zx, Zy, logdet = forward(X, Y)
    f = -log_likelihood(Zx) - log_likelihood(Zy) - logdet
    ΔZx = -∇log_likelihood(Zx)
    ΔZy = -∇log_likelihood(Zy)
    backward(ΔZx, ΔZy, Zx, Zy)
    return f
end

#######################################################################################################################

# Input image
X = glorot_uniform(nx1, nx2, nx_channel, batchsize)
Y = glorot_uniform(ny1, ny2, ny_channel, batchsize)

# Forward/inverse
Zx, Zy, logdet = forward(X, Y)
X_, Y_ = inverse(Zx, Zy)

@test isapprox(norm(X - X_)/norm(X), 0f0; atol=1f-2)
@test isapprox(norm(Y - Y_)/norm(Y), 0f0; atol=1f-2)

f = loss(X, Y)

# Training
maxiter = 10
opt = Flux.ADAM(1f-3)
fval = zeros(Float32, maxiter)

for j=1:maxiter

    # Evaluate objective and gradients
    X = rand(Float32, nx1, nx2, nx_channel, batchsize) .* 2f0 .+ 4f0
    Y = reshape(Op*reshape(X[:,:,1:1,:], :, batchsize), ny1, ny2, ny_channel, batchsize)

    fval[j] = loss(X, Y)
    mod(j, 10) == 0 && (print("Iteration: ", j, "; f = ", fval[j], "\n"))

    # Update params
    for p in Params
        update!(opt, p.data, p.grad)
    end
    clear_grad!(Params)
end