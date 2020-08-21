module NeuralNetwork

"""
    ActivationFunction

ActivationFunction contains several functions that used in neural network. 
"""
@enum ActivationFunction begin
    Linear
    Logistic
    ReLU
    Tanh
end
export ActivationFunction

"""
    Net

The neural network.
"""
mutable struct Net
    # training set and parameters
    X::Array
    Y::Array
    Nodes::Array
    Activations::Array

    # hyper parameters
    α::AbstractFloat
    max_iter::Number

    # method
    train::Function
    loss::Function
    deri_loss::Function
end

# variables
logistic(x) = 1 / (1 + exp(-x))
deri_logistic(x) = begin 
    log_val = logistic(x)
    log_val * (1 - log_val)
end
relu(x) = x>0 ? x : 0
deri_relu(x) = x>0 ? 1 : 0

deri_tanh(x) = 1 - tanh(x)^2

acfun = Dict(
    Linear::ActivationFunction => identity,
    Logistic::ActivationFunction => logistic,
    ReLU::ActivationFunction => relu,
    Tanh::ActivationFunction => tanh,
)

deri_acfun = Dict(
    Linear::ActivationFunction => (x) -> 1,
    Logistic::ActivationFunction => deri_logistic,
    ReLU::ActivationFunction => deri_relu,
    Tanh::ActivationFunction => deri_tanh,
)

# packages
import LinearAlgebra
export backpropagation

# APIs
# function initialize_net(X=[], Y=[]; loss=identity, deri_loss=identity, activations=[], nodes=[], α=0.1, max_iter=1e3)
#     net = Net(X, Y, nodes, activations, α, max_iter, identity, loss, deri_logistic)
#     net.train() = backpropagation(
#         net.X, net.Y,
#         # loss=net.loss, 
#         # deri_loss=net.deri_loss,
#         # activations=net.Activations,
#         # nodes=net.Nodes,
#         # α=net.α
#         # max_iter=net.max_iter
#         )
# end


"""
    backpropagation(X, Y; <keyword arguments>, max_iter=1e3, α=0.1)

Backpropagation NN. The accessible activation functions are `logistic`, `ReLU` and `tanh` which could

be inspected by NeuralNetwork.ActivationFunction.

...
# Arguments
- `X::Array`: samples
- `Y::Array`: labels
- `loss::Function`: loss function
- `deri_loss::Function`: derivative of loss function
- `activations::Array{ActivationFunction}`: arrays that contains activation function types for every layer. 
- `nodes::Array`: number of nodes in every layer
- `max_iter::Number`: maximum time to iterate, default is 1e3
- `α::AbstractFloat`: learning rate, default is 0.001
...

"""
function backpropagation(X, Y; loss, deri_loss, activations, nodes, α=0.001, max_iter=1e3, mode="CPU")
    validate_inputs(X, Y)
    if length(activations) != length(nodes)
        throw(ArgumentError("layer number not match"))
    end
    if mode != "CPU"
        throw(DomainError("only implement in cpu"))
    end
    
    Batch_size = 1000
    M = size(X, 2)
    dim_feature = size(X, 1)
    dim_label = size(Y, 1)

    δ = []
    z = []
    w = []
    b = []
    a = []
    layer_num = length(nodes)
    println("number of layers is $layer_num")
    println("number of traning data is $M")
    println("scale of NN is $dim_feature × $nodes")
    deri_loss_combine(sample) = deri_loss(sample[1:nodes[end]], sample[nodes[end]+1:end])
    loss_combine(sample) = loss(sample[1:nodes[end]], sample[nodes[end]+1:end])

    # initialize
    deri_activations = map(act -> deri_acfun[act], activations)
    activations = map(act -> acfun[act], activations)
    for layer = 1:layer_num
        if layer == 1
            # push!(w, ones(dim_feature, nodes[1]) / M)
            push!(w, randn(dim_feature, nodes[1]))
        else
            push!(w, randn(nodes[layer-1], nodes[layer]))  # zero initialization
        end
        push!(b, zeros(nodes[layer]))  # zero vector
        push!(z, 0)
        push!(δ, 0)  # just set a position
        push!(a, 0)
    end

    # train
    iteration = 0
    while iteration < max_iter 
        iteration += 1
        if iteration % 10 == 0
            println("epochs: $iteration/$max_iter\t")
        end
        
        resnorm = 0
        for batch = 1:Int64(ceil(M/Batch_size))
            if batch < ceil(M/Batch_size)
                batch_samples = X[:, (batch-1)*Batch_size+1:batch*Batch_size]
                batch_labels = Y[:, (batch-1)*Batch_size+1:batch*Batch_size]
            else
                batch_samples = X[:, (batch-1)*Batch_size+1:end]
                batch_labels = Y[:, (batch-1)*Batch_size+1:end]
            end
            # forward propagation
            a_layer = batch_samples
            for layer = 1:layer_num
                a[layer] = a_layer
                z_layer = w[layer]' * a_layer .+ b[layer]  # linear combination
                z[layer] = z_layer  # store
                a_layer = z_layer .|> activations[layer]  # map activation function
            end

            # backward propagation
            Res_samples = [a_layer; batch_labels]
            resnorm += mapslices(loss_combine, Res_samples, dims=[1]) |> sum
            if batch == ceil(M/Batch_size)
                resnorm = resnorm / M
                println("Resnorm at $iteration: $resnorm")
            end
            δ_layer = mapslices(deri_loss_combine, Res_samples, dims=[1]) |> transpose  # M times N_D
            for layer = layer_num:-1:1
                ∇σ = z[layer] .|> deri_activations[layer]  # N_i times M
                δ_layer = δ_layer .* transpose(∇σ)  # M times N_i
                # record ∇w
                ∇w = a[layer] * δ_layer / size(batch_samples, 2)  # here a[layer] = a^{[layer-1]}
                # update b
                b[layer] -= α * sum(transpose(δ_layer), dims=2) / size(batch_samples, 2)  # sum to column
                if layer > 1  # no need to calculate at the last layer, or input layer
                    δ_layer = δ_layer * transpose(w[layer])
                end
                # update w
                w[layer] -= α * ∇w
                
            end
        end
    end

    model(x) = begin
        for layer = 1:layer_num
            x = w[layer]' * x .+ b[layer] .|> activations[layer]
        end
        return x
    end
    labels_hat = []
    for idx = 1:M
        push!(labels_hat, model(X[:, idx]))
    end

    return model, labels_hat, w, b
end

# inner functions
function validate_inputs(X, Y)
    if X === [] || Y === [] 
        throw(ArgumentError("input matrix should not be empty"))
    elseif size(X, 2) != size(Y, 2)
        throw(DimensionMismatch("input samples and labels should have same number of rows")) 
    end
end

end