module Net

using LinearAlgebra, ProgressBars, CUDA
import Random

@enum ActivationFunction begin
    Linear
    Logistic
    ReLU
    Tanh
end

logistic(x) = 1 / (1 + exp(-x))
relu(x) = x>0 ? x : 0

deri_logistic(x) = begin
    log_val = logistic(x)
    log_val * (1 - log_val)
end
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
@enum PoolingFunction begin
    MaxPooling
    AveragePooling
end

"""
    Structure Definition
"""
# Node struct
mutable struct LayerNodes  # all hyper parameters
    Activation::ActivationFunction
    Connection::Array  # nodes that connected at the former layer
    Weight
    Bias
end


# Layer struct
abstract type AbstractLayer end
mutable struct FullyConnectedLayer <: AbstractLayer
    NodeNum::Integer
    Nodes::LayerNodes
    δ

    z
    a
end
mutable struct ConvolutionalLayer <: AbstractLayer
    NodeNum::Integer
    Nodes::LayerNodes
    δ::Array

    Kernel::Array
    Stride::Integer
end
mutable struct MaxPoolingLayer <: AbstractLayer
    NodeNum::Integer
    Nodes::LayerNodes
    δ::Array

    
end

function propagate!(layer1::FullyConnectedLayer, layer2::FullyConnectedLayer)
    # println("z: $(size(layer2.z)), w: $(size(layer2.Nodes.Weight)), a: $(size(layer1.a))")
    layer2.z .= layer2.Nodes.Weight' * layer1.a .+ layer2.Nodes.Bias
    layer2.a .= layer2.z .|> acfun[layer2.Nodes.Activation]
end
function propagate(layer::ConvolutionalLayer, state)

end
function propagate(layer::MaxPoolingLayer, state)

end

function backpropagate!(layer1::FullyConnectedLayer, layer2::FullyConnectedLayer)
    # println("$(size(layer1.δ)), Weight: $(size(layer2.Nodes.Weight)), delta: $(size(layer2.δ)), z: $(size(layer1.z)) ")
    layer1.δ .= layer2.Nodes.Weight * layer2.δ .* broadcast(deri_acfun[layer1.Nodes.Activation], layer1.z)
end

function update!(layer1::FullyConnectedLayer, layer2::FullyConnectedLayer, α, batch_m)
    layer2.Nodes.Weight .-= layer1.a * transpose(layer2.δ) * α / batch_m
    layer2.Nodes.Bias .-= sum(layer2.δ, dims=2) * α / batch_m
end

# Net struct
abstract type AbstractNet end
mutable struct FullyConnectedNet <: AbstractNet
    dataset
    label

    Layers::Array  # [X, L1, L2, ..., Ln]
    α::AbstractFloat  # Learning Rate
    Max_iter::Integer
    Batch_size::Integer

    Loss::Function
    Deri_loss::Function
end

abstract type AbstractNetConfig end
mutable struct FullyConnedtedNetConfig <: AbstractNetConfig
    nodes
    activations
    α
    max_iter
    batch_size

    loss
    deri_loss
    mode
end

"""
    initialize_net(X, Y; kwargs)

...
# Arguments
- `X::Array`: samples
- `Y::Array`: labels
- `loss::Function`: loss function, this function just in normal vector-by-vector form
- `deri_loss::Function`: derivative of loss function, this function ___must___ be in matrix form
- `activations::Array{ActivationFunction}`: arrays that contains activation function types for every layer. 
- `nodes::Array`: number of nodes in every layer
- `max_iter::Number`: maximum time to iterate, default is 1e3
- `α::AbstractFloat`: learning rate, default is 0.001
- `mode::String`: "GPU" or "CPU"
...
"""
function initialize_net(X=[], Y=[]; kwargs...)

    config = FullyConnedtedNetConfig(
        [], # nodes
        [], # activations
        .01, # α
        1000, # max_iter
        1000, # batch_size
        (y_hat, y) -> (y_hat - y |> v -> dot(v, v)/2), # loss
        (y_hat, y) -> y_hat - y, # deri_loss
        "CPU" # mode
    )
    validate_set = (:nodes, :activations, :α, :max_iter, :loss, :deri_loss, :mode, :batch_size)
    
    for kw in kwargs
        if kw.first in validate_set
            :($(config).$(kw.first) = $(kw.second)) |> eval
        end
    end

    layer_num = length(config.nodes) + 1
    nodes = [size(X, 1), config.nodes...]
    activations = [Linear, config.activations...]
    α = config.α
    max_iter = config.max_iter
    batch_size = config.batch_size
    mode = config.mode
    loss = config.loss
    deri_loss = config.deri_loss

    weight = []
    bias = []
    z = []
    a = []
    δ = []
    layers = map(1:layer_num) do idx
        if idx > 1
            size_p = nodes[idx]
            size_f = nodes[idx-1]
            weight = Random.randn(size_f, size_p) / size_f
            bias = zeros(size_p, 1)
            δ = zeros(size_p, batch_size)
            z = zeros(size_p, batch_size)
            a = copy(z)

            if mode == "GPU"
                weight = cu(weight)
                bias = cu(bias)
                δ = cu(δ)
                z = cu(z)
                a = cu(a)
            end
            layernode = LayerNodes(activations[idx], [], weight, bias)
            FullyConnectedLayer(nodes[idx], layernode, δ, z, a)
            
            
        else
            """
                The first layer that takes several parts to match the
            batch stochastic gradient descent, would be a array of Layer
            objects.
            """
            map(1:Int(ceil(size(X, 2)/batch_size))) do batch_idx
                layernode = LayerNodes(activations[idx], [], weight, bias)
                if batch_idx < Int(ceil(size(X, 2)/batch_size))
                    batch = batch_size*(batch_idx-1)+1:batch_size*batch_idx
                else
                    batch = batch_size*(batch_idx-1)+1:size(X, 2)
                end
                a = X[:, batch]

                if mode == "GPU"
                    a = cu(a)
                end

                FullyConnectedLayer(nodes[idx], layernode, δ, z, a)
            end
        end
    end

    if mode == "CPU"
        net = FullyConnectedNet(X, Y, layers, α, max_iter, batch_size, config.loss, config.deri_loss)
    elseif mode == "GPU"
        net = FullyConnectedNet(cu(X), cu(Y), layers, α, max_iter, batch_size, config.loss, config.deri_loss)
    end
    return net
end

function train_once!(net::FullyConnectedNet)
    layer_len = length(net.Layers)
    batch_size = net.Batch_size
    if layer_len > 1
        @views for (batch_idx, layer) in enumerate(net.Layers[1])
            propagate!(layer, net.Layers[2])
            for idx = 2:layer_len-1
                propagate!(net.Layers[idx], net.Layers[idx+1])
            end

            if batch_idx < Int(ceil(size(net.label, 2)/batch_size))
                batch = batch_size*(batch_idx-1)+1:batch_size*batch_idx
            else
                batch = batch_size*(batch_idx-1)+1:size(net.label, 2)
            end

            net.Layers[end].δ = net.Deri_loss(net.Layers[end].a, net.label[:, batch])
            for idx = layer_len-1:-1:2
                backpropagate!(net.Layers[idx], net.Layers[idx+1])
            end
            for idx = layer_len:-1:3
                update!(net.Layers[idx-1], net.Layers[idx], net.α, net.Batch_size)
            end
            update!(layer, net.Layers[2], net.α, net.Batch_size)

        end
    end

    # println("Loss: $(get_loss(net))")
end

function train!(net::FullyConnectedNet)
    for epoch = tqdm(1:net.Max_iter)
        train_once!(net)
    end
end

function get_loss(net::FullyConnectedNet)
    val = net.dataset
    for layer in net.Layers[2:end]
        val = layer.Nodes.Weight' * val .+ layer.Nodes.Bias .|> acfun[layer.Nodes.Activation]
    end
    net.Loss(val, net.label) / size(net.label, 2)
end

function report(net::FullyConnectedNet)
    """
    This is a fully connected net.
    In this net, layer's nodes are
        $([layer.NodeNum for layer in net.Layers])
    with activation functions
        $([layer.Nodes.Activation for layer in net.Layers[2:end]])
    Learing rate is $(net.α)
    Batch size is $(net.Batch_size)
    Max epoch is $(net.Max_iter)
    Loss function is $(net.Loss)
    Derivative of loss function is $(net.Deri_loss)
    """ |> println
end
end