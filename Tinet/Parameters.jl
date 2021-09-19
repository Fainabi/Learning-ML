"""
    Parameters of w, b for all layer,
"""
abstract type AbstractParameters end
mutable struct Parameters <: AbstractParameters
    weight::Array
    bias::Array
end

newParameters(size_w::Tuple, size_b::Union{Tuple, Integer}) = 
    Parameters(rand(size_w...), rand(size_b...))

function toDevice!(para::Parameters, device)
    if device == :cuda
        para.bias = cu(para.bias)
        para.weight = cu(para.weight)
    elseif device == :cpu
        para.bias = Array(para.bias)
        para.weight = Array(para.weight)
    end
end

function update!(para::Parameters, δ::Parameters)
    para.bias .+= δ.bias
    para.weight .+= δ.weight
end

function clear!(para::Parameters)
    para.bias *= 0
    para.weight *= 0
end


"""
    Hyperparameters
"""
abstract type AbstractHyperparameters <: AbstractParameters end
struct ModelHyperparameters <: AbstractHyperparameters
    α::Real  # learning rate
end
struct FullyConnectedLayerHyperparameters <: AbstractHyperparameters
    # common content
    activation::Function  # activation function
    deri_activation::Function
end
struct ConvolutionalLayerHyperparameters <: AbstractHyperparameters
    kernel_num::Integer  # kernel numbers which equal to number of channels of the next layer
    kernel_shape::Tuple  # (x, y, z, ...)
                         # the tuple size is `n` which is same to `a` in prelayer
                         # and the channel numbers is another dim to perform a
                         # tensor product
    padding::Tuple  # same size with kernel_shape

    # common content
    activation::Function
    deri_activation::Function
end
struct MaxPoolingLayerHyperparameters <: AbstractHyperparameters
    kernel_shape::Tuple
    padding::Tuple
end


"""
    Parameters Union
"""
mutable struct LayerParameters <: AbstractParameters
    para::Parameters
    hyper::AbstractHyperparameters
end