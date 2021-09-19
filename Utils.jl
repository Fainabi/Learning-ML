@enum ActivationFunction begin
    Linear
    Logistic
    ReLU
    Tanh
end

logistic(x) = 1 / (1 + exp(-x))
relu(x) = max(x, 0)

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

@enum LayerType begin
    Normal
    Convolutional
    MaxPooling
end