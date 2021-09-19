import LinearAlgebra: dot
# =============== Activations ======================
@enum ActivationFunction begin
    Linear
    Logistic
    ReLU
    Tanh
end

logistic(x) = 1 / (1 + exp(-x))
relu(x) = max(x, 0)

deri_identity(x) = 1
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
    Linear::ActivationFunction => deri_identity,
    Logistic::ActivationFunction => deri_logistic,
    ReLU::ActivationFunction => deri_relu,
    Tanh::ActivationFunction => deri_tanh,
)

@enum LayerType begin
    Normal
    Convolutional
    MaxPooling
end

function fromType(type::String)
    if type == "Linear"
        (acfun[Linear], deri_acfun[Linear])
    elseif type == "Logistic"
        (acfun[Logistic], deri_acfun[Logistic])
    elseif type == "ReLU"
        (acfun[ReLU], deri_acfun[ReLU])
    elseif type == "Tanh"
        (acfun[Tanh], deri_acfun[Tanh])
    else
        throw(DomainError("Wrong activation type"))
    end
end

parse_str(str) = eval(Meta.parse(str))


# ================ Loses ===============
softmax(y) = y ./ sum(exp, y)  # for vectors


MSE(yhat, label) = -dot(softmax(yhat), label)
    