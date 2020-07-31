"""
    GenerativeLearningAlgorithm

Generative learning algorithm differs from the discriminant learning algorithm which contains

logistic regression and GLMs. GLA learns P(x|y) while DLA learns P(y|x) or h_θ.  

Bayes rule whose content is `P(y=y₀|x) = P(x|y=y₀)P(y=y₀)/P(x)` would be the core of GLA.

"""
module GenerativeLearningAlgorithm

import LinearAlgebra

export gaussian_discriminant_analysis

"""
    gaussian_discriminant_analysis(X, Y, x_p)

GDA has prior models that points of every class follows the gaussian distribution with paramter μ and Σ.

Also we have yᵢ ~ B(1, Φ). This function only implements binary classification.

...
# Arguments
- `X::Array`: samples
- `Y::Array`: labels with value 0 or 1, or it would be handled with 1{Y}
- `x_p::Array`: points to predict
...

"""
function gaussian_discriminant_analysis(X, Y, x_p)
    validate_inputs(X, Y)

    # Y = map(y -> y != 0, Y)
    idx_positive = findall(y -> y != 0, Y)
    idx_negative = findall(y -> y == 0, Y)

    m = length(Y)
    dim = size(X, 2)
    X₀ = X[idx_negative, :]
    X₁ = X[idx_positive, :]
    # calculate parameters
    Φ = length(idx_positive) / m
    μ₀ = sum(X₀, dims=1) / length(idx_negative) 
    μ₁ = sum(X₁, dims=1) / length(idx_positive)
    Σ = (sum((X₁ .- μ₁).^2) + sum((X₀ .- μ₀).^2)) / m
    
    μ₀ = μ₀'
    μ₁ = μ₁'    # to row columns
    # prediction
    detΣ = LinearAlgebra.det(Σ)
    invΣ = inv(Σ)
    h(x) = begin    # input must be column vector
        P₁ = exp(-(x-μ₁)'*invΣ*(x-μ₁)/2) / sqrt((2π)^dim * detΣ) |> scalar
        P₀ = exp(-(x-μ₀)'*invΣ*(x-μ₀)/2) / sqrt((2π)^dim * detΣ) |> scalar
        return convert(Int, P₁*Φ > P₀*(1-Φ))
    end
    y_p = zeros(size(x_p, 1))
    for idx = 1:length(y_p)
        y_p[idx] = h(x_p[idx, :])
    end

    return y_p, h
end

"""
    naive_bayes()

Naive Bayes classifer similarly work as GDA, but with different assumptions or priorities. 

Here mainly implement as a spam filter.

...
# Arguments
- `X::Array`: samples
- `Y::Array`: labels with value 0 or 1, or it would be handled with 1{Y}
- `x_p::Array`: point to predict
...
"""
function naive_bayes(X, Y, x_p)
    validate_inputs(X, Y)

    idx_positive = findall(y -> y != 0, Y)
    idx_negative = findall(y -> y == 0, Y)

    m = length(Y)
    dim = size(X, 2)
    X₀ = X[idx_negative, :]
    X₁ = X[idx_positive, :]

    # parameters
    Φ_y = length(idx_positive) / m
    Φ_j1 = sum(X₁, dims=1) / length(idx_positive)
    Φ_j0 = sum(X₀, dims=1) / length(idx_negative)

    # prediction
    h(x) = begin  # also input need column vector
        P₁ = [x[i] != 0 ? Φ_j1[i] : 1-Φ_j1[i] for i in range(1, stop=length(x))] |> prod
        P₀ = [x[i] != 0 ? Φ_j0[i] : 1-Φ_j0[i] for i in range(1, stop=length(x))] |> prod
        return convert(Int, P₁*Φ_y > P₀*(1-Φ_y))
    end
    y_p = [h(x_p[row, :]) for row in range(1, stop=size(x_p, 1))]
    return y_p, h
end


# inner functions
function validate_inputs(X, Y)
    if X === [] || Y === [] 
        throw(ArgumentError("input matrix should not be empty"))
    elseif size(X, 1) != size(Y, 1)
        throw(DimensionMismatch("input samples and labels should have same number of rows")) 
    end
end

function scalar(m)
    if size(m) != 1
        return m
    end
    reshape(m, 1)[1]
end

end