module LinearRegression

export batch_gradient_descent, stochastic_gradient_descent, locally_weighted_regression

function validate_inputs(X, Y)
    if X === [] || Y === [] 
        throw(ArgumentError["input matrix should not be empty"])
    elseif size(X, 1) != size(Y, 1)
        throw(DimensionMismatch["input samples and labels should have same number of rows"]) 
    end
end

"""
    batch_gradient_descent(X, Y[, θ]; <keyword arguments>)

Compute the coefficient(`θ`) of linear regression module with the dataset and labels input, using the batch gradient descent method. 

The linear regression module looks like  
    `` ∑_{i=0}^{m} θ x_i = y `` where `` x_0 = 1 ``.  

Coefficients, residual and iteration will be returned.

See also: [`stochastic_gradient_descent`, `locally_weighted_regression`](@ref)

...
# Arguments
- `X::Array`: the dataset 
- `Y::Array`: the targets or labels of the dataset
- `θ::AbstractFloat`: initial value of coefficient in linear regression module
- `α::AbstractFloat`: learning rate, default value is 0.01
- `ε::AbstractFloat`: condition for terminating iteration, default value is 1e-7
- `max_iter::Number`: the upper bound of iteration, default value is 1e5
...


# Examples
```julia-repl
julia> X = [1; 0]; Y = [1; 0];

julia> LinearRegression.batch_gradient_descent(X, Y)
([0.9999997395054627; 1.609944778088623e-7], 1.7909741854163756e-14, 3877)
```

"""
function batch_gradient_descent(X, Y, θ=missing; α=0.01, ε=1e-7, max_iter=1e5)
    validate_inputs(X, Y)

    N = size(X, 1)              # X could be one single column, in which case size(X) returns (N,)
    dim = size(X, 2)            # so there need to get N and dimesion seperately
    X = [X ones(N, 1)]
    (θ === missing) && (θ = zeros(dim+1, 1))
                                # `θ` is an (N+1) dimention vector, including the coefficient for term `x⁰`
                                # Here the initial value of `θ` is set to zeros if `θ` is not given
    J(θ) = sum((X*θ - Y).^2) / 2

    iteration = 0
    while iteration < max_iter
        iteration += 1
        residual = X*θ - Y      # N rows 1 column
        ▽J = (residual' * X)'  # [▽J]ᵢ = Σⱼ (h(Xⱼ) - Yⱼ)*[Xⱼ]ᵢ, where Xⱼ is a row vector or a sample
        θ = θ - α * ▽J         # move to next state
        if maximum(abs.(▽J)) < ε
            break
        end
    end

    return θ, J(θ), iteration
end


"""
    stochastic_gradient_descent(X, Y[, θ]; <keyword arguments>)

Compute the coefficient(`θ`) of linear regression module with the dataset and labels input with stochastic gradient descent method.

Coefficients, residual and iteration will be returned.

See also: [`batch_gradient_descent`, `locally_weighted_regression`](@ref)

...
# Arguments
- `X::Array`: the dataset 
- `Y::Array`: the targets or labels of the dataset
- `θ::AbstractFloat`: initial value of coefficient in linear regression module
- `α::AbstractFloat`: learning rate, default value is 0.01
- `ε::AbstractFloat`: condition for terminating iteration, default value is 1e-7
- `max_iter::Number`: the upper bound of iteration, default value is 1e5
...


# Examples
```julia-repl
julia> X = [1; 0]; Y = [1; 0];

julia> LinearRegression.stochastic_gradient_descent(X, Y)
([0.999974115975191; 1.5952758992491302e-5], 1.7656028009523187e-10, 2663)
```

"""
function stochastic_gradient_descent(X, Y, θ=missing; α=0.01, ε=1e-7, max_iter=1e5)
    validate_inputs(X, Y)

    N = size(X, 1)
    dim = size(X, 2)
    X = [X ones(N, 1)]
    (θ === missing) && (θ = zeros(dim+1, 1))
    J(θ) = sum((X*θ - Y).^2) / 2

    θ₀ = θ
    iteration = 0
    while iteration < max_iter
        iteration += 1
        for i = 1:N
            ▽Jᵢ = (X[i, :]'*θ .- Y[i]).*X[i, :]       # X[i, :] returns row in column vector
            θ = θ - α*▽Jᵢ
        end
        if maximum(abs.(θ - θ₀)) < ε
            break
        end
        θ₀ = θ
    end

    return θ, J(θ), iteration
end


"""
    locally_weighted_regression()

Apply linear regression in one region near the point that needs predicting.  

To predict in a region, `J(θ)` would be modified as:

`J(θ) = Σwᵢ(yᵢ - θxᵢ)²` where i means the ith sample of dataset

`w` is a "weight"(or window) function, usually like

`wᵢ(x) = exp(-(xᵢ - x)^2 / 2)`

Prediction on point x_p and other parameters would be returned.

See also: [`batch_gradient_descent`, `stochastic_gradient_descent`](@ref)

...
# Arguments
- `X::Array`: the dataset 
- `Y::Array`: the targets or labels of the dataset
- `x_p::Array`: the point need prediction
- `θ::AbstractFloat`: initial value of coefficient in linear regression module
- `α::AbstractFloat`: learning rate, default value is 0.01
- `ε::AbstractFloat`: condition for terminating iteration, default value is 1e-7
- `max_iter::Number`: the upper bound of iteration, default value is 1e5
- `τ::Number`: range for weight function, default value is 1.0
...

# Examples
```julia-repl
julia> X = [1; 0]; Y = [1; 0];

julia> LinearRegression.locally_weighted_regression(X, Y, [0.5])
([0.5000000173472761], [0.9999998530315181; 9.08315169928204e-8], [1.0061985509390273e-14], 2278)
```

"""
function locally_weighted_regression(X, Y, x_p, θ=missing; α=0.01, ε=1e-7, max_iter=1e5, τ=1.)
    validate_inputs(X, Y)

    # variable sets
    N = size(X, 1)
    dim = size(X, 2)
    X = [X ones(N, 1)]
    (θ === missing) && (θ = zeros(dim+1, 1))
    (size(x_p, 1) != 1) && (x_p = x_p')                  # need x_p to be row vector
    x_p = [x_p 1]                                        # append with 1
        
    w = map(exp, -sum((X .- x_p).^2, dims=2) / (2τ^2))   # the weight function, using 2-norm. sum(, dims=2) is row summing, returning column
    J(θ) = w' * (X*θ - Y).^2                             # add weight function

    iteration = 0
    while iteration < max_iter
        iteration += 1
        # here using BGD
        ▽J = ((2w .* (X*θ - Y))' * X)'                  # [▽J]ⱼ = 2w.*(Xθ-Y) * X[j, :]'
        θ = θ - α*▽J
        if maximum(abs.(▽J)) < ε
            break
        end
    end

    y_p = x_p * θ           # the prediction

    return y_p, θ, J(θ), iteration
end


end