module LogisticRegression

export binary_classification

function validate_inputs(X, Y)
    if X === [] || Y === [] 
        throw(ArgumentError("input matrix should not be empty"))
    elseif size(X, 1) != size(Y, 1)
        throw(DimensionMismatch("input samples and labels should have same number of rows")) 
    end
end


"""
    binary_classification(P, N; <keyword arguments>)

h respective to θ is defined as: `h_θ(x) = g(θᵀ) = 1/(1+exp(-θᵀx))`, 

with the theory of "Maximum Likelihood Estimation", just to find the 

maximum of `l(θ) = ∑ yᵢlog(h_θ(xᵢ)) + (1-yᵢ)log(1-h_θ(xᵢ))` with batch gradient ascent method.

See also: 

...
# Arguments
- `P::Array`: positive samples
- `N::Array`: negative samples
- `θ::Array`: initial value for coefficient, default set to zeros
- `α::AbstractFloat`: learning rate, default value is 0.01
- `ε::AbstractFloat`: condition for terminating iteration, default value is 1e-7
- `max_iter::Number`: the upper bound of iteration, default value is 1e5
...


# Examples
```julia-repl
julia> X = collect(-10:10); Y = [zeros(10, 1); ones(11, 1)];

julia> LogisticRegression.binary_classification(X,Y)
(Main.LogisticRegression.var"#ĥ#6"{Main.LogisticRegression.var"#h_θ#4"}(Core.Box([55.0; 4.589246190232406]), Main.LogisticRegression.var"#h_θ#4"()), [55.0; 4.589246190232406], -0.010109243555064496, 100)

julia> LogisticRegression.binary_classification(X,Y,method="newton")
(Main.LogisticRegression.var"#ĥ#10"{Main.LogisticRegression.var"#h_θ#6"}(Core.Box([16.42612130318277; 8.211013447308321]), Main.LogisticRegression.var"#h_θ#6"()), [16.42612130318277; 8.211013447308321], -0.0005421071770579345, 13)
```

"""
function binary_classification(X, Y, θ=missing; α=1, ε=1e-3, max_iter=1e2, method="batch")
    validate_inputs(X, Y)
    if any(val -> val != 0 && val != 1, Y)
        throw(ArgumentError("label should be either zero or one"))
    end
    if method != "batch" && method != "newton" 
        throw(ArgumentError(string("method ", method, " is not implemented")))
    end

    dim = size(X, 2)
    (θ === missing) && (θ = zeros(dim+1, 1))    # θ is (dim+1 × 1) column vector 
    X = [X ones(size(X, 1), 1)]                 # append ones vector
    h_θ(X, θ) = 1 ./ (1 .+ exp.(-X * θ))        # here X = [x1; x2; ...; xn], returns column vector
    L(θ) = *(h_θ(X, θ).^Y..., (1 .- h_θ(X, θ)).^(1 .- Y)...)
    l = log ∘ L                                 # l(θ) = Y'*log.(h_θ(X, θ)) + (1 .- Y')*log.(1 .- h_θ(X, θ))   
                                                # while 1-h might be too close to 1 that log(1-h) = Inf
                                                # so here calculate L(θ) and get log(L(θ)) 

    # find θ that fits maximum of l(θ)
    iteration = 0
    while iteration < max_iter
        iteration += 1
        
        ▽l = ((Y - h_θ(X, θ))' * X)'           # [▽l]ⱼ = ∑ (yᵢ - h_θ(xᵢ))[xᵢ]ⱼ
        if method == "batch"
            θ += α * ▽l
        elseif method == "newton"
            H = begin                           # Hessian matrix, Hᵢⱼ = ∂²l(θ) / ∂θᵢ∂θⱼ
                h_θX = map(x -> x*(1-x), h_θ(X, θ))
                vcat([h_θX' * (X[:, i] .* X) for i in 1:dim+1]...)
            end
            θ += inv(H)*▽l
        end
        if maximum(abs.(▽l)) < ε
            break
        end
    end

    ĥ(x) = h_θ([x 1], θ)                        # get the module trained "h hat"

    return ĥ, θ, l(θ), iteration
end



end