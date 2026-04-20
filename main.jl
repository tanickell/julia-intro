# Student Test Score Modeler
# Predict student scores based on multiple factors

# Tim Nickell  |  Nebula-NLU  |  tim.nickell@nebula-nlu.com
# 2026-04-16

# main.jl

using Statistics
using LinearAlgebra

# Phase 1 - Part 1

# Number of samples
n = 100

# generate synthetic data
hours_studied = rand(1:10, n)
hours_slept   = rand(4:10, n)
distractions  = rand(0:10, n) 

# Generate scores (with some randomness)
# scores = (5 .* hours_studied) .+ (3 .* hours_slept) .- (2 .* distractions) .+ 
#     rand(-5:5, n)
scores = (10 .* hours_studied) .+ (3 .* hours_slept) .- (2 .* distractions) .+ 
    rand(-5:5, n)

# Print first 5 samples
for i in 1:5
    println((hours_studied[i], hours_slept[i], distractions[i], scores[i]))
end

# Phase 1 - Part 2

# Introduce intentionally bad data

# Missing values (use -1 as placeholder)
hours_studied[5] = -1
hours_slept[10] = -1

# Impossible values --> negative sleep = nonsense
hours_slept[15] = -3 

# Outlier (unrealistic score)
scores[20] = 999


# Now, find the bad data (indices of bad data)
bad_indices = Int[]

for i in 1:n
    if hours_studied[i] == -1 ||  # missing value
       hours_slept[i] == -1 ||    # missing value 
       hours_slept[i] < 0 ||      # negative value
       scores[i] > 100            # outlier

        push!(bad_indices, i)
    end
end

println()
println("Bad data indices: ", bad_indices)

# keep only good data
good_mask = trues(n)
for i in bad_indices
    good_mask[i] = false
end

# apply mask
hours_studied = hours_studied[good_mask]
hours_slept = hours_slept[good_mask]
distractions = distractions[good_mask]
scores = scores[good_mask]

println()
println("Cleaned dataset size: ", length(scores))

println()
for i in 1:5
    println((hours_studied[i], hours_slept[i], distractions[i], scores[i]))
end


# Part 1-3: Data Transformation (Feature Engineering)

# Normalize Function
function normalize(x)
    return (x .- minimum(x)) ./ (maximum(x) - minimum(x))
end

# Apply normalization
hours_studied_n = normalize(hours_studied)
hours_slept_n = normalize(hours_slept)
distractions_n = normalize(distractions)
scores_n = normalize(scores)

println()
println("Hours studied (min, max): ", minimum(hours_studied_n), ", ", maximum(hours_studied_n))

effective_study = hours_studied .- (0.5 .* distractions)
effective_study_n = normalize(effective_study)

# println()
# for i in 1:5
#     println((effective_study[i], hours_slept[i], distractions[i], scores[i]))
# end


# Part 1-4: Structuring Data for ML (X and y)

# Combine features into matrix X
X = hcat(
    hours_studied_n,
    hours_slept_n,
    distractions_n,
    effective_study_n
)
println("Size of X: ", size(X))

y = scores_n
println("Size of y: ", size(y))

println()
println("First sample (X): ", X[1, :])
println("First target (y): ", y[1])

println()
for i in 1:5
    println(X[i, :])
end


# Part 1-5: A First "Model-Like" Computation

# Number of features
num_features = size(X, 2)

# Random weights
weights = rand(num_features)

println()
println("Weights: ", weights)

# Predictions
y_pred = X * weights

println()
println("First 5 normalized y:")
println(y[1:5])

println()
println("First 5 predictions:")
println(y_pred[1:5])

# Compare to actual values
println()
println("Actual vs Predicted (first 5):")
for i in 1:5
    println("y: ", y[i], " | y_pred: ", y_pred[i])
end

#Mean Squared Error (MSE)
mse = mean((y_pred .- y) .^ 2)

println()
println("MSE: ", mse)


# Quick troubleshoot / sanity check
println(X[1, :])
println(weights)
println(dot(X[1, :], weights))

