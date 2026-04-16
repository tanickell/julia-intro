# Student Test Score Modeler
# Predict student scores based on multiple factors

# Tim Nickell  |  Nebula-NLU  |  tim.nickell@nebula-nlu.com
# 2026-04-16

# main.jl


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
bad_indices = []

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

for i in 1:5
    println((hours_studied[i], hours_slept[i], distractions[i], scores[i]))
end
