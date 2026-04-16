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
