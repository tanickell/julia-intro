### PHASE 2 ###

# Part 2-1: Vectors and Math
println("Part 2-1:\n")

v = [1.0, 2.0, 3.0]
w = [4.0, 5.0, 6.0]

println("v: ", v)
println("w: ", w)


println("\nAddition:")
println(v + w)

println("\nScalar multiplication:")
println(2 * v)

println("\nElement-wise multiplication:")
println(v .* w)

println("\nLength of v: ", length(v))
println("First element of v: ", v[1])


# Part 2-2: The Dot Product (Core of ML)
# Measures how much two vectors "align"

println("\n\nPart 2-2:")
println("\nDot product (manual):")

v = [1.0, 2.0, 3.0]
w = [4.0, 5.0, 6.0]

dot_manual = (v[1] * w[1]) + (v[2] * w[2]) + (v[3] * w[3])
println(dot_manual)

println("\nDot product (built-in):")
println(dot(v, w))

println("\nYour data dot product:")
println(dot(X[1, :], weights))
println(y_pred[1])


# Part 2-3: Matrices and Matrix Multiplication
println("\n\nPart 2-3:")
println("\nCheck matrix multiplication vs dot product:")
for i in 1:5
    println("Matrix: ", y_pred[i],
            " | Dot: ", dot(X[i, :], weights))
end

println("\nSmall matrix example:")

x_small = [1.0 2.0;
           3.0 4.0]
w_small = [0.5, 1.0]

println("x_small * w_small = ", x_small * w_small)


# Part 2-4: Vectoization vs Loops
println("\n\nPart 2-4:")
println("\nPredictions using a loop:")

y_pred_loop = zeros(length(y))

for i in 1:length(y)
    y_pred_loop[i] = dot(X[i, :], weights)
end

println(y_pred_loop[1:5])


println("\nPredictions using vectorization:")

y_pred_vec = X * weights

println(y_pred_vec[1:5])


println("\nDifference between methods:")
println(round(maximum(abs.(y_pred_loop .- y_pred_vec)))) # should return 0.0


# Final Practice Exercise: Loop to vectorization
z = zeros(length(v))
for i in 1:length(v)
    z[i] = v[i]^2
end

println("\nz using loop:")
for i in 1:length(z)
    println(z[i])
end

z = v .^ 2
println("\nz using vectorization:")
for i in 1:length(z)
    println(z[i])
end

