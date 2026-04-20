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

println("\nPart 2-2:\n")
println("\nDot product (manual):")

v = [1.0, 2.0, 3.0]
w = [4.0, 5.0, 6.0]

dot_manual = (v[1] * w[1]) + (v[2] * w[2]) + (v[3] * w[3])
println(dot_manual)

println("\nDot product (built-in):")
println(dot(v, w))

