import random

# Define the number of points to generate
num_points = 10

# Define the size of the square
square_size = 100

# Generate random points within the square
points = []
for i in range(num_points):
    x = random.uniform(0, square_size)
    y = random.uniform(0, square_size)
    points.append((x, y))

# Print the generated points
for i, point in enumerate(points):
    print(f"Point {i+1}: ({point[0]}, {point[1]})")
