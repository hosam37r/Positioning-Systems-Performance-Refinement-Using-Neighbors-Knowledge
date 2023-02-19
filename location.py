import math

# Define the positions and received powers of the antennas
antennas = [
    {'pos': (0, 0), 'power': -60},
    {'pos': (0, 10), 'power': -70},
    {'pos': (10, 0), 'power': -80},
    {'pos': (10, 10), 'power': -90}
]

# Define a function to calculate the distance from an antenna based on received power
def distance_from_power(power, alpha=2, d0=1):
    return d0 * math.pow(10, (power / (-10 * alpha)))

# Define a function to calculate the location based on distances from antennas
def location_from_distances(distances):
    x1, y1 = antennas[0]['pos']
    x2, y2 = antennas[1]['pos']
    x3, y3 = antennas[2]['pos']

    A = 2 * (x2 - x1)
    B = 2 * (y2 - y1)
    C = math.pow(distances[0], 2) - math.pow(distances[1], 2) - math.pow(x1, 2) + math.pow(x2, 2) - math.pow(y1, 2) + math.pow(y2, 2)
    D = 2 * (x3 - x2)
    E = 2 * (y3 - y2)
    F = math.pow(distances[1], 2) - math.pow(distances[2], 2) - math.pow(x2, 2) + math.pow(x3, 2) - math.pow(y2, 2) + math.pow(y3, 2)

    x = (C * E - F * B) / (E * A - B * D)
    y = (C * D - A * F) / (B * D - A * E)

    return x, y

# Calculate the distances from the received powers of the antennas
distances = [distance_from_power(antenna['power']) for antenna in antennas]

# Calculate the location based on the distances
location = location_from_distances(distances)

# Print the location
print(f"The approximate location is {location}")
