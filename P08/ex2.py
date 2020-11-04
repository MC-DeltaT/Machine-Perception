import numpy


INPUT_FILE = 'data/prac08ex02_data.txt'


with open(INPUT_FILE) as file:
    data = numpy.array([[float(c) for c in line.split()] for line in file])
measured_positions = data[:, [0, 1]]
true_positions = data[:, [2, 3]]

dt = 1
state_noise_std = 5
measurement_noise_std = 20
initial_position = (2000, 2000)
initial_velocity = (5, 5)

F = numpy.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])
B = 0
u = 0
Q = numpy.identity(4) * state_noise_std ** 2
H = numpy.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])
R = numpy.identity(2) * measurement_noise_std ** 2

x = numpy.array([*initial_position, *initial_velocity])
P = Q.copy()

filtered_positions = []
for z in measured_positions:
    x_predicted = numpy.dot(F, x) + numpy.dot(B, u)
    P_predicted = numpy.dot(numpy.dot(F, P), F.T) + Q
    y = z - numpy.dot(H, x_predicted)
    S = numpy.dot(numpy.dot(H, P_predicted), H.T) + R
    K = numpy.dot(numpy.dot(P_predicted, H.T), numpy.linalg.inv(S))
    x = x_predicted + numpy.dot(K, y)
    P = P_predicted - numpy.dot(numpy.dot(K, H), P_predicted)
    filtered_positions.append(numpy.dot(H, x))
filtered_positions = numpy.array(filtered_positions)

avg_measurement_error = numpy.mean(numpy.linalg.norm(measured_positions - true_positions, axis=1))
avg_filtered_error = numpy.mean(numpy.linalg.norm(filtered_positions - true_positions, axis=1))
print('Average error for measured:', avg_measurement_error)
print('Average error for filtered:', avg_filtered_error)
