import matplotlib.pyplot as plt
import random
import numpy as np
import random
from ex2_utils import generate_responses_1, get_patch
from ex1_utils import gausssmooth

def mean_shift(image, start_point, kernel_size):
    curr_x = start_point[0]
    curr_y = start_point[1]
    steps = 0
    points = [(curr_x, curr_y)]
    patch, inliers = get_patch(image, start_point, kernel_size)

    offset_x = kernel_size[0]//2
    offset_y = kernel_size[1]//2
    column = np.arange(-offset_x, offset_x+1)
    row = np.arange(-offset_y, offset_y+1)
    [x_i, y_i] = np.meshgrid(column, row)

    while True:
        denominator = np.sum(patch)
        if denominator == 0:
            break
        dx = np.sum(x_i*patch) / denominator
        dy = np.sum(y_i*patch) / denominator
        if np.linalg.norm(np.array([dx,dy])) < 0.1:
            break
        curr_x += dx
        curr_y += dy
        patch, inliers = get_patch(image, (curr_x, curr_y), kernel_size)
        points.append((curr_x, curr_y))
        steps += 1

    return points, steps

def show_path(image, points):
    plt.imshow(image)
    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]
    plt.plot(x_values[0], y_values[0],color="magenta", marker="o",
        linestyle="None", label="Start point")
    plt.plot(x_values, y_values, color="red", label="Path")
    plt.legend()
    plt.show()


def generate_responses_2():
    random.seed(3)
    responses = np.zeros((100, 100), dtype=np.float32)
    for i in range(4):
        curr_point = (random.randint(0,responses.shape[0]-1), random.randint(0,responses.shape[0]-1))
        responses[curr_point[0], curr_point[1]] = random.random()
    return gausssmooth(responses, 10)


test_image = generate_responses_2()
start_point = (40,40)
kernel_size = (5,5)
points, steps = mean_shift(test_image, start_point, kernel_size)
print(f"Needed {steps} steps.")
show_path(test_image, points)


# HELPER AND MODIFIED FUNCTIONS FOR MAKING PLOTS FOR REPORT

# def mean_shift(image, start_point, kernel_size, thresh):
#     curr_x = start_point[0]
#     curr_y = start_point[1]
#     steps = 0
#     points = [(curr_x, curr_y)]
#     patch, inliers = get_patch(image, start_point, kernel_size)

#     offset_x = kernel_size[0]//2
#     offset_y = kernel_size[1]//2
#     column = np.arange(-offset_x, offset_x+1)
#     row = np.arange(-offset_y, offset_y+1)
#     [x_i, y_i] = np.meshgrid(column, row)

#     while True:
#         denominator = np.sum(patch)
#         if denominator == 0:
#             break
#         dx = np.sum(x_i*patch) / denominator
#         dy = np.sum(y_i*patch) / denominator
#         if np.linalg.norm(np.array([dx,dy])) < thresh:
#             break
#         curr_x += dx
#         curr_y += dy
#         patch, inliers = get_patch(image, (curr_x, curr_y), kernel_size)
#         points.append((curr_x, curr_y))
#         steps += 1

#     return points, steps

# def show_paths(image, all_points):
#     plt.imshow(image)
#     for i, points in enumerate(all_points):
#         x_values = [p[0] for p in points]
#         y_values = [p[1] for p in points]
#         plt.plot(x_values[0], y_values[0], color="red", marker="o",
#             linestyle="None")
#         plt.text(x_values[0]+1, y_values[0]+1, f"{i}", color="red")
#         plt.plot(x_values, y_values, label=f"{i}")
#     plt.show()

# def show_paths(image, all_points, ker):
#     plt.imshow(image)
#     plt.plot(all_points[0][0], all_points[0][1], color="red", marker="o",
#         linestyle="None")
#     for i, points in enumerate(all_points):
#         x_values = [p[0] for p in points]
#         y_values = [p[1] for p in points]
#         plt.plot(x_values, y_values, label=f"({ker[i]},{ker[i]})")
#     plt.legend()
#     plt.show()

# def show_paths(image, runs, epsilons):
#     for k, all_points in enumerate(runs):
#         plt.subplot(1,3,k+1)
#         plt.imshow(image)
#         for i, points in enumerate(all_points):
#             x_values = [p[0] for p in points]
#             y_values = [p[1] for p in points]
#             plt.plot(x_values[0], y_values[0], color="red", marker="o",
#                 linestyle="None")
#             plt.text(x_values[0]+1, y_values[0]+1, f"{i}", color="red")
#             plt.plot(x_values, y_values, label=f"{i}")
#             plt.title(f"Epsilon={epsilons[k]}")
#     plt.show()

# all_points = []
# random.seed(4)
# kernel_size = (9,9)
# start_points = [(random.randint(0,99), random.randint(0,99)) for i in range(10)]
# test_image = generate_responses_2()
# for i in range(10):
#     curr_start = start_points[i]
#     curr_points, curr_steps = mean_shift(test_image, curr_start, kernel_size)
#     all_points.append(curr_points)
#     print(f"{i}: Start: {curr_start}, #Steps: {curr_steps}")

# show_paths(test_image, all_points)

# all_points = []
# start_point = (50,50)
# kernel_sizes = [5,11,15]
# for kernel_size in kernel_sizes:
#     curr_points, curr_steps = mean_shift(test_image, start_point, (kernel_size, kernel_size))
#     all_points.append(curr_points)
#     print(f"Kernel size: {kernel_size}, #Steps: {curr_steps}")
# show_paths(test_image, all_points, kernel_sizes)

# all_points = []
# random.seed(4)
# kernel_size = (9,9)
# thresh = [1, 0.5, 0.1]
# start_points = [(random.randint(0,test_image.shape[0]-1), random.randint(0,test_image.shape[0]-1)) for i in range(10)]
# for t in thresh:
#     curr_run = []
#     print(f"Threshold: {t}")
#     for i in range(10):
#         curr_start = start_points[i]
#         curr_points, curr_steps = mean_shift(test_image, curr_start, kernel_size, t)
#         curr_run.append(curr_points)
#         print(f"{i}: Start: {curr_start}, #Steps: {curr_steps}")
#     all_points.append(curr_run)

# show_paths(test_image, all_points, thresh)
