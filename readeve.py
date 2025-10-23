import numpy as np
import tifffile


filename = f'data/recording_2025-04-04_12-41-35.npy'
new_filename = f'data/recording_2025-04-04_12-41-35.npz'
imagename = f'data/recording_2025-04-04_12-41-35_image.npz'
"""
data = np.load(filename)
data = np.array([[x, y, p, t] for (x, y, p, t) in data], dtype=np.uint32)
np.savez(new_filename, data=data)
"""



data = np.load(new_filename)['data']

x_min = np.min(data[:, 0])
x_max = np.max(data[:, 0])
y_min = np.min(data[:, 1])
y_max = np.max(data[:, 1])
polarity_min = np.min(data[:, 2])
polarity_max = np.max(data[:, 2])
t_min = np.min(data[:, 3])
t_max = np.max(data[:, 3])

data[:, 0] = data[:, 0] - x_min
data[:, 1] = data[:, 1] - y_min
print(np.max(data[:, 1]), y_min, y_max)
timebin = 10000
grid = np.zeros(((int(t_max / timebin) + 1), (y_max - y_min + 1), (x_max - x_min + 1) * 2), dtype=np.uint8)
print(grid.shape)
for dt in data:
    polarity = dt[2]
    if polarity == 1:
        grid[int(dt[3] / timebin), dt[1], dt[0]] += 1
    else:
        grid[int(dt[3] / timebin), dt[1], grid.shape[1]//2 + dt[0]] += 1
print(x_min, x_max, y_min, y_max, polarity_min, polarity_max, t_min, t_max)
print(grid.shape)
#grid = np.swapaxes(grid, axis1, axis2)[source]

np.savez(imagename, data=grid)

data = np.load(imagename)['data']
print(data, data.dtype)
tifffile.imwrite(f'./video.tiff', data=data, imagej=True)