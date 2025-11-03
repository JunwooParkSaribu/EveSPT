import numpy as np
import tifffile


filename = f'eve_data/Simulated/2025-02-27/Tracking evb diffusion_coefficient=[0.1, 1.0] background_level=50.0/Events.npy'
new_filename = f'eve_data/Simulated/2025-02-27/Tracking evb diffusion_coefficient=[0.1, 1.0] background_level=50.0/Events.npz'
imagename = f'eve_data/Simulated/2025-02-27/Tracking evb diffusion_coefficient=[0.1, 1.0] background_level=50.0/Events_image.npz'

"""
data = np.load(filename)
nb_data = len(data)
polarity = np.empty(nb_data, dtype=np.bool_)
time_stamps = np.empty(nb_data, dtype=np.uint32)
xs = np.empty(nb_data, dtype=np.uint16)
ys = np.empty(nb_data, dtype=np.uint16)

def fun(data):
    cnt = 0
    len_data = len(data)
    while cnt < len_data:
        yield data[cnt][0], data[cnt][1], data[cnt][2], data[cnt][3]
        cnt += 1

for idx, (pol, t, y, x) in enumerate(fun(data)):
    polarity[idx] = pol
    time_stamps[idx] = t
    ys[idx] = y
    xs[idx] = x
np.savez(new_filename, x=xs, y=ys, time_stamps=time_stamps, polarity=polarity)
"""


data = np.load(new_filename)
xs = data['x']
ys = data['y']
ts = data['time_stamps']
ps = data['polarity']
print(data['x'])
x_min = np.min(xs)
x_max = np.max(xs)
y_min = np.min(ys)
y_max = np.max(ys)
polarity_min = np.min(ps)
polarity_max = np.max(ps)
t_min = np.min(ts)
t_max = np.max(ts)

xs = xs - x_min
ys = ys - y_min
print(np.max(ys), y_min, y_max)
timebin = 10000
grid = np.zeros(((int(t_max / timebin) + 1), (y_max - y_min + 1), (x_max - x_min + 1) * 2), dtype=np.uint8)
print(grid.shape)
for x, y, p, t in zip(xs, ys, ts, ps):
    polarity = p

    if polarity == 1:
        grid[int(t / timebin), y, x] += 1
    else:
        grid[int(t / timebin), y, grid.shape[1]//2 + x] += 1
print(x_min, x_max, y_min, y_max, polarity_min, polarity_max, t_min, t_max)
print(grid.shape)
#grid = np.swapaxes(grid, axis1, axis2)[source]

np.savez(imagename, data=grid)

print('saved')
data = np.load(imagename)['data']
print(data, data.dtype)
tifffile.imwrite(f'./video.tiff', data=data, imagej=True)
