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

x_min = np.min(xs)
y_min = np.min(ys)

xs = xs - x_min
ys = ys - y_min
x_max = np.max(xs)
y_max = np.max(ys)
t_max = np.max(ts)

timebin = 100000
grid = np.zeros(((int(t_max / timebin) + 1), (y_max + 1), (x_max + 1) * 2), dtype=np.uint8)

for x, y, t, p in zip(xs, ys, ts, ps):
    polarity = p
    if polarity == 1:
        grid[int(t / timebin), y, x] += 1
    else:
        grid[int(t / timebin), y, grid.shape[2]//2 + x] += 1


print(grid.shape)
np.savez(imagename, data=grid)

print('saved')
data = np.load(imagename)['data'][:5000,:,:]
print(data, data.dtype)
tifffile.imwrite(f'./video.tiff', data=data, imagej=True)
