import numpy as np
import tifffile
import matplotlib.pyplot as plt
from itertools import product

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
print(data.dtype)
def fun(data):
    cnt = 0
    len_data = len(data)
    while cnt < len_data:
        yield data[cnt][0], data[cnt][1], data[cnt][2], data[cnt][3]
        cnt += 1

#for idx, (x, y, p, t) in enumerate(fun(data)):
for idx, (p, t, y, x) in enumerate(fun(data)):
    polarity[idx] = p
    time_stamps[idx] = t
    ys[idx] = y
    xs[idx] = x
np.savez(new_filename, x=xs, y=ys, time_stamps=time_stamps, polarity=polarity)
"""

upper_t_limit = 50000000 # 50sec

data = np.load(new_filename)
xs = data['x']
ys = data['y']
ts = data['time_stamps']
ps = data['polarity']

selected_args = np.argwhere(ts < upper_t_limit).flatten()
xs = xs[selected_args]
ys = ys[selected_args]
ts = ts[selected_args]
ps = ps[selected_args]


x_min = np.min(xs)
y_min = np.min(ys)

xs = xs - x_min
ys = ys - y_min
x_max = np.max(xs)
y_max = np.max(ys)
t_max = np.max(ts)


diff_ts_concat = []
positive_to_negative = []

#all
xrange = np.arange(0, x_max)
yrange = np.arange(0, y_max)

"""
#signal included roi
xrange = np.arange(376, 448)
yrange = np.arange(193, 256)
"""
"""
#signal excluded roi
xrange = np.arange(430, 494)
yrange = np.arange(422, 481)
"""

for arr_x, arr_y in product(xrange, yrange):
    indices = np.argwhere((xs == arr_x) & (ys == arr_y)).flatten()

    target_ts = ts[indices].astype(np.int32)
    target_ps = ps[indices].astype(np.int32)


    positive_args = np.argwhere(target_ps == 1).flatten()
    negative_args = np.argwhere(target_ps == 0).flatten()

    positive_ts = target_ts[positive_args]
    negative_ts = target_ts[negative_args]

    #if len(target_ts) >= 2:
    #    target_ts = target_ts[target_ps]

    diff_ts = abs(np.diff(target_ts))
    diff_ts_concat.extend(list(diff_ts))

    for positive_t in positive_ts:
        for negative_t in negative_ts:
            if negative_t > positive_t:
                positive_to_negative.append(negative_t - positive_t)


diff_ts_concat = np.array(diff_ts_concat) / 1000.
positive_to_negative = np.array(positive_to_negative) / 1000.
print(diff_ts_concat)

plt.figure()
plt.hist(diff_ts_concat, bins=np.arange(0, 10000, 10))
plt.savefig('1.png')

plt.figure()
plt.hist(positive_to_negative, bins=np.arange(0, 10000, 10))
plt.savefig('2.png')
plt.show()
#array_ = xs[:,0]




"""
timebin = 10000
grid = np.zeros(((int(t_max / timebin) + 1), (y_max + 1), (x_max + 1) * 2), dtype=np.uint8)

for x, y, t, p in zip(xs, ys, ts, ps):
    polarity = p
    if polarity == 1:
        grid[int(t / timebin), y, x] += 1
    else:
        grid[int(t / timebin), y, grid.shape[2]//2 + x] += 1


print(grid.shape)
np.savez(imagename, data=grid)


data = np.load(imagename)['data'][:10000,:,:]
print(data, data.dtype)
tifffile.imwrite(f'eve_data/Experimental/2022-12-08/video.tiff', data=data, imagej=True)
"""