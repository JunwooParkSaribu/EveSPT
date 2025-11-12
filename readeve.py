import numpy as np
import tifffile
import matplotlib.pyplot as plt
from itertools import product

filename = f'eve_data/Experimental/2022-12-08/recording_2022-12-08T19-47-03.127Z.npy'
new_filename = f'eve_data/Experimental/2022-12-08/Events.npz'
imagename = f'eve_data/Experimental/2022-12-08/Events_image.npz'

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

for idx, (x, y, p, t) in enumerate(fun(data)):
#for idx, (p, t, y, x) in enumerate(fun(data)):
    polarity[idx] = p
    time_stamps[idx] = t
    ys[idx] = y
    xs[idx] = x
np.savez(new_filename, x=xs, y=ys, time_stamps=time_stamps, polarity=polarity)
"""

size = 5000000000000000

data = np.load(new_filename)
xs = data['x'][:size]
ys = data['y'][:size]
ts = data['time_stamps'][:size]
ps = data['polarity'][:size]

x_min = np.min(xs)
y_min = np.min(ys)

xs = xs - x_min
ys = ys - y_min
x_max = np.max(xs)
y_max = np.max(ys)
t_max = np.max(ts)


diff_ts_concat = []
for arr_x, arr_y in product(range(0, x_max), range(0, y_max)):
    #arr_x = 328
    #arr_y = 238
    
    indices = np.argwhere((xs == arr_x) & (ys == arr_y)).flatten()

    target_ts = ts[indices].astype(np.int32)
    target_ps = ps[indices].astype(np.int32)

    diff_ts = abs(np.diff(target_ts))
    diff_ts_concat.extend(list(diff_ts))

diff_ts_concat = np.array(diff_ts_concat)
print(diff_ts_concat)

plt.figure()
plt.hist(diff_ts_concat)
plt.show()
#array_ = xs[:,0]



"""
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


data = np.load(imagename)['data'][:5000,:,:]
print(data, data.dtype)
tifffile.imwrite(f'eve_data/Experimental/2022-12-08/video.tiff', data=data, imagej=True)
"""