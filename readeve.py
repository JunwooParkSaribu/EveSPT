import numpy as np
import tifffile
import matplotlib.pyplot as plt
from itertools import product

path = f"eve_data/Experimental/2022-12-08"
filename = f"{path}/Events.npy"
new_filename = f"{path}/Events.npz"
imagename = f"{path}/Events_image.npz"
gt = f"{path}/Tracks_GT.npy"

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

"""
gt_data = np.load(gt)
gt_xranges = []
gt_yranges = []
gt_tranges = []
for gt_d in gt_data:
    if gt_d[2] < upper_t_limit:
        gt_xranges.append(np.arange(int(max(0, gt_d[4] - 15)), int(gt_d[4] + 15)))
        gt_yranges.append(np.arange(int(max(0, gt_d[3] - 15)), int(gt_d[3] + 15)))
        gt_tranges.append([int(max(0, gt_d[2] - 20000)), int(gt_d[2] + 500000)])
xranges = gt_xranges
yranges = gt_yranges
tranges = gt_tranges
"""

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
"""
#all
xrange = np.arange(0, 50)
yrange = np.arange(0, 50)
"""
"""
#signal included ROIs
xranges = [np.arange(348, 364),
           np.arange(232, 248)
           ]
yranges = [np.arange(252, 268),
           np.arange(113, 129)
           ]
tranges = [[0, 500000],
           [12800000, 14000000],
           ]
"""
"""
#signal excluded ROIs
xranges = [np.arange(0, 45),
           np.arange(0, 45)
           ]
yranges = [np.arange(0, 45),
           np.arange(474, 516)
           ]
tranges = [[0, 10000000],
           [0, 10000000],
           ]
"""
"""
for iii, (xrange, yrange, trange) in enumerate(zip(xranges, yranges, tranges)):
    print(f"{iii} / {len(xranges)}")
    for arr_x, arr_y in product(xrange, yrange):
        indices = np.argwhere((xs == arr_x) & (ys == arr_y)).flatten()

        target_ts = ts[indices].astype(np.int32)
        target_ps = ps[indices].astype(np.int32)
        print(arr_x, arr_y)
        print(target_ts)
        print(target_ps)

        roi_ts = np.argwhere((trange[0] < target_ts) & (target_ts < trange[1])).flatten()
        
        target_ts = target_ts[roi_ts]
        target_ps = target_ps[roi_ts]

        positive_args = np.argwhere(target_ps == 1).flatten()
        negative_args = np.argwhere(target_ps == 0).flatten()

        positive_ts = target_ts[positive_args]
        negative_ts = target_ts[negative_args]


        diff_ts = abs(np.diff(target_ts))
        diff_ts_concat.extend(list(diff_ts))

        for positive_t in positive_ts:
            for negative_t in negative_ts:
                if negative_t > positive_t:
                    positive_to_negative.append(negative_t - positive_t)
                


diff_ts_concat = np.array(diff_ts_concat) / 1000.
positive_to_negative = np.array(positive_to_negative) / 1000.
print(positive_ts, negative_ts)
print(positive_to_negative)

plt.figure()
plt.hist(diff_ts_concat, bins=np.arange(0, 10000, 10))
plt.savefig('1.png')

plt.figure()
plt.hist(positive_to_negative, bins=np.arange(0, 10000, 10))
plt.savefig('2.png')
plt.show()
#array_ = xs[:,0]

"""


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
tifffile.imwrite(f"eve_data/Experimental/2022-12-08/video_{int(timebin / 1000)}ms.tiff", data=data, imagej=True)
"""