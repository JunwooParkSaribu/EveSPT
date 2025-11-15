import numpy as np
import tifffile
import matplotlib.pyplot as plt
from itertools import product
from scipy.fft import fft, fftfreq

"""
# Number of sample points
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N, endpoint=False)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf = fft(y)
xf = fftfreq(N, T)[:N//2]

plt.figure()
plt.plot(x, y)
plt.figure()
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()
"""

path = f"eve_data/Simulated/2025-02-27/Tracking evb diffusion_coefficient=[0.1, 1.0] background_level=50.0"
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



time_div = 1000
upper_t_limit = 50000 # in ms. 50sec

gt_data = np.load(gt)

gt_xranges = []
gt_yranges = []
gt_tranges = []
print(gt_data)
gt_indices = list(set([d[0] for d in gt_data]))
registered_gt_indice = []
for gt_d in gt_data:
    if gt_d[0] not in registered_gt_indice and gt_d[0] < 999:
        if gt_d[2] / time_div < upper_t_limit:
            gt_xranges.append(np.arange(int(max(0, gt_d[4] - 5)), int(gt_d[4] + 5)))
            gt_yranges.append(np.arange(int(max(0, gt_d[3] - 5)), int(gt_d[3] + 5)))
            gt_tranges.append([int(max(0, gt_d[2] - 20000)), int(gt_d[2] + 500000)])
            registered_gt_indice.append(gt_d[0])
print(registered_gt_indice)
xranges = gt_xranges
yranges = gt_yranges
tranges = gt_tranges


data = np.load(new_filename)
xs = data['x']
ys = data['y']
ts = data['time_stamps'].astype(np.float64) / time_div
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
window_length = 20

for iii, (xrange, yrange, trange) in enumerate(zip(xranges, yranges, tranges)):
    print(f"{iii} / {len(xranges)}")
    for arr_x, arr_y in product(xrange, yrange):
        #sliding_window_x = np.arange(max(0, arr_x - window_length//2), min(x_max, arr_x + window_length//2), 1)
        #sliding_window_y = np.arange(max(0, arr_y - window_length//2), min(y_max, arr_y + window_length//2), 1)
        #for cur_x, cur_y in product(sliding_window_x, sliding_window_y):
        indices = np.argwhere((xs == arr_x) & (ys == arr_y)).flatten()

        target_ts = ts[indices].astype(np.float64)
        target_ps = ps[indices].astype(np.int16)
        #print(arr_x, arr_y)
        #print(target_ts)
        #print(target_ps)

        #roi_ts = np.argwhere((trange[0] < target_ts) & (target_ts < trange[1])).flatten()
        
        #target_ts = target_ts[roi_ts]
        #target_ps = target_ps[roi_ts]

        positive_args = np.argwhere(target_ps == 1).flatten()
        negative_args = np.argwhere(target_ps == 0).flatten()

        positive_ts = target_ts[positive_args]
        negative_ts = target_ts[negative_args]


        #diff_ts = abs(np.diff(target_ts))
        #diff_ts_concat.extend(list(diff_ts))
        print(target_ts, target_ps)
        plt.close('all')
        plt.figure(figsize=(10, 10))
        plt.vlines(negative_ts, ymin=-1, ymax=0, colors='red')
        plt.vlines(positive_ts, ymin=0, ymax=1, colors='blue')


        yf = fft(positive_ts)
        xf = fftfreq(len(positive_ts), 100)[:len(positive_ts)//2]
        plt.figure()
        plt.plot(xf, 2.0/len(positive_ts) * np.abs(yf[0:len(positive_ts)//2]))
        plt.show()
        for positive_t in positive_ts:
            for negative_t in negative_ts:
                if negative_t > positive_t and negative_t < positive_t + 10000:
                    positive_to_negative.append(negative_t - positive_t)
                


#diff_ts_concat = np.array(diff_ts_concat) / 1000.
positive_to_negative = np.array(positive_to_negative, dtype=np.float64)
print(positive_ts, negative_ts)
print(positive_to_negative)

#plt.figure()
#plt.hist(diff_ts_concat, bins=np.arange(0, 10000, 10))
#plt.savefig('1.png')

plt.figure()
plt.hist(positive_to_negative, bins=np.arange(0, 10000, 10))
plt.savefig('2.png')
plt.show()
#array_ = xs[:,0]




"""
timebin = 500000
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
tifffile.imwrite(f"{path}/video_{int(timebin / 1000)}ms.tiff", data=data, imagej=True)
"""
