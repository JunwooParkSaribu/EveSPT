import numpy as np
import tifffile
import matplotlib.pyplot as plt
from itertools import product
from scipy.fft import fft, fftfreq
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment



def signal_filter(signal_ts, nb=3, time=200):
    signal_ts = np.array(signal_ts)
    filtered_signals = []
    for t in signal_ts:
        if len(signal_ts[(signal_ts > t - time) & (signal_ts < t + time)]) >= nb:
            filtered_signals.append(t)
    return np.array(filtered_signals)
   

def signal_mean_diff_time(signal1, signal2):
    #clustering1_ = OPTICS(min_samples=2).fit(signal1.reshape(-1, 1))
    #clustering2_ = OPTICS(min_samples=2).fit(signal2.reshape(-1, 1))
    #clustering1 = DBSCAN(eps=3, min_samples=2).fit(signal1.reshape(-1, 1))
    #clustering2 = DBSCAN(eps=3, min_samples=2).fit(signal1.reshape(-1, 1))
    #clustering1 = GaussianMixture(n_components=2, random_state=0).fit(signal1.reshape(-1, 1))
    #clustering2 = GaussianMixture(n_components=2, random_state=0).fit(signal2.reshape(-1, 1))
    
    #print(signal1, clustering1.predict(signal1.reshape(-1, 1)))
    #print(signal2, clustering2.predict(signal2.reshape(-1, 1)))
    #labels1 = clustering1.predict(signal1.reshape(-1, 1))
    #labels2 = clustering2.predict(signal2.reshape(-1, 1))

    signal1 = np.sort(signal1)
    signal2 = np.sort(signal2)
    labels1 = decompose_signal_by_time(signal1, timebin=5000)
    labels2 = decompose_signal_by_time(signal2, timebin=5000)

    #print(signal1, labels1)
    #print(signal2, labels2)
    mean_times1 = []
    for label in np.unique(labels1):
        arg_ = np.argwhere(labels1 == label).flatten()
        mean_time = np.mean(signal1[arg_])
        mean_times1.append(mean_time)
    mean_times1 = np.array(mean_times1)

    mean_times2 = []
    for label in np.unique(labels2):
        arg_ = np.argwhere(labels2 == label).flatten()
        mean_time = np.mean(signal2[arg_])
        mean_times2.append(mean_time)
    mean_times2 = np.array(mean_times2)

    selected_nb_cluster = min(len(np.unique(labels1[labels1 > -1])), len(np.unique(labels2[labels2 > -1])))
    time_diffs = []
    print(mean_times1, mean_times2)
    if len(mean_times1) == len(mean_times2):
        time_diffs.extend(list(abs(mean_times2 - mean_times1)))
    else:
        for mean_time1, mean_time2 in product(mean_times1, mean_times2):
            time_diffs.append(abs(mean_time2 - mean_time1))
    #time_diffs = abs(mean_times2 - mean_times1)

    #print(mean_times1, mean_times2)
    #row_ind, col_ind = linear_sum_assignment(np.array([mean_times1, mean_times2]))
    #print(mean_times1, mean_times2, row_ind, col_ind)
    time_diffs = np.sort(time_diffs)
    print(time_diffs[:selected_nb_cluster])
    return list(time_diffs[:selected_nb_cluster])


def decompose_signal_by_time(signal, timebin):
    decomposed_signals = []
    labels = []
    label_init = 0
    #for t_gap in np.arange(0, np.max(signal), timebin):
        #decomp_signal = np.where(signal[(signal > t_gap) & (signal < t_gap + timebin)]).flatten()
        #labels.extend([label_init] * len(signal[(signal > t_gap) & (signal < t_gap + timebin)]))
        #label_init += 1
        #decomposed_signals.append(decomp_signal)
    #signal = np.sort(signal)
    for sig_idx in range(len(signal) - 1):

        labels.append(label_init)
        if signal[sig_idx+1] - signal[sig_idx] > 100:
            label_init += 1
    #print(labels)
    return np.array(labels)
    

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
upper_t_limit = 250000 # in ms. 50sec
window_length = 15
nb_dense_events = 1 * window_length**2
nb_dense_times = 200

gt_data = np.load(gt)

gt_xranges = []
gt_yranges = []
gt_tranges = []
print(list(gt_data[2000000:2000002]))
gt_indices = list(set([d[0] for d in gt_data]))
registered_gt_indice = []
for gt_d in gt_data:
    if gt_d[0] not in registered_gt_indice and gt_d[0] > 999:
        if gt_d[2] / time_div < upper_t_limit:
            #print(gt_d)
            gt_xranges.append(np.arange(int(max(0, gt_d[4] - window_length//2)), int(gt_d[4] + window_length//2 + 1)))
            gt_yranges.append(np.arange(int(max(0, gt_d[3] - window_length//2)), int(gt_d[3] + window_length//2 + 1)))
            gt_tranges.append([int(max(0, gt_d[2] / time_div)), int(gt_d[2] / time_div + 3000)])
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


for iii, (xrange, yrange, trange) in enumerate(zip(xranges, yranges, tranges)):
    print(f"{iii} / {len(xranges)}")
    """
    indices_tmp = np.argwhere((xs == 355) & (ys == 272)).flatten()
    ts_tmp = ts[indices_tmp]
    ps_tmp = ps[indices_tmp]
    positive_args = np.argwhere(ps_tmp == 1).flatten()
    negative_args = np.argwhere(ps_tmp == 0).flatten()

    positive_ts_tmp = ts_tmp[positive_args]
    negative_ts_tmp = ts_tmp[negative_args]
    plt.close('all')
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    axs[0].vlines(negative_ts_tmp, ymin=-1, ymax=0, colors='red')
    axs[0].vlines(positive_ts_tmp, ymin=0, ymax=1, colors='blue')
    plt.show()
    """

    #for arr_x, arr_y in product(xrange, yrange):
        #sliding_window_x = np.arange(max(0, arr_x - window_length//2), min(x_max, arr_x + window_length//2), 1)
        #sliding_window_y = np.arange(max(0, arr_y - window_length//2), min(y_max, arr_y + window_length//2), 1)
        #for cur_x, cur_y in product(sliding_window_x, sliding_window_y):
        #print(np.argwhere((xs == arr_x)).flatten())
        #print(np.argwhere((xs == np.array(list(xrange)))).flatten())
    concat_indices = []
    for arr_x, arr_y in product(xrange, yrange):
        indices = np.argwhere((xs == arr_x) & (ys == arr_y)).flatten()
        concat_indices.extend(list(indices))
    concat_indices = np.array(concat_indices)
    indices = concat_indices
    for _ in range(1):
        
        #indices = np.argwhere((xs == arr_x) & (ys == arr_y)).flatten()

        target_ts = ts[indices].astype(np.float64)
        target_ps = ps[indices].astype(np.int16)
        #print(arr_x, arr_y)
        #print(target_ts)
        #print(target_ps)

        roi_ts = np.argwhere((trange[0] < target_ts) & (target_ts < trange[1])).flatten()
        
        target_ts = target_ts[roi_ts]
        target_ps = target_ps[roi_ts]

        positive_args = np.argwhere(target_ps == 1).flatten()
        negative_args = np.argwhere(target_ps == 0).flatten()

        positive_ts = target_ts[positive_args]
        negative_ts = target_ts[negative_args]


        #diff_ts = abs(np.diff(target_ts))
        #diff_ts_concat.extend(list(diff_ts))
        #print(target_ts, target_ps)
        print(xrange[len(xrange)//2], yrange[len(yrange)//2])


        for _ in range(4):
            positive_ts = signal_filter(positive_ts, nb_dense_events, nb_dense_times)
            negative_ts = signal_filter(negative_ts, nb_dense_events, nb_dense_times)
        filtered_positives = positive_ts
        filtered_negatives = negative_ts



        
        plt.close('all')
        fig, axs = plt.subplots(2, 1, figsize=(8, 8))
        axs[0].vlines(negative_ts, ymin=-1, ymax=0, colors='red')
        axs[0].vlines(positive_ts, ymin=0, ymax=1, colors='blue')

        axs[1].vlines(filtered_positives, ymin=0, ymax=1, colors='blue')
        axs[1].vlines(filtered_negatives, ymin=-1, ymax=0, colors='red')
        
        axs[0].set_xlim([-10, upper_t_limit+10])
        axs[1].set_xlim([-10, upper_t_limit+10])

        #yf = fft(positive_ts)
        #xf = fftfreq(len(positive_ts), 100)[:len(positive_ts)//2]
        #plt.figure()
        #plt.plot(xf, 2.0/len(positive_ts) * np.abs(yf[0:len(positive_ts)//2]))
        
        
        
        """
        for positive_t in positive_ts:
            for negative_t in negative_ts:
                if negative_t > positive_t and negative_t < positive_t + 10000:
                    positive_to_negative.append(negative_t - positive_t)
        """
        #if len(filtered_positives) > 0 and len(filtered_negatives) > 0 and filtered_negatives[0] - filtered_positives[-1] > 0:
        #    positive_to_negative.append(filtered_negatives[0] - filtered_positives[-1])
        if len(filtered_positives) > 1 and len(filtered_negatives) > 1:
            positive_to_negative.extend(signal_mean_diff_time(filtered_positives, filtered_negatives))

        plt.show()


#diff_ts_concat = np.array(diff_ts_concat) / 1000.
positive_to_negative = np.array(positive_to_negative, dtype=np.float64)
print(positive_ts, negative_ts)
print(positive_to_negative)

#plt.figure()
#plt.hist(diff_ts_concat, bins=np.arange(0, 10000, 10))
#plt.savefig('1.png')

plt.figure()
plt.hist(positive_to_negative, bins=np.arange(0, 2000, 10))
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
tifffile.imwrite(f"{path}/video_{int(timebin / 1000)}ms.tiff", data=data, imagej=True)
"""
