import numpy as np
import tifffile
import matplotlib.pyplot as plt
from itertools import product
from scipy.fft import fft, fftfreq
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
from timeit import default_timer as timer
import cupy as cp
import subprocess
import sys
import networkx as nx
from networkx.algorithms import bipartite



def fetch_indices(x_arr, y_arr, x_range, y_range, single_cache):
    concat_indices = []

    if x_range[0] % 100 == 0 and y_range[0] % 100 == 0:
        single_cache = {}

    for x_val, y_val in product(x_range, y_range):
        if (x_val, y_val) not in single_cache:
            indices = np.argwhere((x_arr == x_val) & (y_arr == y_val)).flatten()
            single_cache[(x_val, y_val)] = indices
            concat_indices.extend(indices)
        else:
            concat_indices.extend(single_cache[(x_val, y_val)])
    concat_indices = np.array(concat_indices).flatten()
    return concat_indices


def window_caching(x_arr, y_arr, x_range, y_range, window_cache, single_cache):
    range_tuple = tuple((x_range[0], x_range[-1], y_range[0], y_range[-1]))

    if x_range[0] % 100 == 0 and y_range[0] % 100 == 0:
        window_cache = {}

    if range_tuple not in window_cache:
        indices = fetch_indices(x_arr, y_arr, x_range, y_range, single_cache)
        window_cache[range_tuple] = indices
    else:
        indices = window_cache[range_tuple]
    return indices


def signal_filter(signal_ts, nb=225, time=200):
    start = timer()
    signal_ts = np.array(signal_ts)
    filtered_signals = []
    for t in signal_ts:
        if len(signal_ts[(signal_ts > t - time) & (signal_ts < t + time)]) >= nb:
            filtered_signals.append(t)
    filtered_signals = np.array(filtered_signals)
    end = timer()
    #print(f"FILLTER 1 : {end-start}, {len(filtered_signals)}") 
    return filtered_signals


def pairing(arr1, arr2):
    arr1_labels = np.arange(len(arr1))
    arr2_labels = np.arange(len(arr2))
    B = nx.Graph()
    B.add_nodes_from([f"l_{l}" for l in arr1_labels], bipartite=0)
    B.add_nodes_from([f"r_{r}" for r in arr2_labels], bipartite=1)

    for arr1_l, arr2_l in product(arr1_labels, arr2_labels):
        if not B.has_edge(f"l_{arr1_l}", f"r_{arr2_l}"):
            if arr1[arr1_l] - arr2[arr2_l] < 0:
                B.add_edge(f"l_{arr1_l}", f"r_{arr2_l}", weight=abs(arr1[arr1_l] - arr2[arr2_l]))
    
    if len(B.edges) == 0:
        return {}
    assert bipartite.is_bipartite(B)
    B.remove_nodes_from(list(nx.isolates(B)))
    left, right = nx.bipartite.sets(B)

    #matches = bipartite.minimum_weight_full_matching(B,  top_nodes=None, weight='weight')
    #left_to_right_matches = {int(l[-1]):(int(matches[l][-1]), B.get_edge_data(l, matches[l], None)['weight']) for l in left if l in matches}

    matches = nx.algorithms.min_weight_matching(B, weight='weight')
    reformatted_matches = {}
    for edge in matches:
        n1, n2 = edge
        if 'l' in n1 and 'r' in n2:
            reformatted_matches[n1] = n2
        if 'r' in n1 and 'l' in n2:
            reformatted_matches[n2] = n1
    left_to_right_matches = {int(l[-1]):(int(reformatted_matches[l][-1]), B.get_edge_data(l, reformatted_matches[l], None)['weight']) for l in left if l in reformatted_matches}

    # dict with {left_index: (right_index, weight)}
    return left_to_right_matches


def signal_mean_diff_time(signal1, signal2):
    signal1 = np.sort(signal1)
    signal2 = np.sort(signal2)
    labels1 = signal_labeling(signal1, timebin=100)
    labels2 = signal_labeling(signal2, timebin=100)

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
    matches = pairing(mean_times1, mean_times2)
    if len(matches) == 0:
        return [], {}, labels1, labels2

    time_diffs.extend([matches[l][1] for l in matches])
    return time_diffs, matches, labels1, labels2


def signal_labeling(signal, timebin):
    labels = []
    label_init = 0
    for sig_idx in range(len(signal) - 1):
        labels.append(label_init)
        if signal[sig_idx+1] - signal[sig_idx] > timebin:
            label_init += 1
    return np.array(labels)
    

def convert_event_to_std_format(x_pos, y_pos, positive_event_ts, negative_event_ts):
    xs = []
    ys = []
    ts = []
    ps = []

    for positive_event_t in positive_event_ts:
        xs.append(x_pos)
        ys.append(y_pos)
        ts.append(positive_event_t)
        ps.append(1)
    
    for negative_event_t in negative_event_ts:
        xs.append(x_pos)
        ys.append(y_pos)
        ts.append(negative_event_t)
        ps.append(0)

    return xs, ys, ts, ps


def gridify(gridname, xs, ys, ts, ps, timebin=10, colorise=False, threshold=275, xmax=-1, ymax=-1, tmax=-1):
    total_time_diffs = []
    if xmax==-1:
        xmax = np.max(xs)
    if ymax==-1:
        ymax = np.max(ys)
    if tmax==-1:
        tmax = np.max(ts)

    if colorise == False:
        grid = np.zeros(((int(tmax / timebin) + 1), (ymax + 1), (xmax + 1) * 2), dtype=np.uint8)
        for x, y, t, p in zip(xs, ys, ts, ps):
            polarity = p
            if polarity == 1:
                grid[int(t / timebin), y, x] += 1
            else:
                grid[int(t / timebin), y, grid.shape[2]//2 + x] += 1

    else:
        grid = np.zeros(((int(tmax / timebin) + 1), (ymax + 1), (xmax + 1) * 2, 3), dtype=np.uint8)
        for x_target, y_target in product(np.arange(xmax), np.arange(ymax)):
            if x_target % 100 == 0 and y_target % 100 == 0: print(x_target, y_target)
            xs_filtered, ys_filtered, ts_filtered, ps_filtered = indice_filter_by_position(x_target, y_target, xs, ys, ts, ps)
            if len(xs) == 0:
                continue
            positive_indices, negative_indices = indice_filter_by_polarity(ps_filtered)
            positive_xs = xs_filtered[positive_indices]
            positive_ys = ys_filtered[positive_indices]
            positive_ts = ts_filtered[positive_indices]
            negative_xs = xs_filtered[negative_indices]
            negative_ys = ys_filtered[negative_indices]
            negative_ts = ts_filtered[negative_indices]
            if len(positive_ts) == 0 or len(negative_ts) == 0:
                continue
            positive_colours, negative_colours, time_diffs = add_colour_by_time_diff(positive_ts, negative_ts, threshold=threshold)
            total_time_diffs.extend(time_diffs)

            for x, y, t, rgb in zip(positive_xs, positive_ys, positive_ts, positive_colours):
                if rgb[0] == 1 and rgb[1] == 0 and rgb[2] == 0:
                    grid[int(t / timebin), y, x, 0] += 1
                elif rgb[0] == 0 and rgb[1] == 1 and rgb[2] == 0:
                    grid[int(t / timebin), y, x, 1] += 1
                else:
                    grid[int(t / timebin), y, x, 2] += 1

            for x, y, t, rgb in zip(negative_xs, negative_ys, negative_ts, negative_colours):
                if rgb[0] == 1 and rgb[1] == 0 and rgb[2] == 0:
                    grid[int(t / timebin), y, grid.shape[2]//2 + x, 0] += 1
                elif rgb[0] ==0 and rgb[1] == 1 and rgb[2] == 0:
                    grid[int(t / timebin), y, grid.shape[2]//2 + x, 1] += 1
                else:
                    grid[int(t / timebin), y, grid.shape[2]//2 + x, 2] += 1

    np.savez(gridname, data=grid)
    return total_time_diffs


def make_video(path, gridfile, nb_frames):
    data = np.load(gridfile)['data'][:nb_frames,:,:]
    tifffile.imwrite(path, data=data, imagej=True)


def read_processed_events(path):
    assert ".npz" in path
    data = np.load(path)
    xs = data['x']
    ys = data['y']
    ts = data['time_stamps'].astype(np.float64)
    ps = data['polarity']
    return xs, ys, ts, ps


def indice_filter_by_polarity(polarities):
    positive_args = np.argwhere(polarities == 1).flatten()
    negative_args = np.argwhere(polarities == 0).flatten()
    return positive_args, negative_args


def indice_filter_by_position(x_target, y_target, xs, ys, ts, ps):
    indices = np.argwhere((xs == x_target) & (ys == y_target)).flatten()
    return xs[indices], ys[indices], ts[indices], ps[indices]


def inverse_bipartitie_graph(graph:dict):
    new_graph = {}
    ls = list(graph.keys())
    for l in ls:
        r, val = graph[l]
        new_graph[r] = (l, val)
    return new_graph


def add_colour_by_time_diff(positive_events, negative_events, threshold):
    positive_colours = []
    negative_colours = []

    time_diffs, matches, positive_labels, negative_labels = signal_mean_diff_time(positive_events, negative_events)
    inversed_matches = inverse_bipartitie_graph(matches)

    for positive_t, positive_label in zip(positive_events, positive_labels):
        if positive_label in matches:
            time_diff = matches[positive_label][1]
            if time_diff > threshold:
                positive_colours.append((1, 0, 0))
            else:
                positive_colours.append((0, 0, 1))
        else:
            positive_colours.append((0, 1, 0))
    
    for negative_t, negative_label in zip(negative_events, negative_labels):
        if negative_label in inversed_matches:
            time_diff = inversed_matches[negative_label][1]
            if time_diff > threshold:
                negative_colours.append((1, 0, 0))
            else:
                negative_colours.append((0, 0, 1))
        else:
            negative_colours.append((0, 1, 0)) 

    return positive_colours, negative_colours, time_diffs


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
filtered_events_name = f"{path}/filtered_events.npz"
gt = f"{path}/Tracks_GT.npy"


"""
data = np.load(filtered_events_name)
xs = data['x']
ys = data['y']
ts = data['time_stamps'].astype(np.float64)
ps = data['polarity']
timebin = 10 #ms
gridify(f"{path}/filtered_events_image_{timebin}ms.npz", xs, ys, ts, ps, timebin=timebin)
make_video(f"{path}/filtered_video_{timebin}ms.tiff", f"{path}/filtered_events_image_{timebin}ms.npz", nb_frames=10000)
exit()
"""

time_div = 1000  # us to ms for original data
timebin = 10
upper_t_limit = 50000 # in ms. 50sec

original_data = np.load(new_filename)
xs = original_data['x']
ys = original_data['y']
ts = original_data['time_stamps'].astype(np.float64) / time_div
selected_args = np.argwhere(ts < upper_t_limit).flatten()
xs = xs[selected_args]
ys = ys[selected_args]
ts = ts[selected_args]
xmax = np.max(xs)
ymax = np.max(ys)
tmax = np.max(ts)

filtered_xs, filtered_ys, filtered_ts, filtered_ps = read_processed_events(filtered_events_name)
time_diffs = gridify(f"{path}/filtered_events_image_{timebin}ms_color.npz", filtered_xs, filtered_ys, filtered_ts, filtered_ps, timebin=timebin, colorise=True, threshold=275, xmax=xmax, ymax=ymax, tmax=tmax)
make_video(f"{path}/filtered_video_{timebin}ms_color.tiff", f"{path}/filtered_events_image_{timebin}ms_color.npz", nb_frames=10000)
plt.figure()
plt.hist(list(time_diffs), bins=np.arange(0, 1000, 10))
plt.savefig('timediffs.png')
exit()



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
window_length = 7
nb_dense_events = 1 * window_length**2
nb_dense_times = 100
single_cache = {}
window_cache = {}


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


## do all for every pixels
xranges = []
yranges = []
tranges = []
for x in np.arange(0 + window_length//2, x_max - window_length//2, 1):
    for y in np.arange(0 + window_length//2, y_max - window_length//2, 1):
        xranges.append(np.arange(int(max(0, x - window_length//2)), int(x + window_length//2 + 1)))
        yranges.append(np.arange(int(max(0, y - window_length//2)), int(y + window_length//2 + 1)))
        tranges.append([0, 9999999999999])


diff_ts_concat = []
positive_to_negative = []
filtered_xs = []
filtered_ys = []
filtered_ps = []
filtered_ts = []

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
    if iii%100 == 0: print(f"{iii} / {len(xranges)}")
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
    
    start = timer()
    #indices = fetch_indices(xs, ys, xrange, yrange, cache)
    indices = window_caching(xs, ys, xrange, yrange, window_cache, single_cache)
    end = timer()
    #print(f"cached indice concatenation: {end-start}", len(indices)) 
    
    start = timer()

    """
    concat_indices = []
    for arr_x, arr_y in product(xrange, yrange):
        indices = np.argwhere((xs == arr_x) & (ys == arr_y)).flatten()
        concat_indices.extend(list(indices))
    concat_indices = np.array(concat_indices)
    indices = concat_indices
    """

    end = timer()
    #print(f"indice concatenation: {end-start}", len(indices)) 
    for _ in range(1):
        
        #indices = np.argwhere((xs == arr_x) & (ys == arr_y)).flatten()
        start = timer()


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


        end = timer()
        #print(f"argwhere : {end-start}") 

        #diff_ts = abs(np.diff(target_ts))
        #diff_ts_concat.extend(list(diff_ts))
        #print(target_ts, target_ps)
        #print(xrange[len(xrange)//2], yrange[len(yrange)//2])
        picked_pos_x = xrange[len(xrange)//2]
        picekd_pos_y = yrange[len(yrange)//2]


        start = timer()
        for _ in range(4):
            positive_ts = signal_filter(positive_ts, nb_dense_events, nb_dense_times)
            negative_ts = signal_filter(negative_ts, nb_dense_events, nb_dense_times)
        filtered_positives = positive_ts
        filtered_negatives = negative_ts



        end = timer()
        #print(f"filtering : {end-start}") 
        start = timer()

        xs_tmp, ys_tmp, ts_tmp, ps_tmp = convert_event_to_std_format(picked_pos_x, picekd_pos_y, filtered_positives, filtered_negatives)
        filtered_xs.extend(xs_tmp)
        filtered_ys.extend(ys_tmp)
        filtered_ts.extend(ts_tmp)
        filtered_ps.extend(ps_tmp)

        """
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
        
        
        end = timer()
        #print(f"Rest : {end-start}") 

        #plt.show()


np.savez(filtered_events_name, x=filtered_xs, y=filtered_ys, time_stamps=filtered_ts, polarity=filtered_ps)


#diff_ts_concat = np.array(diff_ts_concat) / 1000.
positive_to_negative = np.array(positive_to_negative, dtype=np.float64)
print(positive_ts, negative_ts)
print(positive_to_negative)

#plt.figure()
#plt.hist(diff_ts_concat, bins=np.arange(0, 10000, 10))
#plt.savefig('1.png')

#plt.figure()
#plt.hist(positive_to_negative, bins=np.arange(0, 2000, 10))
#plt.savefig('2.png')
#plt.show()
#array_ = xs[:,0]