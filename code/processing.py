#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Data preprocessing and graph construction module
'''

import os
import glob
import math
import numpy as np
import obspy
from sklearn.model_selection import train_test_split

# ==============================================================================
# Grid dimensions (rows x columns) of the four Longmenshan fault segments
FAULT_DIMS = [(16, 31), (11, 7), (11, 19), (7, 29)]
TOTAL_FAULT_NODES = sum([r * c for r, c in FAULT_DIMS])


# ==============================================================================
# 1. Extract three-component waveforms (E, N, Z)
# ==============================================================================
def rENZ(home, project_name, run_name, GFlist, tcs_samples, outdir='Tmpout_X', ncpus=1, filter_freq=0.5):
    """Extract SAC waveforms, apply low-pass filtering, truncate, and save as .npy files."""
    if not (os.path.exists(outdir)):
        os.makedirs(outdir)

    use_parallel = False
    if ncpus > 1:
        try:
            from joblib import Parallel, delayed
            use_parallel = True
        except:
            print("Import joblib failed")
            use_parallel = False

    waveforms_dir = os.path.join(home, project_name, 'output', 'waveforms')
    ruptures = glob.glob(os.path.join(waveforms_dir, run_name + '*'))
    ruptures.sort()

    def genENZ(ruptures):
        print('number of directories:%d' % (len(ruptures)))
        for nrupt, rupt in enumerate(ruptures):
            pname = project_name
            eqid = os.path.basename(rupt).split('.')[-1]

            if nrupt % 100 == 0:
                print(rupt, eqid)

            all_stname = []
            with open(GFlist, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                st_name = parts[0]
                all_stname.append(st_name)

            E_sta = []
            N_sta = []
            Z_sta = []
            for stname in all_stname:
                sac_base = os.path.join(rupt, stname)

                if os.path.exists(sac_base + '.LYE.sac') and \
                        os.path.exists(sac_base + '.LYN.sac') and \
                        os.path.exists(sac_base + '.LYZ.sac'):

                    O_E = obspy.read(sac_base + '*LYE.sac')
                    O_N = obspy.read(sac_base + '*LYN.sac')
                    O_Z = obspy.read(sac_base + '*LYZ.sac')

                    # ==========================================================
                    # Pad the waveform edges before low-pass filtering, then trim the padding.
                    # ==========================================================
                    if filter_freq is not None and filter_freq > 0:
                        try:
                            dt = O_E[0].stats.delta
                            pad_len = int(200.0 / dt)

                            O_E[0].data -= O_E[0].data[0]
                            O_N[0].data -= O_N[0].data[0]
                            O_Z[0].data -= O_Z[0].data[0]

                            O_E[0].data = np.pad(O_E[0].data, (pad_len, pad_len), mode='edge')
                            O_N[0].data = np.pad(O_N[0].data, (pad_len, pad_len), mode='edge')
                            O_Z[0].data = np.pad(O_Z[0].data, (pad_len, pad_len), mode='edge')

                            O_E.filter('lowpass', freq=filter_freq, corners=2, zerophase=False)
                            O_N.filter('lowpass', freq=filter_freq, corners=2, zerophase=False)
                            O_Z.filter('lowpass', freq=filter_freq, corners=2, zerophase=False)

                            # 5. Remove the padding to restore the original waveform length.
                            O_E[0].data = O_E[0].data[pad_len:-pad_len]
                            O_N[0].data = O_N[0].data[pad_len:-pad_len]
                            O_Z[0].data = O_Z[0].data[pad_len:-pad_len]

                        except Exception as e:
                            print(f"Filter error on {stname}: {e}")

                    E = O_E[0].data
                    N = O_N[0].data
                    Z = O_Z[0].data
                    O_E.clear()
                    O_N.clear()
                    O_Z.clear()

                    E_cut = E[:tcs_samples]
                    N_cut = N[:tcs_samples]
                    Z_cut = Z[:tcs_samples]
                else:
                    E_cut = np.zeros(tcs_samples)
                    N_cut = np.zeros(tcs_samples)
                    Z_cut = np.zeros(tcs_samples)

                # ==========================================================
                E_sta.append(E_cut)
                N_sta.append(N_cut)
                Z_sta.append(Z_cut)

            E_sta = np.array(E_sta)
            N_sta = np.array(N_sta)
            Z_sta = np.array(Z_sta)

            np.save(os.path.join(outdir, f'{pname}.{eqid}.E.npy'), E_sta)
            np.save(os.path.join(outdir, f'{pname}.{eqid}.N.npy'), N_sta)
            np.save(os.path.join(outdir, f'{pname}.{eqid}.Z.npy'), Z_sta)

    if use_parallel:
        sub_ruptures = {i: [] for i in range(ncpus)}
        for i_rupt, rupt in enumerate(ruptures):
            gp = i_rupt % ncpus
            sub_ruptures[gp].append(rupt)
        Parallel(n_jobs=ncpus, verbose=0)(delayed(genENZ)(i) for i in sub_ruptures.values())
    else:
        genENZ(ruptures)


# ==============================================================================
# 2. Extract physical source parameters
# ==============================================================================
def rMnS(home, project_name, run_name, outdir='Tmpout_MnS'):
    os.makedirs(outdir, exist_ok=True)

    total_patch = TOTAL_FAULT_NODES
    print(f"Target Geometry: Longmenshan (4 Segments)")
    print(f"Physical Subfaults: {total_patch} Nodes (STGNN Mode, No Padding)")

    log_dir = os.path.join(home, project_name, 'output', 'ruptures')
    rupts = glob.glob(os.path.join(log_dir, f'{run_name}*rupt'))
    rupts.sort()

    y = {'ss_slip': [], 'ds_slip': [], 'rise_time': [], 'rupt_time': []}

    for n, rupt in enumerate(rupts):
        if n % 1000 == 0:
            print(f'Processing {n}/{len(rupts)}: {os.path.basename(rupt)}')

        arr = np.genfromtxt(rupt)
        raw_ss = arr[:, 8]
        raw_ds = arr[:, 9]
        raw_rise = arr[:, 7]
        raw_time = arr[:, 12]

        if raw_ss.size != total_patch:
            if raw_ss.size > total_patch:
                raw_ss, raw_ds, raw_rise, raw_time = raw_ss[:total_patch], raw_ds[:total_patch], raw_rise[
                                                                                                 :total_patch], raw_time[
                                                                                                                :total_patch]
            else:
                raise ValueError(f'{rupt} Size mismatch: {raw_ss.size} vs {total_patch}')

        y['ss_slip'].append(raw_ss)
        y['ds_slip'].append(raw_ds)
        y['rise_time'].append(raw_rise)
        y['rupt_time'].append(raw_time)

    for key in y.keys():
        y[key] = np.array(y[key], dtype=np.float32)

    save_path = os.path.join(outdir, 'y.npy')
    np.save(save_path, y)


# ==============================================================================
# 3. Build the station spatial topology (graph adjacency matrix)
# ==============================================================================
def build_graph_adj(station_file, out_dir='Split_info', sigma=2.0):
    if not os.path.exists(station_file):
        print(f"Error: Station file {station_file} not found.")
        return

    STA = np.genfromtxt(station_file, 'S12', skip_header=1)
    sta_coords = [[float(ista[1]), float(ista[2])] for ista in STA]
    num_stations = len(sta_coords)

    print(f"Building Graph Adjacency Matrix for {num_stations} stations...")

    dist_matrix = np.zeros((num_stations, num_stations))

    for i in range(num_stations):
        for j in range(num_stations):
            if i == j:
                dist_matrix[i, j] = 0.0
            else:
                dist = obspy.geodetics.locations2degrees(
                    lat1=sta_coords[i][1], long1=sta_coords[i][0],
                    lat2=sta_coords[j][1], long2=sta_coords[j][0]
                )
                dist_matrix[i, j] = dist

    adj_matrix = np.exp(-(dist_matrix ** 2) / (sigma ** 2)).astype(np.float32)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'adj_matrix.npy'), adj_matrix)
    print(f"Adjacency matrix saved (Shape: {adj_matrix.shape})")


# ==============================================================================
# 3.5 Build the physical fault topology
# ==============================================================================
def build_fault_adj_3d(fault_file, out_dir='Split_info'):
    print(f"Building 3D Fault Adjacency Matrix from {fault_file} (Hybrid + Fault Isolation)...")
    if not os.path.exists(fault_file):
        print(f"Error: Fault file {fault_file} not found!")
        return

    fault_data = np.genfromtxt(fault_file)
    lon = fault_data[:, 1]
    lat = fault_data[:, 2]
    dep = fault_data[:, 3]
    num_nodes = len(lon)

    fault_adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    dims = FAULT_DIMS

    # Record the segment containing each node (0, 1, 2, or 3).
    segments = []
    start_idx = 0
    for (rows, cols) in dims:
        segments.append(range(start_idx, start_idx + rows * cols))
        start_idx += rows * cols

    def get_seg_id(node_idx):
        for s, r in enumerate(segments):
            if node_idx in r: return s
        return -1

    # ==========================================================
    # Step A: Construct the within-segment grid topology.
    # ==========================================================
    start_idx = 0
    for seg_id, (rows, cols) in enumerate(dims):
        for r in range(rows):
            for c in range(cols):
                idx = start_idx + r * cols + c
                fault_adj[idx, idx] = 1.0

                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            n_idx = start_idx + nr * cols + nc
                            fault_adj[idx, n_idx] = 1.0
        start_idx += rows * cols

    # ==========================================================
    # Step B: Connect nearby segments while isolating the front- and back-range fault segments.
    # ==========================================================
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            seg_i = get_seg_id(i)
            seg_j = get_seg_id(j)

            if seg_i != seg_j:

                if {seg_i, seg_j} == {0, 3}:
                    continue

                dist_m, _, _ = obspy.geodetics.gps2dist_azimuth(lat[i], lon[i], lat[j], lon[j])
                dist_epi_km = dist_m / 1000.0
                delta_dep_km = abs(dep[i] - dep[j])

                if dist_epi_km <= 7.0 and delta_dep_km <= 6.0:
                    fault_adj[i, j] = 1.0
                    fault_adj[j, i] = 1.0

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fault_adj.npy')
    np.save(out_path, fault_adj)

    # ==========================================================
    # Step C: Write a readable topology table as a text file.
    # ==========================================================
    txt_out_path = os.path.join(out_dir, 'fault_topology_connections.txt')
    with open(txt_out_path, 'w', encoding='utf-8') as f:
        f.write("# =========================================================\n")
        f.write(f"# 3D Fault Topology Connections (Total Nodes: {num_nodes})\n")
        f.write("# Format: Node_ID (Segment_ID) -> [Neighbor_1, Neighbor_2, ...]\n")
        f.write("# =========================================================\n\n")

        for i in range(num_nodes):
            neighbors = np.where(fault_adj[i] == 1.0)[0]
            neighbors = neighbors[neighbors != i]
            neighbors_str = ", ".join(map(str, neighbors))
            seg_id = get_seg_id(i) + 1

            f.write(f"Node {i:03d} (Seg {seg_id}, Neighbors: {len(neighbors)}) -> [{neighbors_str}]\n")

    avg_neighbors = (np.sum(fault_adj) - num_nodes) / num_nodes
    print(f" 3D Fault Adjacency Tensor saved to: {out_path} (Shape: {fault_adj.shape})")
    print(f" Topology Text Log saved to: {txt_out_path}")
    print(f" Graph Physics Info: Average physical neighbors per node = {avg_neighbors:.2f}")


# ==============================================================================
# 4. Generate waveform path lists
# ==============================================================================
def gen_Xylist(X_dirs, y_dirs=None, outname='Datalist'):
    if not (type(X_dirs) is list): X_dirs = [X_dirs]
    E, N, Z = [], [], []
    for Xdir in X_dirs:
        E += sorted(glob.glob(os.path.join(Xdir, '*.E.npy')))
        N += sorted(glob.glob(os.path.join(Xdir, '*.N.npy')))
        Z += sorted(glob.glob(os.path.join(Xdir, '*.Z.npy')))

    with open(outname + '_E' + '.txt', 'w') as OUTE:
        for line in E: OUTE.write('%s\n' % line)
    with open(outname + '_N' + '.txt', 'w') as OUTN:
        for line in N: OUTN.write('%s\n' % line)
    with open(outname + '_Z' + '.txt', 'w') as OUTZ:
        for line in Z: OUTZ.write('%s\n' % line)


# ==============================================================================
# 5. Merge lists
# ==============================================================================
def merge_lists(name_merge, name_out):
    suffixs = glob.glob(name_merge[0] + '*')
    suffixs = [os.path.basename(i).split('_')[-1] for i in suffixs]
    for suffix in suffixs:
        mg = []
        for i in name_merge:
            tmp = glob.glob(i + suffix)
            if len(tmp) == 0:
                mg = []
                break
            mg += tmp
        if len(mg) == 0: continue
        with open(name_out + suffix, 'w') as OUT:
            for m in mg:
                with open(m, 'r') as IN: OUT.write(IN.read())


# ==============================================================================
# 6. Extract earthquake metadata
# ==============================================================================
def get_EQinfo(home, project_name, run_name, outname='EQinfo', fmt='short'):
    def get_source_Info(logfile, fmt='short'):
        with open(logfile, 'r') as IN1:
            for line in IN1.readlines():
                if 'Lmax' in line: L = float(line.split()[3])
                if 'Wmax' in line: W = float(line.split()[3])
                if 'Target magnitude' in line: tar_Mw = float(line.split()[3])
                if 'Actual magnitude' in line: Mw = float(line.split()[3])
                if 'Hypocenter (lon,lat,z[km])' in line:
                    Hypo_xyz = line.split(':')[-1].replace('(', '').replace(')', '').strip().split(',')
                if 'Centroid (lon,lat,z[km])' in line:
                    Cent_xyz = line.split(':')[-1].replace('(', '').replace(')', '').strip().split(',')
                    break
        if fmt == 'short':
            return Mw, float(Hypo_xyz[0]), float(Hypo_xyz[1]), float(Hypo_xyz[2]), float(Cent_xyz[0]), float(
                Cent_xyz[1]), float(Cent_xyz[2])
        elif fmt == 'long':
            return Mw, float(Hypo_xyz[0]), float(Hypo_xyz[1]), float(Hypo_xyz[2]), float(Cent_xyz[0]), float(
                Cent_xyz[1]), float(Cent_xyz[2]), tar_Mw, L, W

    def get_slip_Info(rupt_file, eqlon, eqlat, fmt='short'):
        A = np.genfromtxt(rupt_file)
        lon, lat = A[:, 1], A[:, 2]
        Slip = (A[:, 8] ** 2.0 + A[:, 9] ** 2.0) ** 0.5
        Rise = A[:, 7]
        idx_slip = np.where(Slip != 0)[0]
        hypo_misft = np.abs(lon - eqlon) ** 2.0 + np.abs(lat - eqlat) ** 2.0
        hypoidx = np.where(hypo_misft == np.min(hypo_misft))[0][0]
        hyposlip = Slip[hypoidx]
        if fmt == 'short':
            return hyposlip, Slip.max()
        elif fmt == 'long':
            return hyposlip, Slip.max(), Slip[idx_slip].mean(), Slip[idx_slip].std(), Rise[idx_slip].max(), Rise[
                idx_slip].mean(), Rise[idx_slip].std()

    ruptures_dir = os.path.join(home, project_name, 'output', 'ruptures')
    logs = sorted(glob.glob(os.path.join(ruptures_dir, run_name + '*log')))
    rupts = sorted(glob.glob(os.path.join(ruptures_dir, run_name + '*rupt')))

    with open(outname + '.EQinfo', 'w') as OUT1:
        for n, logfile in enumerate(logs):
            if n == 0:
                if fmt == 'short':
                    OUT1.write('#ID Mw Hypo_lon Hypo_lat Hypo_dep Cen_lon Cen_lat Cen_dep hypo_slip max_slip\n')
                elif fmt == 'long':
                    OUT1.write(
                        '#ID[0] Mw[1] Hypo_lon[2] Hypo_lat[3] Hypo_dep[4] Cen_lon[5] Cen_lat[6] Cen_dep[7] hypo_slip[8] max_slip[9] mean_slip[10] std_slip[11] max_rise[12] mean_rise[13] std_rise[14] tarMw[15] Len[16] Wid[17] Rupt[18]\n')
            ID = '%06d' % (n)
            if fmt == 'short':
                Mw, eqlon, eqlat, eqdep, cenlon, cenlat, cendep = get_source_Info(logfile, fmt=fmt)
                hypo_slip, max_slip = get_slip_Info(rupts[n], eqlon, eqlat, fmt=fmt)
                OUT1.write('%s  %.4f  %.6f %.6f %.2f   %.6f %.6f %.2f %f %f\n' % (
                    ID, Mw, eqlon, eqlat, eqdep, cenlon, cenlat, cendep, hypo_slip, max_slip))
            elif fmt == 'long':
                Mw, eqlon, eqlat, eqdep, cenlon, cenlat, cendep, tar_Mw, L, W = get_source_Info(logfile, fmt=fmt)
                hypo_slip, max_slip, mean_slip, std_slip, max_rise, mean_rise, std_rise = get_slip_Info(rupts[n], eqlon,
                                                                                                        eqlat, fmt=fmt)
                OUT1.write('%s  %.4f  %.6f %.6f %.2f   %.6f %.6f %.2f %f %f %f %f %f %f %f %.4f %f %f %s\n' % (
                    ID, Mw, eqlon, eqlat, eqdep, cenlon, cenlat, cendep, hypo_slip, max_slip, mean_slip, std_slip,
                    max_rise,
                    mean_rise, std_rise, tar_Mw, L, W, rupts[n]))


# ==============================================================================
# 7. Split the dataset (train / validation / test)
# ==============================================================================
def gen_split(files, params):
    split = params['split']
    if len(split) != 3 or math.fsum(split) != 1: return

    EQinfo = np.genfromtxt(files['EQinfo'], dtype=None, encoding='UTF-8')

    E = np.array([i.decode() for i in np.genfromtxt(files['E'], 'S')])
    N = np.array([i.decode() for i in np.genfromtxt(files['N'], 'S')])
    Z = np.array([i.decode() for i in np.genfromtxt(files['Z'], 'S')])

    y = np.load(files['y'], allow_pickle=True).item()

    print("Adding waveform paths to the label dictionary...")
    y['wave_paths_E'], y['wave_paths_N'], y['wave_paths_Z'] = E, N, Z

    print("Extracting reference magnitudes and adding them to the label dictionary...")
    try:
        Mw_array = np.array([float(row[1]) for row in EQinfo], dtype=np.float32)
    except:
        Mw_array = np.array([float(row['f1']) for row in EQinfo], dtype=np.float32)
    y['mag_output'] = Mw_array.reshape(-1, 1)

    train_idx, valid_test_idx = train_test_split(np.arange(0, len(E)),
                                                 test_size=(split[1] + split[2]) / math.fsum(split), random_state=130)

    X_train_E, X_train_N, X_train_Z = E[train_idx], N[train_idx], Z[train_idx]
    EQinfo_train = EQinfo[train_idx]
    y_train = {k: v[train_idx] for k, v in y.items()}

    X_valid_test_E, X_valid_test_N, X_valid_test_Z = E[valid_test_idx], N[valid_test_idx], Z[valid_test_idx]
    EQinfo_valid_test = EQinfo[valid_test_idx]
    y_valid_test = {k: v[valid_test_idx] for k, v in y.items()}

    valid_idx, test_idx = train_test_split(np.arange(0, len(valid_test_idx)),
                                           test_size=split[2] / (split[1] + split[2]), random_state=130)

    X_valid_E, X_valid_N, X_valid_Z = X_valid_test_E[valid_idx], X_valid_test_N[valid_idx], X_valid_test_Z[valid_idx]
    EQinfo_valid = EQinfo_valid_test[valid_idx]
    y_valid = {k: y_valid_test[k][valid_idx] for k in y.keys()}

    X_test_E, X_test_N, X_test_Z = X_valid_test_E[test_idx], X_valid_test_N[test_idx], X_valid_test_Z[test_idx]
    EQinfo_test = EQinfo_valid_test[test_idx]
    y_test = {k: y_valid_test[k][test_idx] for k in y.keys()}

    out_split_dir = './Split_info'
    os.makedirs(out_split_dir, exist_ok=True)

    np.save(os.path.join(out_split_dir, 'EQinfo_train.npy'), EQinfo_train)
    np.save(os.path.join(out_split_dir, 'EQinfo_valid.npy'), EQinfo_valid)
    np.save(os.path.join(out_split_dir, 'EQinfo_test.npy'), EQinfo_test)

    np.save(os.path.join(out_split_dir, 'X_train_E.npy'), X_train_E)
    np.save(os.path.join(out_split_dir, 'X_train_N.npy'), X_train_N)
    np.save(os.path.join(out_split_dir, 'X_train_Z.npy'), X_train_Z)
    np.save(os.path.join(out_split_dir, 'X_valid_E.npy'), X_valid_E)
    np.save(os.path.join(out_split_dir, 'X_valid_N.npy'), X_valid_N)
    np.save(os.path.join(out_split_dir, 'X_valid_Z.npy'), X_valid_Z)
    np.save(os.path.join(out_split_dir, 'X_test_E.npy'), X_test_E)
    np.save(os.path.join(out_split_dir, 'X_test_N.npy'), X_test_N)
    np.save(os.path.join(out_split_dir, 'X_test_Z.npy'), X_test_Z)

    np.save(os.path.join(out_split_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(out_split_dir, 'y_valid.npy'), y_valid)
    np.save(os.path.join(out_split_dir, 'y_test.npy'), y_test)

    print("Dataset split completed. saved to ./Split_info")


# ==============================================================================
# 8. Waveform noise bank
# ==============================================================================
def generate_noise_bank(num_traces=10000, num_pts=512, dt=0.5, filter_freq=0.5, outdir='./Split_info'):
    print(f"\nGenerating {num_traces} background-noise displacement traces...")
    os.makedirs(outdir, exist_ok=True)

    # 1. Generate Gaussian white noise.
    white_noise = np.random.randn(num_traces, num_pts)

    # 2. Integrate to generate red (Brownian) noise.
    base_noise = np.cumsum(white_noise, axis=-1)

    # 3. Remove the mean and linear trend.
    x = np.arange(num_pts)
    for i in range(num_traces):
        p = np.polyfit(x, base_noise[i], 1)
        base_noise[i] -= np.polyval(p, x)

    # 4. Apply the same causal low-pass filter and edge padding used for the waveforms.
    print(f"Applying a {filter_freq} Hz low-pass filter with edge padding...")
    filtered_noise = np.zeros_like(base_noise)

    for i in range(num_traces):
        tr = obspy.Trace(data=base_noise[i])
        tr.stats.delta = dt

        pad_len = int(200.0 / dt)

        tr.data -= tr.data[0]
        tr.data = np.pad(tr.data, (pad_len, pad_len), mode='edge')
        tr.filter('lowpass', freq=filter_freq, corners=2, zerophase=False)

        tr.data = tr.data[pad_len:-pad_len]
        filtered_noise[i] = tr.data

    # 5. Standardization
    std_dev = np.std(filtered_noise, axis=-1, keepdims=True)
    std_dev[std_dev == 0] = 1e-8
    final_noise = filtered_noise / std_dev

    # 6. saved to Split_info
    outpath = os.path.join(outdir, 'noise_bank.npy')
    np.save(outpath, final_noise)
    print(f"Noise bank saved to: {outpath}")
    print(f"Noise bank array shape: {final_noise.shape}")

    return final_noise


def build_fault_area(fault_file, out_dir='./Split_info'):
    """
    Read the fault file, extract each subfault's physical area, and save it
    as a NumPy array suitable for conversion to a tensor.
    The input dimensions are in meters (m), so their product is in square meters (m^2).
    """
    import os
    import numpy as np

    print(f"Extracting physical subfault areas from {fault_file} ...")
    if not os.path.exists(fault_file):
        print(f"Error: Fault file {fault_file} not found!")
        return

    # Read the fault file.
    fault_data = np.genfromtxt(fault_file)
    length_m = fault_data[:, 8]
    width_m = fault_data[:, 9]

    areas_m2 = length_m * width_m
    areas_m2 = areas_m2.astype(np.float32)

    # Save as a .npy file.
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fault_areas_m2.npy')
    np.save(out_path, areas_m2)

    print(f"subfault area(m^2) saved to: {out_path} (Shape: {areas_m2.shape})")