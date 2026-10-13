import os, sys
import numpy as np
import glob

HOME = "/root/autodl-tmp"
FAULT_FILE = os.path.join(HOME, 'longmenshan.fault')

BATCHES = [
    ('longmenshan01', 'longmenshan01'),
    ('longmenshan02', 'longmenshan02'),
    ('longmenshan03', 'longmenshan03')
]


OUT_ROOT = os.path.join(HOME, 'Output')
GF = os.path.join(HOME, 'station.txt')
N_POINTS = 512
N_CPUS = 10

# Enable the preprocessing stages needed for the experiment.
RUN_PHASE_1_EXTRACT = True
RUN_PHASE_2_MERGE = True
RUN_PHASE_3_SPLIT = True
RUN_PHASE_4_FAULT_GRAPH = False
RUN_PHASE_5_NOISE = False
RUN_PHASE_6_FAULT_AREA = False

sys.path.append(os.getcwd())
import processing


def main():
    if not os.path.exists(OUT_ROOT): os.makedirs(OUT_ROOT)

    batch_results = []
    for project_name, _ in BATCHES:
        batch_out_dir = os.path.join(OUT_ROOT, project_name)
        dir_MnS = os.path.join(batch_out_dir, 'Tmpout_MnS')
        dir_List = os.path.join(batch_out_dir, 'Datalist')
        batch_results.append({
            'name': project_name,
            'E_list': dir_List + '_E.txt',
            'N_list': dir_List + '_N.txt',
            'Z_list': dir_List + '_Z.txt',
            'y_npy': os.path.join(dir_MnS, 'y.npy'),
            'eq_info': os.path.join(batch_out_dir, 'EQinfo.EQinfo')
        })

    total_list_prefix = os.path.join(OUT_ROOT, 'Total_Datalist')
    total_eq_path = os.path.join(OUT_ROOT, 'Total_EQinfo.EQinfo')

    if RUN_PHASE_1_EXTRACT:
        for project_name, run_name in BATCHES:
            print(f"Processing batch: {project_name}")
            batch_out_dir = os.path.join(OUT_ROOT, project_name)
            dir_X = os.path.join(batch_out_dir, 'Tmpout_X')
            dir_MnS = os.path.join(batch_out_dir, 'Tmpout_MnS')
            dir_List = os.path.join(batch_out_dir, 'Datalist')
            eq_out_prefix = os.path.join(batch_out_dir, 'EQinfo')

            processing.rENZ(HOME, project_name, run_name, GF, N_POINTS, outdir=dir_X, filter_freq=0.5,ncpus=N_CPUS)
            processing.rMnS(HOME, project_name, run_name, outdir=dir_MnS)
            processing.get_EQinfo(HOME, project_name, run_name, outname=eq_out_prefix, fmt='long')
            processing.gen_Xylist([dir_X], [], dir_List)

    if RUN_PHASE_2_MERGE:
        print("Merging batches...")
        for suffix in ['_E.txt', '_N.txt', '_Z.txt']:
            key = suffix[1:2] + '_list'
            with open(total_list_prefix + suffix, 'w') as f_out:
                for batch in batch_results:
                    src = batch[key]
                    if os.path.exists(src):
                        with open(src, 'r') as f_in:
                            content = f_in.read()
                            f_out.write(content)
                            if content and not content.endswith('\n'): f_out.write('\n')

        y_list = [np.load(b['y_npy'], allow_pickle=True).item() for b in batch_results if os.path.exists(b['y_npy'])]
        if y_list:
            y_total = {key: (
                np.hstack([d[key] for d in y_list]) if np.array([d[key] for d in y_list][0]).ndim == 1 else np.vstack(
                    [d[key] for d in y_list])) for key in y_list[0].keys()}
            np.save(os.path.join(OUT_ROOT, 'Total_y.npy'), y_total)

        with open(total_eq_path, 'w') as f_out:
            header_written = False
            for batch in batch_results:
                if not os.path.exists(batch['eq_info']): continue
                with open(batch['eq_info'], 'r') as f_in:
                    lines = f_in.readlines()
                if lines:
                    if not header_written:
                        f_out.write(lines[0]);
                        header_written = True
                    for line in lines[1:]: f_out.write(line)

    if RUN_PHASE_3_SPLIT:
        files = {
            'EQinfo': total_eq_path,
            'E': total_list_prefix + '_E.txt',
            'N': total_list_prefix + '_N.txt',
            'Z': total_list_prefix + '_Z.txt',
            'y': os.path.join(OUT_ROOT, 'Total_y.npy'),
        }
        if os.path.exists(files['y']):
            processing.gen_split(files, {'split': [0.7, 0.2, 0.1]})
            processing.build_graph_adj(GF, out_dir='./Split_info', sigma=2.0)
        else:
            print("Error: Total_y.npy not found; dataset split skipped.")

    if RUN_PHASE_4_FAULT_GRAPH:
        processing.build_fault_adj_3d(FAULT_FILE, out_dir='./Split_info')

    if RUN_PHASE_5_NOISE:
        processing.generate_noise_bank(
            num_traces=10000,
            num_pts=N_POINTS,
            dt=0.5,
            filter_freq=0.5,
            outdir='./Split_info'
        )

    if RUN_PHASE_6_FAULT_AREA:
        try:
            processing.build_fault_area(FAULT_FILE, out_dir='./Split_info')
        except AttributeError:
            print("Error: build_fault_area is missing from processing.py.")


if __name__ == "__main__":
    main()
