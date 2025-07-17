import sys
import os
import json
import numpy as np
import wfdb
import heartpy as hp
from scipy import signal as sp_signal

# Ensure record_id is always passed. If not, exit immediately.
if len(sys.argv) < 2:
    print("Error: record_id not provided. Usage: python process_ecg.py <record_id>")
    sys.exit(1)

record_id = sys.argv[1] # This should now always be the UUID from app.py

project_root = os.path.dirname(os.path.abspath(__file__))

uploads_data = os.path.join(project_root, 'uploads')
output_data = os.path.join(project_root, 'outputs')

print(f"Processing record_id: {record_id} in {uploads_data}")

# Verify all three required files exist
required_exts = [".hea", ".dat", ".atr"]
for ext in required_exts:
    file_path = os.path.join(uploads_data, f"{record_id}.{ext}")
    if not os.path.exists(file_path):
        print(f"Error: Missing expected file: {file_path}")
        sys.exit(1) # Exit with an error code if a file is missing

# CRITICAL CHANGE: wfdb.rdrecord expects just the base record name,
# and it will look for the .hea, .dat, .atr files in the specified directory.
# The directory is implicitly the current working directory, or can be specified with 'pb_dir'.
# Since we are already in the correct directory, or have constructed paths, we give it the base name.

# Option 1: Change current directory to uploads_data, then call wfdb.rdrecord
# This is often the simplest way to handle wfdb
original_cwd = os.getcwd()
try:
    os.chdir(uploads_data)
    print(f"Changed current directory to: {os.getcwd()}")
    record = wfdb.rdrecord(record_id) # Now simply pass the record_id
    fs = record.fs
    ecg_signal = record.p_signal[:, 0]
    print(f"Successfully read record: {record_id}, Sampling Frequency: {fs}")
except Exception as e:
    print(f"Error reading WFDB record {record_id} from {uploads_data}: {e}")
    sys.exit(1)
finally:
    # Always change back to original directory to avoid affecting other parts of the app
    os.chdir(original_cwd)
    print(f"Changed back to original directory: {os.getcwd()}")


# Option 2 (Alternative - if Option 1 causes issues): Pass pb_dir explicitly
# This requires wfdb.rdrecord to support pb_dir for local paths, which it usually does.
# try:
#     record = wfdb.rdrecord(record_id, pb_dir=uploads_data)
#     fs = record.fs
#     ecg_signal = record.p_signal[:, 0]
#     print(f"Successfully read record: {record_id}, Sampling Frequency: {fs}")
# except Exception as e:
#     print(f"Error reading WFDB record {record_id} from {uploads_data}: {e}")
#     sys.exit(1)


def fir_bandpass(ecg, fs, low=3, high=45, taps=101):
    nyq = 0.5 * fs
    if taps % 2 == 0:
        taps += 1
    filt = sp_signal.firwin(taps, [low / nyq, high / nyq], pass_zero=False)
    return sp_signal.lfilter(filt, 1.0, ecg)

filtered = fir_bandpass(ecg_signal, fs)
print("ECG signal filtered.")

def detect_r(filtered, fs):
    try:
        wd, _ = hp.process(filtered, sample_rate=fs)
        return np.array(wd['peaklist'])
    except Exception as e:
        print(f"Error during R-peak detection: {e}")
        return np.array([])

r_peaks = detect_r(filtered, fs)
print(f"Detected {len(r_peaks)} R-peaks.")

def detect_pqrst(filtered, r_peaks, fs):
    p, q, s, t = [], [], [], []
    for r in r_peaks:
        r_int = int(r)
        
        q_idx_start = max(0, r_int - int(0.08 * fs))
        q_idx_end = r_int
        q_val = None
        if q_idx_end > q_idx_start:
            q_val_relative = np.argmin(filtered[q_idx_start:q_idx_end])
            q_val = q_idx_start + q_val_relative
            
        s_idx_start = r_int
        s_idx_end = min(len(filtered), r_int + int(0.08 * fs))
        s_val = None
        if s_idx_end > s_idx_start:
            s_val_relative = np.argmin(filtered[s_idx_start:s_idx_end])
            s_val = s_idx_start + s_val_relative

        q.append(q_val)
        s.append(s_val)

        qp = q[-1]
        sp = s[-1]

        p_val = None
        if qp is not None:
            p_idx_start = max(0, int(qp - 0.2 * fs))
            p_idx_end = int(qp)
            if p_idx_end > p_idx_start:
                p_val_relative = np.argmax(filtered[p_idx_start:p_idx_end])
                p_val = p_idx_start + p_val_relative

        t_val = None
        if sp is not None:
            t_idx_start = int(sp)
            t_idx_end = min(len(filtered), int(sp + 0.4 * fs))
            if t_idx_end > t_idx_start:
                t_val_relative = np.argmax(filtered[t_idx_start:t_idx_end])
                t_val = t_idx_start + t_val_relative

        p.append(p_val)
        t.append(t_val)

    return {
        'P': np.array(p),
        'Q': np.array(q),
        'R': np.array(r_peaks),
        'S': np.array(s),
        'T': np.array(t)
    }

info = detect_pqrst(filtered, r_peaks, fs)
print("PQRST waves detected.")

os.makedirs(output_data, exist_ok=True)
plot_path = os.path.join(output_data, f"ecg_plot{record_id}.json")
phases_path = os.path.join(output_data, f"ecg_phases{record_id}.json")

with open(plot_path, 'w') as f:
    json.dump(filtered[:fs * 60].tolist(), f)
print(f"Plot data saved to {plot_path}")

phases = []
waves = {w: [i / fs for i in info[w] if i is not None] for w in ['P', 'Q', 'S', 'T']}

for i in range(min(len(waves.get('P', [])), len(waves.get('Q', [])), len(waves.get('S', [])), len(waves.get('T', [])))):
    try:
        if waves['P'][i] is not None and waves['Q'][i] is not None:
            phases.append({"entry": waves['P'][i], "duration": waves['Q'][i] - waves['P'][i], "phase": "PQ"})
        if waves['Q'][i] is not None and waves['S'][i] is not None:
            phases.append({"entry": waves['Q'][i], "duration": waves['S'][i] - waves['Q'][i], "phase": "QRS"})
        if waves['S'][i] is not None and waves['T'][i] is not None:
            phases.append({"entry": waves['S'][i], "duration": waves['T'][i] - waves['S'][i], "phase": "ST"})
    except IndexError:
        print(f"Warning: Skipping phase calculation for index {i} due to missing wave data.")
        continue
    except Exception as e:
        print(f"Error calculating phases for index {i}: {e}")
        continue

with open(phases_path, 'w') as f:
    json.dump(phases, f, indent=2)
print(f"Phases data saved to {phases_path}")

print(f"✅ Done processing for record_id: {record_id}. Outputs: {plot_path}, {phases_path}")