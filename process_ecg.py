# === process_ecg.py ===
import sys
import os
import json
import numpy as np
import wfdb
import heartpy as hp
from scipy import signal as sp_signal

# ===============================
# 🧠 Input Setup
# ===============================
# Get record number from command-line or default to "100"
record_num = sys.argv[1] if len(sys.argv) > 1 else "100"
project_root = os.path.dirname(os.path.abspath(__file__))

# Paths
default_data = os.path.join(project_root, 'mit-bih-arrhythmia-database-1.0.0')
uploads_data = os.path.join(project_root, 'uploads')
output_data = os.path.join(project_root, 'outputs')

# Determine source of files
record_path = os.path.join(
    uploads_data if all(os.path.exists(os.path.join(uploads_data, f"{record_num}{ext}")) for ext in [".hea", ".dat", ".atr"]) 
    else default_data, 
    record_num
)

# ===============================
# 📊 Load ECG Signal
# ===============================
record = wfdb.rdrecord(record_path)
fs = record.fs  # Sampling rate
ecg_signal = record.p_signal[:, 0]  # Lead 1 signal

# ===============================
# 🧹 Filter Signal
# ===============================
def fir_bandpass(ecg, fs, low=3, high=45, taps=101):
    nyq = 0.5 * fs
    filt = sp_signal.firwin(taps, [low / nyq, high / nyq], pass_zero=False)
    return sp_signal.lfilter(filt, 1.0, ecg)

filtered = fir_bandpass(ecg_signal, fs)

# ===============================
# 📍 Detect R Peaks
# ===============================
def detect_r(filtered, fs):
    wd, _ = hp.process(filtered, sample_rate=fs)
    return np.array(wd['peaklist'])

r_peaks = detect_r(filtered, fs)

# ===============================
# 📍 Detect PQRST
# ===============================
def detect_pqrst(filtered, r_peaks, fs):
    p, q, s, t = [], [], [], []
    for r in r_peaks:
        q_idx = max(0, r - int(0.08 * fs))
        s_idx = min(len(filtered), r + int(0.08 * fs))

        q_val = q_idx + np.argmin(filtered[q_idx:r]) if r > q_idx else None
        s_val = r + np.argmin(filtered[r:s_idx]) if s_idx > r else None

        q.append(q_val)
        s.append(s_val)

        qp = q[-1]
        sp = s[-1]

        p_val = max(0, qp - int(0.2 * fs)) + np.argmax(filtered[max(0, qp - int(0.2 * fs)):qp]) if qp is not None else None
        t_val = sp + np.argmax(filtered[sp:min(len(filtered), sp + int(0.4 * fs))]) if sp is not None else None

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

# ===============================
# 💾 Save Output JSON Files
# ===============================
os.makedirs(output_data, exist_ok=True)
plot_path = os.path.join(output_data, f"ecg_plot{record_num}.json")
phases_path = os.path.join(output_data, f"ecg_phases{record_num}.json")

# Save filtered signal (first 60 seconds)
with open(plot_path, 'w') as f:
    json.dump(filtered[:fs * 60].tolist(), f)

# Prepare phase data
phases = []
waves = {w: [i / fs for i in info[w] if i is not None] for w in ['P', 'Q', 'S', 'T']}

for i in range(min(map(len, waves.values()))):
    try:
        phases.append({"entry": waves['P'][i], "duration": waves['Q'][i] - waves['P'][i], "phase": "PQ"})
        phases.append({"entry": waves['Q'][i], "duration": waves['S'][i] - waves['Q'][i], "phase": "QRS"})
        phases.append({"entry": waves['S'][i], "duration": waves['T'][i] - waves['S'][i], "phase": "ST"})
    except Exception:
        continue

with open(phases_path, 'w') as f:
    json.dump(phases, f, indent=2)

# ===============================
# ✅ Done
# ===============================
print(f"✅ Done: {plot_path}, {phases_path}")
