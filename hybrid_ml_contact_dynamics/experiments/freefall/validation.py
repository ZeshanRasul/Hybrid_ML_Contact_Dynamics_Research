import numpy as np
import json
from hybrid_ml_contact_dynamics.physics.primitives.circle import Circle
from hybrid_ml_contact_dynamics.physics.primitives.plane import Plane

def validate_restitution(j, data, circle: Circle, plane: Plane, dt: float, run_count, date):
    y = data['position'][:, 1]
    vy = data['velocity'][:, 1]
    t = data['time'][:, ]
    e = data['e'][:, ]
    h = y - circle.get_radius() - plane.get_offset()
    impact_times = list()
    impact_steps = list()
    impact_indices = []
    e_estimates = []
    x_input = []
    v_eps = 0.5
    pos_eps = 1e-3
    W = 7
    half = 7 // 3
    sigma = 0.25
    jitter = 0
    vy_windows = list()
    h_windows = list()
    vyidx = 0
    hidx = 0
    for i in range (len(vy) - 1):
        if (i - half >= 0 and i + half <= (len(vy) -1) and vy[i] < -v_eps and vy[i+1] > v_eps):
            e_est_i = vy[i+1] / (-vy[i])
            e_estimates.append(e_est_i)
            impact_indices.append(i)
            vy_window = vy[i-half : i + half]
            vy_noise = np.random.normal(0, 1, len(vy_window))
            vy_window = vy_window + vy_noise
            vy_window = vy_window.tolist()
           # vy_window = np.asarray(vy_window, dtype=np.float64)

            h_window = h[i-half : i + half]
            h_window = h_window.tolist()
          #  h_window = np.asarray(h_window, dtype=np.float64)
            vy_windows.append(vy_window)
            h_windows.append(h_window)
            vyidx += 1
            hidx += 1

 #   impact_indices = np.asarray(impact_indices, dtype=np.int64)
    e_estimates = np.asarray(e_estimates, dtype=np.float64)
   # e = np.asarray(e, dtype=np.float64)
  #  vy_windows = np.asarray(vy_windows, dtype=np.array)
  #  h_windows = np.asarray(h_windows, dtype=np.array)

    impact_times = t[impact_indices]
    impact_steps = np.diff(impact_indices)
    impact_dt = np.diff(impact_times)

    h_peaks = []
    start = 0
    for index in list(impact_indices) + [len(h) - 1]:
        end = int(index)
        h_peaks.append(np.max(h[start:end+1]))
        start = end + 1
  
    h_peaks = np.asarray(h_peaks, dtype=np.float64)

    h_min = 1e-3
    valid = h_peaks > h_min
    h_peaks_valid = h_peaks[valid]

    h_ratios = h_peaks_valid[2:] / h_peaks_valid[1:-1] if len(h_peaks_valid) >= 2 else np.array([])

    h_ratios_mean = 0
    h_ratios_min = 0
    h_ratios_max = 0

    if (len(h_ratios > 0)):
        h_ratios_mean = h_ratios.mean()
        h_ratios_min = h_ratios.min()
        h_ratios_max = h_ratios.max()

    e_estimates_mean = 0
    e_estimates_min = 0
    e_estimates_max = 0
    e_estimates_std = 0

    if (len(e_estimates) > 0):
        e_estimates_mean = e_estimates.mean()
        e_estimates_min = e_estimates.min()
        e_estimates_max = e_estimates.max()
        e_estimates_std = e_estimates.std()

        e_estimates = e_estimates.tolist()
    data = {
        'Delta Time': dt,
        'e estimates mean': e_estimates_mean,
        'e actual mean': e.mean(),
        'e estimates std': e_estimates_std,
        'e actual std': e.std(),
        'e estimates min': e_estimates_min,
        'e actual min': e.min(),
        'e estimates max': e_estimates_max,
        'e actual max': e.max(),
        'Impact count': len(impact_indices),
        'Height ratios mean': h_ratios_mean,
        'Height ratios min': h_ratios_min,
        'Height ratios max': h_ratios_max,
        'Y Windows': vy_windows,
        'H Windows': h_windows
        }
    
    with open(f"hybrid_ml_contact_dynamics/experiments/freefall/runs/{j}/{date}/results/validation.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)