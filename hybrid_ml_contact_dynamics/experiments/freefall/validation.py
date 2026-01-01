import numpy as np
import json
from hybrid_ml_contact_dynamics.physics.primitives.circle import Circle
from hybrid_ml_contact_dynamics.physics.primitives.plane import Plane

def validate_restitution(data, circle: Circle, plane: Plane, dt: float, date):
    y = data['position'][:, 1]
    vy = data['velocity'][:, 1]
    t = data['time'][:, ]
    h = y - circle.get_radius() - plane.get_offset()
    impact_times = list()
    impact_steps = list()
    impact_indices = []
    e_estimates = []
    v_eps = 0.5
    for i in range (len(vy) - 1):
        if (vy[i] < v_eps and vy[i+1] > v_eps):
            e_est_i = vy[i+1] / (-vy[i])
            e_estimates.append(e_est_i)
            impact_indices.append(i)


    impact_indices = np.asarray(impact_indices, dtype=np.int64)
    e_estimates = np.asarray(e_estimates, dtype=np.float64)

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

    h_ratios = h_peaks[2:] / h_peaks[1:-1] if len(h_peaks) >= 3 else np.array([])

    data = {
        'Delta Time': dt,
        'e estimates mean: ': e_estimates.mean(),
        'e estimates std: ': e_estimates.std(),
        'e estimates min: ': e_estimates.min(),
        'e estimates max: ': e_estimates.max(),
        'Impact count: ': len(impact_indices),
        'Height ratios mean: ': h_ratios.mean(),
        'Height ratios min: ': h_ratios.min(),
        'Height ratios max: ': h_ratios.max()
        }
    
    with open(f"hybrid_ml_contact_dynamics/experiments/freefall/runs/{date}/results/validation.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)