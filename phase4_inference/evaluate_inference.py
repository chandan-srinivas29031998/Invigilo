"""

Phase 4 Evaluation: Compare Inference Timeline vs Ground Truth Event Log


Compares the per-second inference timeline CSV (from phase4_inference.py)
against the manually annotated ground truth event log to compute:

  1. EVENT-LEVEL DETECTION:  Was each GT suspicious event caught?
  2. CLASS ACCURACY:         Was the correct merged class predicted?
  3. FALSE ALARM ANALYSIS:   How often did the model flag during quiet periods?
  4. PER-PERSON BREAKDOWN:   Which tracked people were over/under-flagged?
  5. TEMPORAL ANALYSIS:      Detection latency and flag duration

"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_style('whitegrid')


# 
# CONFIGURATION
# 

# Maps original 5 suspicious classes to 3 merged classes (matching Phase 3 v9)
MERGE_MAP = {
    'looking_sideways':     'head_turn',
    'talking_to_neighbor':  'head_turn',
    'leaning_to_neighbor':  'lateral_movement',
    'passing_note':         'lateral_movement',
    'looking_down':         'looking_down',
    'using_phone':          'using_phone',   # not in pose model
    'normal':               'normal',
}

# Classes the pose model can detect (using_phone is handled separately by YOLOv8)
POSE_CLASSES = {'head_turn', 'lateral_movement', 'looking_down'}

# Tolerance: GT event counts as "detected" if ANY suspicious prediction
# occurs within this many seconds of the event window (accounts for
# classification latency from the 3-second sliding window)
DETECTION_TOLERANCE_SEC = 2


# HELPERS 

def parse_mmss(time_str):
    """Convert 'MM:SS' string to total seconds."""
    parts = str(time_str).strip().split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0


def sec_to_mmss(sec):
    """Convert seconds to 'M:SS' string."""
    m = int(sec) // 60
    s = int(sec) % 60
    return f'{m}:{s:02d}'

# LOADING

def load_ground_truth(csv_path, video_id):
    """ Load and parse the ground truth event log for a specific video. """

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df.columns = [c.strip() for c in df.columns]

    # Filter to the requested video
    df = df[df['video_id'] == video_id].copy()
    if len(df) == 0:
        raise ValueError(f'No events found for video_id="{video_id}" in {csv_path}')

    # Parse times
    df['start_sec'] = df['start_time'].apply(parse_mmss)
    df['end_sec'] = df['end_time'].apply(parse_mmss)
    df['duration_sec'] = df['end_sec'] - df['start_sec']

    # Add merged class
    df['merged_class'] = df['behavior_class'].map(MERGE_MAP).fillna('unknown')

    # Flag whether the pose model can detect this class
    df['pose_detectable'] = df['merged_class'].isin(POSE_CLASSES)

    # Split
    gt_sus = df[df['behavior_class'] != 'normal'].copy()
    gt_norm = df[df['behavior_class'] == 'normal'].copy()

    return df, gt_sus, gt_norm


def load_timeline(csv_path):
    """Load the Phase 4 inference timeline CSV."""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df.columns = [c.strip() for c in df.columns]
    return df

# EVENT-LEVEL MATCHING

def match_events(gt_sus, timeline, tolerance=DETECTION_TOLERANCE_SEC):
    """
    For each ground truth suspicious event, determine if the model detected it.

    Detection criteria (evaluated in order):
    1. DETECTED:     Any person classified as suspicious during the event window
    2. FLAGGED:      Any person triggered the 3-of-5 flagging rule during the window
    3. CLASS_MATCH:  The correct merged class was predicted during the window
    4. LATENCY:      Seconds from event start to first suspicious prediction

    """
    max_timeline_sec = timeline['time_sec'].max() if len(timeline) > 0 else 0
    results = []

    for _, event in gt_sus.iterrows():
        s = event['start_sec']
        e = event['end_sec']
        seat = event.get('target_seat_id', '')
        gt_class = event['behavior_class']
        merged = event['merged_class']

        # Check if this event falls within the timeline's coverage
        in_coverage = s <= max_timeline_sec

        # Expand window by tolerance to account for classification latency
        s_tol = max(0, s - tolerance)
        e_tol = e + tolerance

        # Find all timeline entries during this expanded window
        window = timeline[
            (timeline['time_sec'] >= s_tol) &
            (timeline['time_sec'] <= e_tol)
        ]

        # Detection checks
        sus_window = window[window['label'] != 'normal']
        any_detected = len(sus_window) > 0
        any_flagged = window['is_flagged'].any() if len(window) > 0 else False

        # Class match: check if the correct merged class was predicted
        if merged == 'using_phone':
            # Phone is not in the pose model — count ANY suspicious as a partial match
            class_match = any_detected
            class_note = 'phone→any_sus'
        else:
            class_match = (window['label'] == merged).any() if len(window) > 0 else False
            class_note = ''

        # Detection latency: seconds from event start to first suspicious prediction
        latency = np.nan
        if any_detected:
            first_sus_time = sus_window['time_sec'].min()
            latency = max(0, first_sus_time - s)

        # Number of suspicious predictions during the window (helps gauge how strongly the model detected the event)
        n_sus_preds = len(sus_window)
        n_total_preds = len(window)

        results.append({
            'event_idx': event.name,
            'start_time': event.get('start_time', sec_to_mmss(s)),
            'end_time': event.get('end_time', sec_to_mmss(e)),
            'start_sec': s,
            'end_sec': e,
            'duration_sec': e - s,
            'target_seat': seat,
            'gt_class': gt_class,
            'merged_class': merged,
            'pose_detectable': merged in POSE_CLASSES,
            'in_coverage': in_coverage,
            'detected': any_detected,
            'flagged': any_flagged,
            'class_correct': class_match,
            'class_note': class_note,
            'latency_sec': latency,
            'n_sus_predictions': n_sus_preds,
            'n_total_predictions': n_total_preds,
        })

    return pd.DataFrame(results)

# TIME-BASED ANALYSIS

def time_based_analysis(gt_sus, gt_norm, timeline):
    """ Classify every second of the video as:
    - Active:  at least one GT suspicious event is happening
    - Hard negative: a GT normal (hard negative) event is happening, no suspicious
    - Quiet:   nothing annotated is happening

    Then check the model's behaviour during each category. """
    
    max_t = int(timeline['time_sec'].max()) if len(timeline) > 0 else 0

    # Build per-second arrays
    sus_active = np.zeros(max_t + 1, dtype=bool)      # Any suspicious event
    hn_active = np.zeros(max_t + 1, dtype=bool)        # Hard negative event
    model_sus = np.zeros(max_t + 1, dtype=bool)        # Model says suspicious
    model_flagged = np.zeros(max_t + 1, dtype=bool)    # Model flagged (3-of-5)

    for _, ev in gt_sus.iterrows():
        s, e = ev['start_sec'], min(ev['end_sec'], max_t)
        sus_active[s:e + 1] = True

    for _, ev in gt_norm.iterrows():
        s, e = ev['start_sec'], min(ev['end_sec'], max_t)
        hn_active[s:e + 1] = True

    for _, row in timeline.iterrows():
        t = int(row['time_sec'])
        if t <= max_t:
            if row['label'] != 'normal':
                model_sus[t] = True
            if row['is_flagged']:
                model_flagged[t] = True

    # Classify each second
    # Priority: suspicious > hard_negative > quiet
    categories = np.full(max_t + 1, 'quiet', dtype=object)
    categories[hn_active & ~sus_active] = 'hard_negative'
    categories[sus_active] = 'suspicious'

    stats = {}
    for cat in ['suspicious', 'hard_negative', 'quiet']:
        mask = categories == cat
        n = int(mask.sum())
        if n == 0:
            stats[cat] = {'n_seconds': 0, 'model_sus_rate': 0, 'model_flag_rate': 0}
            continue
        stats[cat] = {
            'n_seconds': n,
            'model_sus_rate': float(model_sus[mask].sum()) / n,
            'model_flag_rate': float(model_flagged[mask].sum()) / n,
        }

    return stats, categories, sus_active, hn_active, model_sus, model_flagged

# PER-PERSON ANALYSIS

def per_person_analysis(timeline):
    """Compute per-tracked-person statistics."""
    results = []
    for tid in sorted(timeline['track_id'].unique()):
        person = timeline[timeline['track_id'] == tid]
        n_obs = len(person)
        n_sus = int((person['label'] != 'normal').sum())
        n_flag = int(person['is_flagged'].sum())
        t_start = person['time_sec'].min()
        t_end = person['time_sec'].max()
        duration = t_end - t_start + 1

        # Label breakdown
        label_counts = person['label'].value_counts().to_dict()

        results.append({
            'track_id': tid,
            'person_label': f'P{tid + 1}',
            'n_observations': n_obs,
            'time_range': f'{sec_to_mmss(t_start)}-{sec_to_mmss(t_end)}',
            'duration_sec': duration,
            'n_suspicious': n_sus,
            'suspicious_rate': n_sus / max(n_obs, 1),
            'n_flagged': n_flag,
            'flagged_rate': n_flag / max(n_obs, 1),
            'label_breakdown': label_counts,
        })

    return pd.DataFrame(results)

# REPORT GENERATION

def generate_report(video_id, gt_all, gt_sus, gt_norm, timeline,
                    event_results, time_stats, person_stats, output_dir):
    """Generate the full evaluation report as a text file."""
    report_path = output_dir / f'{video_id}_evaluation_report.txt'
    max_t = int(timeline['time_sec'].max()) if len(timeline) > 0 else 0

    # Filter to in-coverage events
    covered = event_results[event_results['in_coverage']]
    beyond = event_results[~event_results['in_coverage']]

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('=' * 70 + '\n')
        f.write(f'  PHASE 4 EVALUATION REPORT — {video_id}\n')
        f.write('=' * 70 + '\n\n')

        # -- Overview --
        f.write('1. OVERVIEW\n')
        f.write('-' * 40 + '\n')
        f.write(f'  Video duration:         {gt_all["duration_seconds"].iloc[0]}s\n')
        f.write(f'  Timeline coverage:      0-{max_t}s ({max_t / gt_all["duration_seconds"].iloc[0] * 100:.0f}%)\n')
        f.write(f'  GT events (total):      {len(gt_all)}\n')
        f.write(f'  GT suspicious events:   {len(gt_sus)}\n')
        f.write(f'  GT hard negatives:      {len(gt_norm)}\n')
        f.write(f'  Timeline observations:  {len(timeline)}\n')
        f.write(f'  Tracked people:         {timeline["track_id"].nunique()}\n\n')

        # -- GT class distribution --
        f.write('  GT suspicious class distribution:\n')
        for cls, cnt in gt_sus['behavior_class'].value_counts().items():
            merged = MERGE_MAP.get(cls, cls)
            f.write(f'    {cls:<25s} {cnt:3d}  → {merged}\n')

        # -- Event-level detection --
        f.write(f'\n\n2. EVENT-LEVEL DETECTION (within coverage, {len(covered)} events)\n')
        f.write('-' * 40 + '\n')

        n_det = int(covered['detected'].sum())
        n_flag = int(covered['flagged'].sum())
        n_cls = int(covered['class_correct'].sum())
        n_cov = len(covered)

        f.write(f'  Detected (any suspicious):  {n_det:3d} / {n_cov} ({n_det / max(n_cov, 1) * 100:.1f}%)\n')
        f.write(f'  Flagged (3-of-5 rule):      {n_flag:3d} / {n_cov} ({n_flag / max(n_cov, 1) * 100:.1f}%)\n')
        f.write(f'  Class correct:              {n_cls:3d} / {n_cov} ({n_cls / max(n_cov, 1) * 100:.1f}%)\n')

        # Latency stats
        latencies = covered.loc[covered['detected'], 'latency_sec'].dropna()
        if len(latencies) > 0:
            f.write(f'\n  Detection latency (from event start):\n')
            f.write(f'    Mean:   {latencies.mean():.1f}s\n')
            f.write(f'    Median: {latencies.median():.1f}s\n')
            f.write(f'    Max:    {latencies.max():.1f}s\n')

        # Pose-only (excluding using_phone)
        pose_only = covered[covered['pose_detectable']]
        n_po = len(pose_only)
        if n_po > 0:
            f.write(f'\n  Pose-detectable events only (excl. using_phone):\n')
            f.write(f'    Detected: {int(pose_only["detected"].sum()):3d} / {n_po} ({pose_only["detected"].sum() / n_po * 100:.1f}%)\n')
            f.write(f'    Flagged:  {int(pose_only["flagged"].sum()):3d} / {n_po}\n')
            f.write(f'    Class OK: {int(pose_only["class_correct"].sum()):3d} / {n_po}\n')

        # Phone events (detected as any suspicious)
        phone_evts = covered[covered['merged_class'] == 'using_phone']
        if len(phone_evts) > 0:
            n_phone_det = int(phone_evts['detected'].sum())
            f.write(f'\n  using_phone events (not in pose model, detected as any suspicious):\n')
            f.write(f'    Detected: {n_phone_det} / {len(phone_evts)}\n')

        if len(beyond) > 0:
            f.write(f'\n  Events beyond timeline coverage: {len(beyond)} (not evaluated)\n')

        # -- Per-event detail --
        f.write(f'\n\n3. PER-EVENT DETAIL\n')
        f.write('-' * 40 + '\n')
        f.write(f'  {"Time":<12s} {"Seat":<5s} {"GT Class":<25s} {"Det":>4s} {"Flag":>5s} {"Class":>6s} {"Lat":>5s}\n')
        f.write(f'  {"─" * 65}\n')
        for _, r in covered.iterrows():
            det = '✓' if r['detected'] else '✗'
            flag = '✓' if r['flagged'] else ''
            cls = '✓' if r['class_correct'] else '✗'
            lat = f'{r["latency_sec"]:.0f}s' if not np.isnan(r['latency_sec']) else '—'
            note = f'  ({r["class_note"]})' if r['class_note'] else ''
            f.write(f'  {r["start_time"]:<5s}-{r["end_time"]:<5s} {r["target_seat"]:<5s} '
                    f'{r["gt_class"]:<25s} {det:>4s} {flag:>5s} {cls:>6s} {lat:>5s}{note}\n')

        # -- Missed events --
        missed = covered[~covered['detected']]
        if len(missed) > 0:
            f.write(f'\n  MISSED EVENTS ({len(missed)}):\n')
            for _, r in missed.iterrows():
                f.write(f'    {r["start_time"]}-{r["end_time"]} {r["target_seat"]} {r["gt_class"]}\n')

        # -- Time-based analysis --
        f.write(f'\n\n4. TIME-BASED ANALYSIS (0-{max_t}s)\n')
        f.write('-' * 40 + '\n')
        f.write(f'  {"Period":<20s} {"Seconds":>8s} {"Model sus%":>11s} {"Model flag%":>12s}\n')
        f.write(f'  {"─" * 55}\n')
        for cat in ['suspicious', 'hard_negative', 'quiet']:
            st = time_stats[cat]
            n = st['n_seconds']
            sr = st['model_sus_rate'] * 100
            fr = st['model_flag_rate'] * 100
            f.write(f'  {cat:<20s} {n:>8d} {sr:>10.1f}% {fr:>11.1f}%\n')

        f.write(f'\n  Interpretation:\n')
        f.write(f'    "suspicious" = GT suspicious event active → model SHOULD flag\n')
        f.write(f'    "hard_negative" = GT normal event (fidgeting etc.) → model should NOT flag\n')
        f.write(f'    "quiet" = nothing annotated → model should NOT flag\n')

        # -- Per-person breakdown --
        f.write(f'\n\n5. PER-PERSON BREAKDOWN\n')
        f.write('-' * 40 + '\n')
        f.write(f'  {"Person":<8s} {"Obs":>5s} {"Sus":>5s} {"Sus%":>6s} {"Flag":>5s} {"Flag%":>7s} {"Time Range":<15s}\n')
        f.write(f'  {"─" * 55}\n')
        for _, p in person_stats.iterrows():
            f.write(f'  {p["person_label"]:<8s} {p["n_observations"]:>5d} '
                    f'{p["n_suspicious"]:>5d} {p["suspicious_rate"]:>5.0%} '
                    f'{p["n_flagged"]:>5d} {p["flagged_rate"]:>6.0%}  '
                    f'{p["time_range"]:<15s}\n')
            # Show non-normal labels
            for lbl, cnt in p['label_breakdown'].items():
                if lbl != 'normal':
                    f.write(f'           {lbl}: {cnt}\n')

        # -- Tracker stability --
        n_unique_tracks = timeline['track_id'].nunique()
        expected_people = 8
        f.write(f'\n\n6. TRACKER STABILITY\n')
        f.write('-' * 40 + '\n')
        f.write(f'  Unique track IDs:    {n_unique_tracks}\n')
        f.write(f'  Expected people:     {expected_people}\n')
        f.write(f'  ID reassignments:    {max(0, n_unique_tracks - expected_people)}\n')

        # Find tracks that appeared and disappeared
        stable = person_stats[person_stats['duration_sec'] > max_t * 0.8]
        transient = person_stats[person_stats['duration_sec'] <= max_t * 0.8]
        f.write(f'  Stable tracks (>80% coverage): {len(stable)}\n')
        f.write(f'  Transient tracks:              {len(transient)}\n')
        if len(transient) > 0:
            for _, p in transient.iterrows():
                f.write(f'    {p["person_label"]} (ID={p["track_id"]}): {p["time_range"]}, {p["n_observations"]} obs\n')

        # -- Summary --
        f.write(f'\n\n7. SUMMARY\n')
        f.write('=' * 70 + '\n')
        f.write(f'  Event detection rate:     {n_det}/{n_cov} ({n_det / max(n_cov, 1) * 100:.0f}%)\n')
        f.write(f'  Event flagging rate:      {n_flag}/{n_cov} ({n_flag / max(n_cov, 1) * 100:.0f}%)\n')
        f.write(f'  Class accuracy:           {n_cls}/{n_cov} ({n_cls / max(n_cov, 1) * 100:.0f}%)\n')

        quiet_fa = time_stats['quiet']['model_flag_rate'] * 100
        hn_fa = time_stats['hard_negative']['model_flag_rate'] * 100
        f.write(f'  False alarm (quiet):      {quiet_fa:.0f}% of quiet seconds\n')
        f.write(f'  False alarm (hard neg):   {hn_fa:.0f}% of hard negative seconds\n')
        f.write(f'  Tracker stability:        {len(stable)}/{expected_people} stable tracks\n')

    print(f'  Report saved: {report_path}')
    return report_path

# VISUALISATION 

def plot_temporal_heatmap(video_id, categories, sus_active, model_sus, model_flagged,
                          gt_sus, timeline, output_dir):
    """
    Plot a temporal heatmap showing GT events vs model predictions over time.
    """
    max_t = len(categories) - 1
    fig, axes = plt.subplots(3, 1, figsize=(18, 7), sharex=True,
                              gridspec_kw={'height_ratios': [1.5, 1, 1]})

    # --- Panel 1: Ground truth events ---
    ax = axes[0]
    ax.set_title(f'{video_id} — Ground Truth Events', fontsize=11, fontweight='bold')
    y_map = {}
    y_counter = 0
    colors_gt = {
        'looking_sideways': '#E74C3C', 'talking_to_neighbor': '#C0392B',
        'leaning_to_neighbor': '#E67E22', 'passing_note': '#F39C12',
        'looking_down': '#9B59B6', 'using_phone': '#3498DB',
        'normal': '#95A5A6',
    }
    for _, ev in gt_sus.iterrows():
        seat = ev.get('target_seat_id', '')
        if seat not in y_map:
            y_map[seat] = y_counter
            y_counter += 1
        y = y_map[seat]
        s, e = ev['start_sec'], ev['end_sec']
        c = colors_gt.get(ev['behavior_class'], '#7F8C8D')
        ax.barh(y, e - s, left=s, height=0.7, color=c, alpha=0.8,
                edgecolor='white', linewidth=0.5)
    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(list(y_map.keys()), fontsize=8)
    ax.set_ylabel('Seat', fontsize=9)
    ax.set_xlim(0, max_t)

    # Legend for GT
    legend_patches = [mpatches.Patch(color=c, label=l) for l, c in colors_gt.items()
                       if l != 'normal']
    ax.legend(handles=legend_patches, loc='upper right', fontsize=7, ncol=3)

    # --- Panel 2: Model predictions per person ---
    ax = axes[1]
    ax.set_title('Model Predictions (per tracked person)', fontsize=11, fontweight='bold')
    colors_model = {'head_turn': '#E74C3C', 'lateral_movement': '#E67E22',
                     'looking_down': '#9B59B6', 'normal': '#2ECC71'}
    stable_ids = sorted(timeline['track_id'].unique())[:8]  # Top 8 stable tracks
    for i, tid in enumerate(stable_ids):
        person = timeline[timeline['track_id'] == tid]
        for _, row in person.iterrows():
            t = row['time_sec']
            label = row['label']
            c = colors_model.get(label, '#BDC3C7')
            alpha = 0.9 if label != 'normal' else 0.2
            ax.barh(i, 1, left=t, height=0.7, color=c, alpha=alpha, linewidth=0)
    ax.set_yticks(range(len(stable_ids)))
    ax.set_yticklabels([f'P{tid + 1}' for tid in stable_ids], fontsize=8)
    ax.set_ylabel('Track', fontsize=9)
    legend_patches2 = [mpatches.Patch(color=c, label=l) for l, c in colors_model.items()]
    ax.legend(handles=legend_patches2, loc='upper right', fontsize=7, ncol=4)

    # --- Panel 3: Time category + model flags ---
    ax = axes[2]
    ax.set_title('Period Classification + Model Flags', fontsize=11, fontweight='bold')
    cat_colors = {'suspicious': '#FADBD8', 'hard_negative': '#FCF3CF', 'quiet': '#D5F5E3'}
    for t in range(max_t + 1):
        ax.axvspan(t, t + 1, color=cat_colors.get(categories[t], '#FFFFFF'), alpha=0.5)

    # Overlay model flags
    flag_times = [t for t in range(max_t + 1) if model_flagged[t]]
    if flag_times:
        ax.scatter(flag_times, [0.5] * len(flag_times), c='red', s=8,
                    marker='|', label='Model flagged', zorder=5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Time (seconds)', fontsize=10)

    legend_patches3 = [
        mpatches.Patch(color='#FADBD8', label='GT suspicious'),
        mpatches.Patch(color='#FCF3CF', label='GT hard negative'),
        mpatches.Patch(color='#D5F5E3', label='Quiet'),
    ]
    ax.legend(handles=legend_patches3, loc='upper right', fontsize=7, ncol=3)

    plt.tight_layout()
    save_path = output_dir / f'{video_id}_temporal_heatmap.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Heatmap saved: {save_path}')


def plot_summary_charts(video_id, event_results, time_stats, person_stats, output_dir):
    """Plot summary bar charts."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Chart 1: Event detection by class ---
    ax = axes[0]
    covered = event_results[event_results['in_coverage']]
    if len(covered) > 0:
        by_class = covered.groupby('gt_class').agg(
            total=('detected', 'count'),
            detected=('detected', 'sum'),
            flagged=('flagged', 'sum'),
        ).reset_index()
        x = range(len(by_class))
        w = 0.35
        ax.bar([i - w / 2 for i in x], by_class['total'], w, label='GT events', color='#BDC3C7')
        ax.bar([i + w / 2 for i in x], by_class['detected'], w, label='Detected', color='#2ECC71')
        ax.set_xticks(x)
        ax.set_xticklabels(by_class['gt_class'], rotation=45, ha='right', fontsize=8)
        ax.set_title('Detection by GT Class')
        ax.legend(fontsize=8)
        ax.set_ylabel('Count')

    # --- Chart 2: False alarm rates by period ---
    ax = axes[1]
    cats = ['suspicious', 'hard_negative', 'quiet']
    flag_rates = [time_stats[c]['model_flag_rate'] * 100 for c in cats]
    colors_bar = ['#2ECC71', '#F39C12', '#E74C3C']
    ax.bar(cats, flag_rates, color=colors_bar, edgecolor='white')
    for i, (c, r) in enumerate(zip(cats, flag_rates)):
        n = time_stats[c]['n_seconds']
        ax.text(i, r + 1, f'n={n}s', ha='center', fontsize=8)
    ax.set_title('Model Flag Rate by Period Type')
    ax.set_ylabel('% of seconds flagged')
    ax.set_ylim(0, 110)

    # --- Chart 3: Per-person suspicious rate ---
    ax = axes[2]
    stable = person_stats[person_stats['duration_sec'] > 10].head(8)
    if len(stable) > 0:
        ax.bar(stable['person_label'], stable['suspicious_rate'] * 100,
               color='steelblue', edgecolor='white')
        for i, (_, p) in enumerate(stable.iterrows()):
            ax.text(i, p['suspicious_rate'] * 100 + 1,
                    f'{p["n_suspicious"]}/{p["n_observations"]}',
                    ha='center', fontsize=7)
        ax.set_title('Per-Person Suspicious Rate')
        ax.set_ylabel('% observations suspicious')
        ax.set_ylim(0, 100)

    plt.suptitle(f'{video_id} — Evaluation Summary', fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = output_dir / f'{video_id}_summary_charts.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Charts saved: {save_path}')

# MAIN EVALUATION FUNCTION 

def evaluate_video(timeline_path, gt_path, video_id, output_dir):
    """Run the full evaluation pipeline for one video."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"=" * 60}')
    print(f'  Evaluating: {video_id}')
    print(f'{"=" * 60}')

    # Load data
    gt_all, gt_sus, gt_norm = load_ground_truth(gt_path, video_id)
    timeline = load_timeline(timeline_path)

    print(f'  GT events: {len(gt_all)} ({len(gt_sus)} suspicious, {len(gt_norm)} hard negatives)')
    print(f'  Timeline: {len(timeline)} observations, {timeline["time_sec"].max():.0f}s coverage')

    # Event-level matching
    event_results = match_events(gt_sus, timeline)

    # Save event matching CSV
    event_csv_path = output_dir / f'{video_id}_event_matching.csv'
    event_results.to_csv(event_csv_path, index=False)
    print(f'  Event matching saved: {event_csv_path}')

    # Time-based analysis
    time_stats, categories, sus_active, hn_active, model_sus, model_flagged = \
        time_based_analysis(gt_sus, gt_norm, timeline)

    # Per-person analysis
    person_stats = per_person_analysis(timeline)

    # Generate report
    generate_report(video_id, gt_all, gt_sus, gt_norm, timeline,
                    event_results, time_stats, person_stats, output_dir)

    # Generate plots
    plot_temporal_heatmap(video_id, categories, sus_active, model_sus, model_flagged,
                          gt_sus, timeline, output_dir)
    plot_summary_charts(video_id, event_results, time_stats, person_stats, output_dir)

    # Print quick summary
    covered = event_results[event_results['in_coverage']]
    n = len(covered)
    print(f'\n  QUICK SUMMARY:')
    print(f'    Detection rate:    {int(covered["detected"].sum())}/{n} ({covered["detected"].sum() / max(n, 1) * 100:.0f}%)')
    print(f'    Flagging rate:     {int(covered["flagged"].sum())}/{n} ({covered["flagged"].sum() / max(n, 1) * 100:.0f}%)')
    print(f'    Class accuracy:    {int(covered["class_correct"].sum())}/{n} ({covered["class_correct"].sum() / max(n, 1) * 100:.0f}%)')
    print(f'    Quiet false alarm: {time_stats["quiet"]["model_flag_rate"] * 100:.0f}%')

# CLI 

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Phase 4 inference results against ground truth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate a single video
    python evaluate_inference.py \\
        --timeline output/V01_timeline.csv \\
        --ground_truth long_video_event_log.csv \\
        --video_id V01

    # Evaluate all videos in a directory
    python evaluate_inference.py \\
        --timeline_dir output/ \\
        --ground_truth long_video_event_log.csv
        """
    )
    parser.add_argument('--timeline', type=str, help='Path to a single timeline CSV')
    parser.add_argument('--timeline_dir', type=str, help='Directory of timeline CSVs')
    parser.add_argument('--ground_truth', type=str, required=True,
                        help='Path to ground truth event log CSV')
    parser.add_argument('--video_id', type=str, help='Video ID (e.g., V01). Required with --timeline')
    parser.add_argument('--output', type=str, default='./evaluation_results',
                        help='Output directory (default: ./evaluation_results)')

    args = parser.parse_args()

    if not os.path.exists(args.ground_truth):
        print(f'ERROR: Ground truth not found: {args.ground_truth}')
        sys.exit(1)

    if args.timeline:
        if not args.video_id:
            # Try to infer from filename
            stem = Path(args.timeline).stem.replace('_timeline', '')
            args.video_id = stem
            print(f'  Inferred video_id: {args.video_id}')

        evaluate_video(args.timeline, args.ground_truth, args.video_id, args.output)

    elif args.timeline_dir:
        tl_dir = Path(args.timeline_dir)
        timelines = sorted(tl_dir.glob('*_timeline.csv'))
        if not timelines:
            print(f'No *_timeline.csv files found in {tl_dir}')
            sys.exit(1)

        print(f'Found {len(timelines)} timeline files')
        for tl_path in timelines:
            vid = tl_path.stem.replace('_timeline', '')
            try:
                evaluate_video(str(tl_path), args.ground_truth, vid, args.output)
            except Exception as e:
                print(f'  ERROR processing {vid}: {e}')
    else:
        parser.error('Provide --timeline or --timeline_dir')


if __name__ == '__main__':
    main()
