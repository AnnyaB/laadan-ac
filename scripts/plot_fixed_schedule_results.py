
from __future__ import annotations

import argparse, csv, json, math

from pathlib import Path

from typing import Dict, List

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from matplotlib.patches import Patch

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10.0,
    'axes.titlesize': 12.0,
    'axes.labelsize': 11.0,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9.5,
    'figure.titlesize': 14.5,
    'axes.linewidth': 0.95,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.facecolor': 'white',
    'figure.facecolor': 'white',
})

C = {
    'BC': '#B7B7B7',
    'CQL': '#7D7D7D',
    'VOAC': '#E98B88',
    'POST': '#E6A05B',
    'LAADAN': '#6C9BD2',
    'MASK': '#A7C7E2',
    'NOCQL': '#9786C7',
    'INK': '#2B2B2B',
    'GRID': '#D9D9D9',
}

MAIN = [
    ('Behavior Cloning', 'BC', 'BC'),
    ('CQL-regularised fitted-Q', 'CQL', 'CQL'),
    ('Vanilla Offline Actor-Critic', 'VOAC', 'VOAC'),
    ('Post-hoc Masked VOAC', 'POST', 'Post-hoc\nmask'),
    ('LAADAN-AC', 'LAADAN', 'LAADAN-AC'),
]

SEEDS = [42, 43, 44, 45, 46]

def load_json(p: Path):
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)

def load_csv(p: Path):
    with p.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def val(x):
    if x in (None, ''):
        return np.nan
    return float(x)

def clean(ax, grid='y'):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=C['GRID'], linestyle=(0, (3, 3)), linewidth=0.75, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', width=0.9, length=4)

def panel(ax, letter):
    ax.text(
        -0.14, 1.11, letter,
        transform=ax.transAxes,
        weight='bold',
        fontsize=13.5,
        va='top',
        ha='left',
        clip_on=False
    )

def save(fig, out: Path, stem: str):
    fig.savefig(out / f'{stem}.png', dpi=600, bbox_inches='tight', pad_inches=0.07)
    plt.close(fig)

def per_seed(rows, method, metric):
    return {int(r['seed']): val(r.get(metric, '')) for r in rows if r['method'] == method}

def stat(main, method, metric_name):
    s = main[method][metric_name]
    return float(s['mean']), float(s['ci95_half'])

def jitter_positions(n, width=0.11):
    return np.linspace(-width, width, n)

def bars_with_seeds(
    ax, summary, rows, methods, metric, ylabel,
    ylim=None, percent=False, value_fmt=None,
    xtick_rotation=0, value_pad_frac=0.016
):
    xs = np.arange(len(methods))
    means, cis, colors, tops = [], [], [], []

    for full, key, label in methods:
        m, ci = stat(summary, full, metric)
        if percent:
            m *= 100
            ci *= 100
        means.append(m)
        cis.append(ci)
        colors.append(C[key])

    bars = ax.bar(
        xs, means, width=.62, color=colors,
        edgecolor=C['INK'], linewidth=.7,
        yerr=cis, capsize=3,
        error_kw={'elinewidth': 1.0, 'capthick': 1.0, 'ecolor': C['INK']}
    )

    for i, (full, key, label) in enumerate(methods):
        sm = per_seed(rows, full, metric)
        vals = [sm[s] for s in SEEDS if s in sm and np.isfinite(sm[s])]
        if percent:
            vals = [100 * v for v in vals]
        offs = jitter_positions(len(vals))
        if len(vals) > 0:
            ax.scatter(
                np.full(len(vals), i) + offs, vals,
                s=26, facecolor='white', edgecolor='#555555',
                linewidth=.9, zorder=4
            )
            tops.append(max(max(vals), means[i] + cis[i]))
        else:
            tops.append(means[i] + cis[i])

    ax.set_xticks(xs, [m[2] for m in methods], rotation=xtick_rotation)
    ax.tick_params(axis='x', pad=5)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.margins(x=0.05)

    if ylim:
        ax.set_ylim(*ylim)

    clean(ax)

    if value_fmt:
        yr = ax.get_ylim()[1] - ax.get_ylim()[0]
        for b, m, top in zip(bars, means, tops):
            label_y = top + value_pad_frac * yr
            label_y = min(label_y, ax.get_ylim()[1] - 0.03 * yr)
            ax.text(
                b.get_x() + b.get_width() / 2,
                label_y,
                value_fmt.format(m),
                ha='center',
                va='bottom',
                fontsize=8.8,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.92)
            )
    return bars

def t_ci(vals):
    vals = np.array(vals, dtype=float)
    n = len(vals)
    mean = float(np.mean(vals))
    if n < 2:
        return mean, 0.0
    tcrit = {2: 12.7062, 3: 4.3027, 4: 3.1824, 5: 2.77645}.get(n, 1.96)
    ci = tcrit * float(np.std(vals, ddof=1)) / math.sqrt(n)
    return mean, ci

def fig1(main, rows, out):
    fig = plt.figure(figsize=(16.4, 4.9))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.78], wspace=0.42)
    specs = [
        ('survival_rate', 'Exact survival / return', (0.755, 0.802), False, '{:.3f}'),
        ('inadmissibility_rate', 'Selected-action inadmissibility (%)', (0, 27), True, '{:.2f}'),
        ('expert_argmax_match', 'Expert argmax match', (0, 1.04), False, '{:.3f}'),
        ('mean_kl_to_expert', 'KL to expert', (0, 17.8), False, '{:.2f}'),
    ]
    for i, (met, yl, ylim, pct, fmt) in enumerate(specs):
        ax = fig.add_subplot(gs[0, i])
        bars_with_seeds(ax, main, rows, MAIN, met, yl, ylim, pct, fmt)
        panel(ax, chr(65 + i))

    ax = fig.add_subplot(gs[0, 4])
    ax.axis('off')
    ax.text(0, 1, 'Experiment', weight='bold', fontsize=12, va='top')
    details = (
        'ICU-Sepsis\n'
        '5 seeds: 42–46\n'
        '1,000 training epochs\n'
        'Predetermined final checkpoint\n'
        'Exact finite-horizon evaluation\n'
        'Mean ± 95% t-CI'
    )
    ax.text(0, .90, details, va='top', linespacing=1.7, fontsize=10.1, color=C['INK'])
    legend = [Patch(facecolor=C[k], edgecolor=C['INK'], label=l.replace('\n', ' ')) for _, k, l in MAIN]
    ax.legend(handles=legend, loc='lower left', bbox_to_anchor=(-.03, .01), frameon=False, borderaxespad=0)
    fig.suptitle('Fixed-checkpoint policy comparison', x=.43, y=.98, weight='bold')
    fig.subplots_adjust(left=0.055, right=0.985, top=0.84, bottom=0.17)
    save(fig, out, 'fixed_checkpoint_policy_comparison')

def fig2(main, rows, paired, out):
    methods = [MAIN[2], MAIN[3], MAIN[4]]
    fig = plt.figure(figsize=(15.8, 4.9))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.92], wspace=0.42)
    specs = [
        ('survival_rate', 'Exact survival / return', (0.755, .800), False, '{:.3f}'),
        ('inadmissibility_rate', 'Selected-action inadmissibility (%)', (0, 26), True, '{:.2f}'),
        ('expert_argmax_match', 'Expert argmax match', (0, 1.03), False, '{:.3f}'),
        ('mean_kl_to_expert', 'KL to expert', (0, 17.5), False, '{:.2f}')
    ]
    for i, (met, yl, ylim, pct, fmt) in enumerate(specs):
        ax = fig.add_subplot(gs[0, i])
        bars_with_seeds(ax, main, rows, methods, met, yl, ylim, pct, fmt)
        panel(ax, chr(65 + i))

    ax = fig.add_subplot(gs[0, 4])
    ax.axis('off')
    ax.text(0, 1, 'Paired difference', weight='bold', fontsize=12, va='top')
    d = paired['metrics']
    txt = (
        f"LAADAN-AC − post-hoc mask\n\n"
        f"Return     {d['survival_rate']['mean']:+.4f} ± {d['survival_rate']['ci95_half']:.4f}\n"
        f"Expert     {d['expert_argmax_match']['mean']:+.4f} ± {d['expert_argmax_match']['ci95_half']:.4f}\n"
        f"KL         {d['mean_kl_to_expert']['mean']:+.4f} ± {d['mean_kl_to_expert']['ci95_half']:.4f}\n"
        f"Inadmiss.  {d['inadmissibility_rate']['mean']:+.4f}\n\n"
        f"n = 5 paired seeds"
    )
    ax.text(0, .90, txt, va='top', family='monospace', fontsize=9.6, linespacing=1.65)
    fig.suptitle('Training-time admissibility and post-hoc masking', x=.42, y=.98, weight='bold')
    fig.subplots_adjust(left=0.055, right=0.985, top=0.84, bottom=0.17)
    save(fig, out, 'training_time_vs_posthoc_masking')

def fig3(rows, paired, out):
    mets = [
        ('survival_rate', 'Exact survival / return', (0.762, 0.796)),
        ('expert_argmax_match', 'Expert argmax match', (0.69, .965)),
        ('mean_kl_to_expert', 'KL to expert', (3.85, 5.15)),
        ('mean_action_deviation_from_expert', 'Mean action deviation', (1.25, 2.85)),
    ]
    seed_colors = {42: '#4E79A7', 43: '#F28E2B', 44: '#59A14F', 45: '#E15759', 46: '#B07AA1'}
    fig = plt.figure(figsize=(15.6, 4.9))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, .78], wspace=.42)

    for i, (met, yl, ylim) in enumerate(mets):
        ax = fig.add_subplot(gs[0, i])
        a = per_seed(rows, 'Post-hoc Masked VOAC', met)
        b = per_seed(rows, 'LAADAN-AC', met)
        for s in SEEDS:
            ax.plot([0, 1], [a[s], b[s]], color=seed_colors[s], lw=1.8, alpha=.95)
            ax.scatter([0, 1], [a[s], b[s]], s=34, color=seed_colors[s], edgecolor='white', linewidth=.45, zorder=3)
        ax.set_xticks([0, 1], ['Post-hoc\nmasked VOAC', 'LAADAN-AC'])
        ax.set_ylabel(yl, labelpad=10)
        ax.set_ylim(*ylim)
        clean(ax)
        panel(ax, chr(65 + i))
        if met in paired['metrics']:
            p = paired['metrics'][met]
            ax.set_title(f"Δ {p['mean']:+.3f} ± {p['ci95_half']:.3f}", fontsize=9.8, pad=10)

    ax = fig.add_subplot(gs[0, 4])
    ax.axis('off')
    ax.text(0, 1, 'Seed', weight='bold', fontsize=12, va='top')
    handles = [Line2D([0], [0], marker='o', color=seed_colors[s], lw=1.8, label=str(s), markersize=6) for s in SEEDS]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(-.05, .87), frameon=False, labelspacing=1.0)
    ax.text(0, .36, 'Each line links the\nsame trained-seed pair.\nNo seed selection.', va='top', fontsize=9.6, linespacing=1.45)
    fig.suptitle('Paired seed-level comparison', x=.42, y=.98, weight='bold')
    fig.subplots_adjust(left=0.055, right=0.985, top=0.84, bottom=0.17)
    save(fig, out, 'paired_seed_comparison')

def ablation_dataset(main_rows, abl_rows):
    out = {}
    for label, source, method, key in [
        ('Full', 'main', 'LAADAN-AC', 'LAADAN'),
        ('Mask only', 'abl', 'Masking-only actor-critic', 'MASK'),
        ('No conservative', 'abl', 'LAADAN without conservative critic', 'NOCQL')
    ]:
        rows = main_rows if source == 'main' else abl_rows
        out[label] = {'rows': [r for r in rows if r['method'] == method], 'key': key}
    return out

def fig4(main, main_rows, abl_summary, abl_rows, out):
    data = ablation_dataset(main_rows, abl_rows)
    fig = plt.figure(figsize=(15.8, 4.9))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, .92], wspace=.42)
    specs = [
        ('survival_rate', 'Exact survival / return', (0.788, .810), '{:.3f}'),
        ('inadmissibility_rate', 'Selected-action inadmissibility', (0, .012), '{:.3f}'),
        ('expert_argmax_match', 'Expert argmax match', (.70, .98), '{:.3f}'),
        ('mean_kl_to_expert', 'KL to expert', (3.8, 4.8), '{:.2f}')
    ]
    labels = list(data.keys())
    xs = np.arange(3)

    for i, (met, yl, ylim, fmt) in enumerate(specs):
        ax = fig.add_subplot(gs[0, i])
        means, cis, cols, tops = [], [], [], []
        for lab in labels:
            vals = [val(r[met]) for r in data[lab]['rows'] if r.get(met, '') != '']
            m, ci = t_ci(vals)
            means.append(m)
            cis.append(ci)
            cols.append(C[data[lab]['key']])
            tops.append(max(vals) if vals else m + ci)

        bars = ax.bar(xs, means, width=.62, color=cols, edgecolor=C['INK'], linewidth=.7, yerr=cis, capsize=3, error_kw={'elinewidth': 1.0, 'capthick': 1.0, 'ecolor': C['INK']})
        for j, lab in enumerate(labels):
            vals = [val(r[met]) for r in data[lab]['rows'] if r.get(met, '') != '']
            ax.scatter(np.full(len(vals), j) + jitter_positions(len(vals)), vals, s=26, facecolor='white', edgecolor='#555555', linewidth=.9, zorder=4)

        ax.set_xticks(xs, ['Full', 'Mask only', 'No\nconservative'])
        ax.set_ylabel(yl, labelpad=10)
        ax.set_ylim(*ylim)
        ax.margins(x=0.05)
        clean(ax)
        panel(ax, chr(65 + i))

        yr = ax.get_ylim()[1] - ax.get_ylim()[0]
        for b, m, ci, top in zip(bars, means, cis, tops):
            label_y = max(top, m + ci) + 0.016 * yr
            label_y = min(label_y, ax.get_ylim()[1] - 0.03 * yr)
            ax.text(
                b.get_x() + b.get_width() / 2,
                label_y,
                fmt.format(m),
                ha='center',
                va='bottom',
                fontsize=8.8,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.92)
            )

    ax = fig.add_subplot(gs[0, 4])
    ax.axis('off')
    ax.text(0, 1, 'Mechanism summary', weight='bold', fontsize=12, va='top')
    ax.text(0, .88,
            'All three variants:\n'
            '0 selected-action\n'
            'inadmissibility across\n'
            'all five seeds.\n\n'
            'Full LAADAN-AC selects\n'
            'a markedly more expert-\n'
            'aligned operating point.',
            va='top', fontsize=9.6, linespacing=1.5)
    fig.suptitle('Ablation of admissibility and conservative policy shaping', x=.42, y=.98, weight='bold')
    fig.subplots_adjust(left=0.055, right=0.985, top=0.84, bottom=0.17)
    save(fig, out, 'mechanism_ablation')

def fig5(main, rows, out):
    methods = [MAIN[2], MAIN[3], MAIN[4]]
    fig = plt.figure(figsize=(16.0, 8.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.05, 1.05, 1.05, .92], hspace=.62, wspace=.44)

    specs = [
        ('soft_survival_rate', 'Soft-policy survival / return', (0.76, .795), False, '{:.3f}'),
        ('soft_policy_entropy', 'Soft-policy entropy', (0, 3.45), False, '{:.2f}'),
        ('soft_mean_kl_to_expert', 'Soft-policy KL to expert', (0, 3.05), False, '{:.2f}'),
        ('mean_action_deviation_from_expert', 'Greedy action deviation', (0, 12.7), False, '{:.2f}'),
    ]
    positions = [(0, 0), (0, 1), (0, 2), (1, 0)]

    for i, (spec, pos) in enumerate(zip(specs, positions)):
        met, yl, ylim, pct, fmt = spec
        ax = fig.add_subplot(gs[pos])
        bars_with_seeds(ax, main, rows, methods, met, yl, ylim, pct, fmt)
        panel(ax, chr(65 + i))

    ax = fig.add_subplot(gs[1, 1])
    sm = per_seed(rows, 'Post-hoc Masked VOAC', 'state_mask_intervention_rate')
    vals = [100 * sm[s] for s in SEEDS]
    m, ci = t_ci(vals)
    ax.bar([0], [m], width=.56, color=C['POST'], edgecolor=C['INK'], linewidth=.7)
    ax.scatter(np.zeros(5) + jitter_positions(5, .09), vals, s=30, facecolor='white', edgecolor='#555555', linewidth=.9, zorder=3)
    ax.errorbar([0], [m], yerr=[ci], fmt='none', ecolor=C['INK'], capsize=4, lw=1.0, zorder=4)
    ax.set_xticks([0], ['Post-hoc\nmasked VOAC'])
    ax.set_ylabel('States where greedy action changes (%)', labelpad=10)
    ax.set_ylim(0, 50)
    clean(ax)
    panel(ax, 'E')
    ax.text(0, min(m + ci + 1.4, 48), f'{m:.1f}%', ha='center', fontsize=8.8,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.92))

    ax = fig.add_subplot(gs[1, 2])
    sm1 = per_seed(rows, 'Post-hoc Masked VOAC', 'mean_inadmissible_probability_mass_before_mask')
    sm2 = per_seed(rows, 'Post-hoc Masked VOAC', 'max_inadmissible_probability_mass_before_mask')
    groups = [[100 * sm1[s] for s in SEEDS], [100 * sm2[s] for s in SEEDS]]
    labs = ['Mean mass', 'Max mass']
    cols = [C['POST'], '#D67842']
    means = [np.mean(v) for v in groups]
    cis = [t_ci(v)[1] for v in groups]
    bars = ax.bar([0, 1], means, width=.56, color=cols, edgecolor=C['INK'], linewidth=.7, yerr=cis, capsize=3)
    for j, v in enumerate(groups):
        ax.scatter(np.full(5, j) + jitter_positions(5, .09), v, s=30, facecolor='white', edgecolor='#555555', linewidth=.9, zorder=3)
    ax.set_xticks([0, 1], labs)
    ax.set_ylabel('Inadmissible probability mass before mask (%)', labelpad=10)
    ax.set_ylim(0, 105)
    clean(ax)
    panel(ax, 'F')
    yr = ax.get_ylim()[1] - ax.get_ylim()[0]
    for b, mval, ci in zip(bars, means, cis):
        y = min(mval + ci + 0.02 * yr, 101)
        ax.text(b.get_x() + b.get_width()/2, y, f'{mval:.1f}%', ha='center', va='bottom',
                fontsize=8.8, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.92))

    ax = fig.add_subplot(gs[:, 3])
    ax.axis('off')
    ax.text(0, 1, 'Policy diagnostics', weight='bold', fontsize=12, va='top')
    text = (
        'Soft-policy quantities use the\n'
        'probability distribution rather\n'
        'than greedy action extraction.\n\n'
        'Post-hoc diagnostics measure\n'
        'how often the mask changes the\n'
        'unrestricted VOAC decision and\n'
        'how much pre-mask probability\n'
        'mass lies on inadmissible actions.\n\n'
        'Points = individual seeds\n'
        'Bars = seed mean\n'
        'Error bars = 95% t-CI'
    )
    ax.text(0, .91, text, va='top', fontsize=9.6, linespacing=1.5)

    fig.suptitle('Policy-distribution and masking diagnostics', x=.42, y=.98, weight='bold')
    fig.subplots_adjust(left=0.06, right=0.985, top=0.88, bottom=0.10)
    save(fig, out, 'policy_distribution_diagnostics')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-root', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, required=True)
    a = ap.parse_args()

    root = a.results_root
    out = a.outdir
    out.mkdir(parents=True, exist_ok=True)

    agg = root / 'aggregate'
    required = [
        agg / 'main_summary.json',
        agg / 'per_seed_metrics.csv',
        agg / 'paired_laadan_minus_posthoc_voac.json',
        agg / 'ablation_summary.json',
        agg / 'ablation_per_seed_metrics.csv',
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError('Missing required files:\n' + '\n'.join(missing))

    main_summary = load_json(required[0])
    rows = load_csv(required[1])
    paired = load_json(required[2])
    abl_summary = load_json(required[3])
    abl_rows = load_csv(required[4])

    for method, _, _ in MAIN:
        seeds = sorted(int(r['seed']) for r in rows if r['method'] == method)
        if seeds != SEEDS:
            raise ValueError(f'{method}: expected seeds {SEEDS}, got {seeds}')

    for method in ['Masking-only actor-critic', 'LAADAN without conservative critic']:
        seeds = sorted(int(r['seed']) for r in abl_rows if r['method'] == method)
        if seeds != SEEDS:
            raise ValueError(f'{method}: expected seeds {SEEDS}, got {seeds}')

    fig1(main_summary, rows, out)
    fig2(main_summary, rows, paired, out)
    fig3(rows, paired, out)
    fig4(main_summary, rows, abl_summary, abl_rows, out)
    fig5(main_summary, rows, out)

    print('\nPASS: generated five publication figures from frozen result files only.')
    print('Output:', out.resolve())
    for p in sorted(out.iterdir()):
        print(' -', p.name)

if __name__ == '__main__':
    main()
