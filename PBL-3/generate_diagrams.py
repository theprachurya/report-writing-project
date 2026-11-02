"""Generate all diagrams for PBL-3 submission."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)

# ─── 1. Comprehensive AUROC Comparison (all models + deltas) ──────────────
def plot_auroc_with_deltas():
    models = ['Visual-only', 'Audio-only', 'Equal-weight\nFixed Fusion', 'Reliability-aware\nFusion']
    aurocs = [0.7974, 0.8625, 0.9426, 0.7957]
    colors = ['#5B9BD5', '#ED7D31', '#70AD47', '#FFC000']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.2, 1]})
    
    bars = ax1.bar(models, aurocs, color=colors, width=0.55, edgecolor='#333', linewidth=0.8)
    for bar, val in zip(bars, aurocs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Validation AUROC', fontsize=13)
    ax1.set_title('One-Epoch Validation AUROC\n(FakeAVCeleb v1.2, Seed 42)', fontsize=12)
    ax1.set_ylim(0.7, 1.0)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3)
    
    # Delta matrix
    delta_data = [
        [0, -0.0651, -0.1452, +0.0017],
        [+0.0651, 0, -0.0801, +0.0668],
        [+0.1452, +0.0801, 0, +0.1469],
        [-0.0017, -0.0668, -0.1469, 0]
    ]
    short_names = ['Visual', 'Audio', 'Fixed', 'Reliab.']
    
    im = ax2.imshow(delta_data, cmap='RdYlGn', aspect='auto', vmin=-0.15, vmax=0.15)
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    ax2.set_xticklabels(short_names, fontsize=10)
    ax2.set_yticklabels(short_names, fontsize=10)
    ax2.set_title('Pairwise AUROC Differences\n(Row − Column)', fontsize=12)
    
    for i in range(4):
        for j in range(4):
            val = delta_data[i][j]
            color = 'white' if abs(val) > 0.1 else 'black'
            ax2.text(j, i, f'{val:+.4f}', ha='center', va='center', fontsize=9,
                    fontweight='bold', color=color)
    
    fig.colorbar(im, ax=ax2, shrink=0.8, label='AUROC Difference')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'auroc_with_deltas.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'auroc_with_deltas.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("✓ auroc_with_deltas")

# ─── 2. MLAAD Transfer Per-Generator Breakdown ───────────────────────────
def plot_mlaad_per_generator():
    generators = ['Chatterbox\nMultilingual', 'Kartoffelbox', 'Higgs-Audio\nV2', 'OuteTTS', 
                   'griffin_lim', 'Llasa-1B\nMultilingual', 'Resemble.ai', 'Edge-TTS',
                   'RVC', 'CaroTTS', 'Cartesia.ai\n(Sonic-3)', 'facebook/\nmms-tts-deu',
                   'FishTTS', 'NVidia\nMagpie-TTS', 'Kani-TTS\n370M']
    aurocs = [0.7238, 0.7208, 0.6969, 0.5414, 0.5180, 0.4879, 0.4129, 0.3960,
              0.2696, 0.2094, 0.1992, 0.1548, 0.1460, 0.0963, 0.0920]
    
    colors = ['#2E7D32' if a > 0.6 else '#F57F17' if a > 0.4 else '#C62828' for a in aurocs]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.barh(range(len(generators)), aurocs, color=colors, edgecolor='#333', linewidth=0.7)
    ax.set_yticks(range(len(generators)))
    ax.set_yticklabels(generators, fontsize=8.5)
    ax.set_xlabel('AUROC on German MLAAD-tiny Slice', fontsize=12)
    ax.set_title('Audio-Only Cross-Dataset Transfer: Per-Generator AUROC\n(FakeAVCeleb audio branch → MLAAD-tiny German, threshold not retuned)', fontsize=11)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline (0.5)')
    ax.axvline(x=0.3777, color='blue', linestyle='--', alpha=0.5, label='Overall AUROC (0.3777)')
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim(0, 0.85)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'mlaad_per_generator.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'mlaad_per_generator.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("✓ mlaad_per_generator")

# ─── 3. Loss Component Breakdown for All Models ──────────────────────────
def plot_loss_components():
    import json
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'claude', 'data_and_reports')
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    
    configs = [
        ('unimodal_visual_history.json', 'Visual-Only', axes[0, 0], '#5B9BD5'),
        ('unimodal_audio_history.json', 'Audio-Only', axes[0, 1], '#ED7D31'),
        ('fixed_fusion_history.json', 'Fixed Fusion', axes[1, 0], '#70AD47'),
        ('reliability_fusion_history.json', 'Reliability Fusion', axes[1, 1], '#FFC000'),
    ]
    
    for fname, title, ax, main_color in configs:
        with open(os.path.join(data_dir, fname)) as f:
            d = json.load(f)
        
        components = {
            'L_f (Fused BCE)': d['train_L_f'][0],
            'L_unimodal': d['train_L_unimodal'][0],
            'L_rel': d['train_L_rel'][0],
            'L_con': d['train_L_con'][0],
            'L_bal': d['train_L_bal'][0],
        }
        
        nonzero = {k: v for k, v in components.items() if v > 0.001}
        if not nonzero:
            nonzero = {'L_f (Fused BCE)': d['train_L_f'][0]}
        
        comp_colors = ['#5B9BD5', '#ED7D31', '#FFC000', '#70AD47', '#A5A5A5']
        
        bars = ax.bar(range(len(nonzero)), list(nonzero.values()), 
                      color=comp_colors[:len(nonzero)], edgecolor='#333', linewidth=0.8)
        ax.set_xticks(range(len(nonzero)))
        ax.set_xticklabels(list(nonzero.keys()), fontsize=8, rotation=20, ha='right')
        
        for bar, val in zip(bars, nonzero.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel('Loss Value', fontsize=10)
        ax.set_title(f'{title}\n(Train Loss: {d["train_loss"][0]:.4f}, Val Loss: {d["val_loss"][0]:.4f})', 
                     fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Training Loss Component Breakdown (One Epoch)', fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'loss_components.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'loss_components.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("✓ loss_components")

# ─── 4. Hypothesis Status Dashboard ──────────────────────────────────────
def plot_hypothesis_status():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    hypotheses = [
        ('H1', 'Reliability fusion > fixed fusion\non degradation suite', 'NOT EVALUATED', '#FFCDD2'),
        ('H2', 'Gate shifts under degradation;\nreduces performance drop', 'NOT EVALUATED', '#FFCDD2'),
        ('H3', 'Matches/exceeds best unimodal\non unseen manipulations', 'NOT EVALUATED', '#FFCDD2'),
        ('H4', 'Practical latency with\ndocumented accuracy cost', 'NOT EVALUATED', '#FFCDD2'),
    ]
    
    for i, (hid, desc, status, color) in enumerate(hypotheses):
        x = i * 3 + 0.2
        rect = FancyBboxPatch((x, 0.5), 2.6, 3.8, boxstyle="round,pad=0.15",
                               facecolor=color, edgecolor='#C62828', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 1.3, 3.8, hid, ha='center', va='center', fontsize=14, fontweight='bold', color='#C62828')
        ax.text(x + 1.3, 2.6, desc, ha='center', va='center', fontsize=8.5, style='italic')
        ax.text(x + 1.3, 1.2, status, ha='center', va='center', fontsize=9, fontweight='bold',
                color='#C62828',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='#C62828', alpha=0.8))
    
    ax.set_title('Hypothesis Evaluation Status — All Pending Final Experiments', 
                 fontsize=13, fontweight='bold', pad=15)
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'hypothesis_status.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'hypothesis_status.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("✓ hypothesis_status")

# ─── 5. Completion Timeline (Gantt-style) ─────────────────────────────────
def plot_completion_timeline():
    fig, ax = plt.subplots(figsize=(14, 5))
    
    tasks = [
        ('Full unimodal + fixed fusion training', '27-30 Oct', 0, 4, '#5B9BD5'),
        ('Reliability fusion + gate diagnostics', '31 Oct-2 Nov', 4, 3, '#FFC000'),
        ('Degradation + cross-manipulation eval', '3-5 Nov', 7, 3, '#ED7D31'),
        ('Final model selection + calibration', '6-7 Nov', 10, 2, '#70AD47'),
        ('Primary test (once-only)', '8 Nov', 12, 1, '#C62828'),
        ('Report + viva preparation', '9 Nov', 13, 1, '#9C27B0'),
        ('VIVA ASSESSMENT', '10 Nov', 14, 1, '#000000'),
    ]
    
    for i, (task, dates, start, duration, color) in enumerate(tasks):
        ax.barh(i, duration, left=start, color=color, edgecolor='#333', linewidth=0.8, height=0.6)
        ax.text(start + duration/2, i, f'{task}\n({dates})', ha='center', va='center', 
                fontsize=8, fontweight='bold', color='white' if color in ['#C62828', '#000000'] else 'black')
    
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t[0] for t in tasks], fontsize=9)
    ax.set_xlabel('Days from 27 Oct 2026', fontsize=11)
    ax.set_title('Completion Plan: PBL-3 → Final Viva (27 Oct – 10 Nov 2026)', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)
    
    # Today marker
    ax.axvline(x=0, color='red', linestyle='-', linewidth=2, alpha=0.7, label='Today (27 Oct)')
    ax.legend(fontsize=10)
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'completion_timeline.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'completion_timeline.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("✓ completion_timeline")

# ─── 6. Confusion Matrix for MLAAD Transfer ──────────────────────────────
def plot_mlaad_confusion():
    cm = np.array([[110, 1210], [177, 1323]])
    
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nBona-fide', 'Predicted\nSpoof'], fontsize=11)
    ax.set_yticklabels(['Actual\nBona-fide', 'Actual\nSpoof'], fontsize=11)
    
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > 800 else 'black'
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center', fontsize=16,
                    fontweight='bold', color=color)
    
    ax.set_title('MLAAD-tiny German: Confusion Matrix\n(Audio-only transfer, threshold=0.295 from FakeAVCeleb val)', fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    
    # Add metrics annotation
    metrics_text = (f'AUROC: 0.3777\n95% CI: [0.3588, 0.3964]\n'
                    f'Balanced Acc: 0.4827\nSpecificity: 0.0833')
    ax.text(1.55, 0.5, metrics_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='#333', alpha=0.9))
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'mlaad_confusion.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'mlaad_confusion.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("✓ mlaad_confusion")

if __name__ == '__main__':
    plot_auroc_with_deltas()
    plot_mlaad_per_generator()
    plot_loss_components()
    plot_hypothesis_status()
    plot_completion_timeline()
    plot_mlaad_confusion()
    print("\nAll PBL-3 diagrams generated successfully!")
