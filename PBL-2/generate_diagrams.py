"""Generate all diagrams for PBL-2 submission."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)

# ─── 1. Validation AUROC Comparison Bar Chart ─────────────────────────────
def plot_auroc_comparison():
    models = ['Visual-only', 'Audio-only', 'Equal-weight\nFixed Fusion', 'Reliability-aware\nFusion']
    aurocs = [0.7974, 0.8625, 0.9426, 0.7957]
    colors = ['#5B9BD5', '#ED7D31', '#70AD47', '#FFC000']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, aurocs, color=colors, width=0.55, edgecolor='#333333', linewidth=0.8)
    
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Validation AUROC', fontsize=13)
    ax.set_title('Preliminary One-Epoch Validation AUROC Comparison\n(FakeAVCeleb v1.2, Identity-Disjoint Split, Seed 42)', fontsize=13)
    ax.set_ylim(0.7, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='Random baseline')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'auroc_comparison.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'auroc_comparison.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("✓ auroc_comparison")

# ─── 2. Data Split Distribution ────────────────────────────────────────────
def plot_data_splits():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Split sizes
    splits = ['Train\n(15,018)', 'Validation\n(2,136)', 'Test\n(4,390)']
    sizes = [15018, 2136, 4390]
    colors_split = ['#5B9BD5', '#ED7D31', '#70AD47']
    
    bars = ax1.bar(splits, sizes, color=colors_split, edgecolor='#333', linewidth=0.8)
    for bar, val in zip(bars, sizes):
        pct = val / 21544 * 100
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Clips', fontsize=12)
    ax1.set_title('Identity-Disjoint Split Sizes\n(Total: 21,544 clips)', fontsize=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3)
    
    # Class distribution per split  
    real = [350, 50, 100]
    fake = [14668, 2086, 4290]
    split_labels = ['Train', 'Validation', 'Test']
    x = np.arange(len(split_labels))
    w = 0.3
    
    bars_real = ax2.bar(x - w/2, real, w, label='Real', color='#70AD47', edgecolor='#333', linewidth=0.8)
    bars_fake = ax2.bar(x + w/2, fake, w, label='Fake', color='#C00000', edgecolor='#333', linewidth=0.8)
    
    for bar, val in zip(bars_real, real):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
                str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars_fake, fake):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
                f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(split_labels, fontsize=11)
    ax2.set_ylabel('Number of Clips', fontsize=12)
    ax2.set_title('Class Distribution per Split\n(Highly Imbalanced: ~97.7% Fake)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'data_splits_detailed.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'data_splits_detailed.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("✓ data_splits_detailed")

# ─── 3. Training Loss Comparison ──────────────────────────────────────────
def plot_training_losses():
    import json
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'claude', 'data_and_reports')
    
    models = {
        'Visual-only': ('unimodal_visual_history.json', '#5B9BD5'),
        'Audio-only': ('unimodal_audio_history.json', '#ED7D31'),
        'Fixed Fusion': ('fixed_fusion_history.json', '#70AD47'),
        'Reliability Fusion': ('reliability_fusion_history.json', '#FFC000')
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    
    train_losses = []
    val_losses = []
    labels = []
    colors = []
    
    for name, (fname, color) in models.items():
        path = os.path.join(data_dir, fname)
        with open(path) as f:
            d = json.load(f)
        train_losses.append(d['train_loss'][0])
        val_losses.append(d['val_loss'][0])
        labels.append(name)
        colors.append(color)
    
    x = np.arange(len(labels))
    w = 0.35
    
    bars1 = ax1.bar(x - w/2, train_losses, w, label='Train Loss', color=[c for c in colors], alpha=0.7, edgecolor='#333', linewidth=0.8)
    bars2 = ax1.bar(x + w/2, val_losses, w, label='Val Loss', color=[c for c in colors], edgecolor='#333', linewidth=0.8)
    
    for bar, val in zip(bars1, train_losses):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, rotation=45)
    for bar, val in zip(bars2, val_losses):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, rotation=45)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=9)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training vs Validation Loss\n(One Epoch)', fontsize=12)
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.3)
    
    # Loss component breakdown for reliability fusion
    rel_path = os.path.join(data_dir, 'reliability_fusion_history.json')
    with open(rel_path) as f:
        rel = json.load(f)
    
    components = ['L_f (Fused)', 'L_unimodal', 'L_rel', 'L_con', 'L_bal']
    values = [rel['train_L_f'][0], rel['train_L_unimodal'][0], rel['train_L_rel'][0], 
              rel['train_L_con'][0], rel['train_L_bal'][0]]
    comp_colors = ['#5B9BD5', '#ED7D31', '#FFC000', '#70AD47', '#A5A5A5']
    
    bars = ax2.barh(components, values, color=comp_colors, edgecolor='#333', linewidth=0.8)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Loss Value', fontsize=12)
    ax2.set_title('Reliability Fusion Loss Components\n(One Epoch)', fontsize=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'training_losses.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'training_losses.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("✓ training_losses")

# ─── 4. Technology Stack Diagram ──────────────────────────────────────────
def plot_technology_stack():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Layer definitions (bottom to top)
    layers = [
        (0.5, 0.3, 13, 1.1, '#E8EAF6', 'Data & Infrastructure',
         'FFmpeg/ffprobe · OpenCV · torchaudio · NumPy/pandas\nIdentity-disjoint CSV manifests · YAML configs · pytest'),
        (0.5, 1.7, 13, 1.1, '#E3F2FD', 'Preprocessing Pipeline',
         'Face detection (Haar) · 160×160 crops · 80-bin log-Mel\nQuality features (blur, SNR, VAD) · Temporal alignment (T=8 bins)'),
        (0.5, 3.1, 6, 1.1, '#E8F5E9', 'Visual Branch',
         'MobileNetV3-Small\n+ temporal Conv1D\nz_v ∈ ℝ^{8×256}'),
        (7.5, 3.1, 6, 1.1, '#FFF3E0', 'Audio Branch',
         'Depthwise-sep. CNN\n80-bin log-Mel input\nz_a ∈ ℝ^{8×256}'),
        (0.5, 4.5, 13, 1.1, '#FCE4EC', 'Fusion & Reliability Layer',
         'Sync features (cosine cross-corr, δ∈{-2..+2}) · Modality heads (p_v, p_a)\nReliability MLP (532→128→2) · Softmax gate (w_v, w_a) · Weighted fusion classifier'),
        (0.5, 5.9, 13, 1.1, '#F3E5F5', 'Training & Optimization',
         'PyTorch 2.x · AdamW · AMP (float16) · Gradient accumulation (eff. batch 32)\nMulti-task loss: L_f + α(L_v+L_a) + βL_rel + γL_con + ηL_bal'),
        (0.5, 7.3, 13, 0.5, '#EFEBE9', 'Evaluation & Reporting',
         'scikit-learn · SciPy · Bootstrap CIs · Calibration (ECE, Brier) · Latency profiling'),
    ]
    
    for x, y, w, h, color, title, desc in layers:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor='#555', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.3, y + h - 0.15, title, fontsize=11, fontweight='bold', va='top')
        ax.text(x + w/2, y + h/2 - 0.1, desc, fontsize=8.5, ha='center', va='center',
                style='italic', color='#333')
    
    # Arrows between layers
    for y_start, y_end in [(1.4, 1.7), (2.8, 3.1), (4.2, 4.5), (5.6, 5.9), (7.0, 7.3)]:
        ax.annotate('', xy=(7, y_end), xytext=(7, y_start),
                   arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))
    
    ax.set_title('Technology Stack: Reliability-Aware AV Deepfake Detector', 
                 fontsize=14, fontweight='bold', pad=10)
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'technology_stack.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'technology_stack.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("✓ technology_stack")

# ─── 5. Architecture Flow Diagram ─────────────────────────────────────────
def plot_architecture_flow():
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    box_style = "round,pad=0.15"
    
    # Input
    input_v = FancyBboxPatch((0.3, 4.5), 2.0, 1.5, boxstyle=box_style, facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5)
    input_a = FancyBboxPatch((0.3, 1.0), 2.0, 1.5, boxstyle=box_style, facecolor='#FFF3E0', edgecolor='#E65100', linewidth=1.5)
    ax.add_patch(input_v); ax.add_patch(input_a)
    ax.text(1.3, 5.25, '2-s Video\n16 face frames\n160×160 RGB', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(1.3, 1.75, '2-s Audio\n16 kHz mono\n80-bin log-Mel', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Encoders
    enc_v = FancyBboxPatch((3.0, 4.5), 2.2, 1.5, boxstyle=box_style, facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=1.5)
    enc_a = FancyBboxPatch((3.0, 1.0), 2.2, 1.5, boxstyle=box_style, facecolor='#FFF8E1', edgecolor='#F57F17', linewidth=1.5)
    ax.add_patch(enc_v); ax.add_patch(enc_a)
    ax.text(4.1, 5.25, 'Visual Encoder\nMobileNetV3-S\n+ Temporal Conv', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(4.1, 1.75, 'Audio Encoder\nDS-CNN\n(width=1.0)', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Quality
    qual = FancyBboxPatch((3.0, 3.1), 2.2, 1.0, boxstyle=box_style, facecolor='#F1F8E9', edgecolor='#558B2F', linewidth=1.2)
    ax.add_patch(qual)
    ax.text(4.1, 3.6, 'Quality Features\nq_v (7) + q_a (5)', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Heads
    head_v = FancyBboxPatch((5.8, 4.8), 1.8, 1.0, boxstyle=box_style, facecolor='#E8EAF6', edgecolor='#283593', linewidth=1.2)
    head_a = FancyBboxPatch((5.8, 1.2), 1.8, 1.0, boxstyle=box_style, facecolor='#FBE9E7', edgecolor='#BF360C', linewidth=1.2)
    ax.add_patch(head_v); ax.add_patch(head_a)
    ax.text(6.7, 5.3, 'Visual Head\np_v, u_v', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(6.7, 1.7, 'Audio Head\np_a, u_a', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Sync
    sync = FancyBboxPatch((5.8, 3.0), 1.8, 1.2, boxstyle=box_style, facecolor='#E0F7FA', edgecolor='#006064', linewidth=1.2)
    ax.add_patch(sync)
    ax.text(6.7, 3.6, 'Sync Features\nc ∈ ℝ⁴\nδ ∈ {-2..+2}', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Reliability MLP
    gate = FancyBboxPatch((8.3, 2.8), 2.2, 1.6, boxstyle=box_style, facecolor='#FCE4EC', edgecolor='#880E4F', linewidth=2)
    ax.add_patch(gate)
    ax.text(9.4, 3.6, 'Reliability\nEstimator\nMLP (532→128→2)', ha='center', va='center', fontsize=8, fontweight='bold', color='#880E4F')
    
    # Softmax gate
    sw = FancyBboxPatch((11.0, 3.0), 1.5, 1.2, boxstyle=box_style, facecolor='#FFF9C4', edgecolor='#F57F17', linewidth=1.5)
    ax.add_patch(sw)
    ax.text(11.75, 3.6, 'Softmax\nGate\nw_v + w_a = 1', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Fusion classifier
    fus = FancyBboxPatch((13.0, 2.8), 1.8, 1.6, boxstyle=box_style, facecolor='#E8F5E9', edgecolor='#1B5E20', linewidth=2)
    ax.add_patch(fus)
    ax.text(13.9, 3.6, 'Fusion\nClassifier\np_f', ha='center', va='center', fontsize=9, fontweight='bold', color='#1B5E20')
    
    # Arrows
    arrow_style = dict(arrowstyle='->', color='#555', lw=1.5, connectionstyle='arc3,rad=0')
    for (x1, y1, x2, y2) in [
        (2.3, 5.25, 3.0, 5.25),   # video -> encoder_v
        (2.3, 1.75, 3.0, 1.75),   # audio -> encoder_a
        (5.2, 5.25, 5.8, 5.25),   # encoder_v -> head_v
        (5.2, 1.75, 5.8, 1.75),   # encoder_a -> head_a
        (5.2, 3.6, 5.8, 3.6),     # quality -> sync
        (7.6, 5.0, 8.3, 4.0),     # head_v -> gate
        (7.6, 1.9, 8.3, 3.2),     # head_a -> gate
        (7.6, 3.6, 8.3, 3.6),     # sync -> gate
        (10.5, 3.6, 11.0, 3.6),   # gate -> softmax
        (12.5, 3.6, 13.0, 3.6),   # softmax -> fusion
    ]:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_style)
    
    ax.set_title('System Architecture: Reliability-Aware Audio-Visual Deepfake Detector', 
                 fontsize=13, fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'architecture_flow.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'architecture_flow.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("✓ architecture_flow")

# ─── 6. Staged Training Pipeline ──────────────────────────────────────────
def plot_staged_training():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    stages = [
        (0.2, 'Stage 1\nData Audit', '#E3F2FD', '• Decode & validate\n• Group split\n• 200-clip inspection'),
        (2.5, 'Stage 2\nUnimodal', '#E8F5E9', '• Visual baseline\n• Audio baseline\n• Error profiles'),
        (4.8, 'Stage 3\nFixed Fusion', '#FFF3E0', '• Equal-weight\n• Concatenation\n• Fusion helps?'),
        (7.1, 'Stage 4\nReliability', '#FCE4EC', '• Gate + quality\n• L_rel loss\n• Staged unfreeze'),
        (9.4, 'Stage 5\nRobustness', '#E0F7FA', '• Degradation suite\n• Missing modality\n• Cross-manip.'),
        (11.7, 'Stage 6\nCalibration', '#F3E5F5', '• Temperature\n• Efficiency\n• Final test'),
    ]
    
    for x, title, color, desc in stages:
        rect = FancyBboxPatch((x, 1.0), 2.0, 3.2, boxstyle="round,pad=0.15",
                               facecolor=color, edgecolor='#555', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 1.0, 3.8, title, ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(x + 1.0, 2.2, desc, ha='center', va='center', fontsize=7.5, style='italic')
    
    for i in range(5):
        x1 = stages[i][0] + 2.0
        x2 = stages[i+1][0]
        ax.annotate('', xy=(x2, 2.6), xytext=(x1, 2.6),
                   arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    
    ax.set_title('Staged Training Pipeline', fontsize=14, fontweight='bold', pad=15)
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'staged_training.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'staged_training.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("✓ staged_training")

# ─── 7. Evaluation Protocol Diagram ───────────────────────────────────────
def plot_eval_protocol():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Data source
    r = FancyBboxPatch((3.5, 6.0), 5, 0.7, boxstyle="round,pad=0.1", facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=1.5)
    ax.add_patch(r)
    ax.text(6, 6.35, 'FakeAVCeleb v1.2 (21,544 clips, identity-grouped)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Splits
    for x, label, count, color in [(1, 'Train\n15,018', 15018, '#E8F5E9'), (4.5, 'Val\n2,136', 2136, '#FFF3E0'), (8, 'Test\n4,390', 4390, '#FCE4EC')]:
        r = FancyBboxPatch((x, 4.5), 2.5, 1.0, boxstyle="round,pad=0.1", facecolor=color, edgecolor='#555', linewidth=1.2)
        ax.add_patch(r)
        ax.text(x+1.25, 5.0, label, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows from source to splits
    for x_target in [2.25, 5.75, 9.25]:
        ax.annotate('', xy=(x_target, 5.5), xytext=(6, 6.0),
                   arrowprops=dict(arrowstyle='->', color='#666', lw=1.2))
    
    # Evaluation types
    tests = [
        (0.2, 2.5, 'In-domain\nTest', '#C8E6C9'),
        (2.5, 2.5, 'Cross-\nManipulation', '#FFECB3'),
        (4.8, 2.5, 'Degradation\nSuite', '#FFCDD2'),
        (7.1, 2.5, 'Missing\nModality', '#D1C4E9'),
        (9.4, 2.5, 'Calibration\n& Efficiency', '#B2EBF2'),
    ]
    for x, y, label, color in tests:
        r = FancyBboxPatch((x, y), 2.0, 1.3, boxstyle="round,pad=0.1", facecolor=color, edgecolor='#555', linewidth=1.2)
        ax.add_patch(r)
        ax.text(x+1.0, y+0.65, label, ha='center', va='center', fontsize=8.5, fontweight='bold')
    
    # Metrics
    r = FancyBboxPatch((1.5, 0.5), 9, 1.2, boxstyle="round,pad=0.1", facecolor='#EFEBE9', edgecolor='#555', linewidth=1.5)
    ax.add_patch(r)
    ax.text(6, 1.1, 'Metrics: AUROC · PR-AUC · F1 · ECE · Brier · Bootstrap 95% CI\nGate diagnostics: w_v, w_a distributions · Correlation with degradation · Collapse detection', 
            ha='center', va='center', fontsize=9, style='italic')
    
    ax.set_title('Evaluation Protocol', fontsize=14, fontweight='bold', pad=10)
    
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'eval_protocol.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'eval_protocol.pdf'), bbox_inches='tight')
    plt.close(fig)
    print("✓ eval_protocol")

# ─── Run all ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    plot_auroc_comparison()
    plot_data_splits()
    plot_training_losses()
    plot_technology_stack()
    plot_architecture_flow()
    plot_staged_training()
    plot_eval_protocol()
    print("\nAll PBL-2 diagrams generated successfully!")
