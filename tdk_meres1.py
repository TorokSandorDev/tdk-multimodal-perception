import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("lane_metrics.csv")
df_active = df[(df["pred_grid_cells"] > 0) | (df["raw_grid_cells"] > 0)].copy()

valid_threshold = 30
df_active["raw_valid_real"] = (df_active["raw_max_span_px"] >= valid_threshold).astype(int)
df_active["pred_valid_real"] = (df_active["pred_max_span_px"] >= valid_threshold).astype(int)

raw_avail = df_active["raw_valid_real"].mean() * 100.0
pred_avail = df_active["pred_valid_real"].mean() * 100.0

raw_span_mean = df_active["raw_max_span_px"].mean()
pred_span_mean = df_active["pred_max_span_px"].mean()

raw_cells_mean = df_active["raw_grid_cells"].mean()
pred_cells_mean = df_active["pred_grid_cells"].mean()

total_bridged_gaps = df_active["bridged_gaps"].sum()

print("=== ÚJRAOSZTOTT 3 PANELES ÁBRA GENERÁLÁSA ===")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4.8), dpi=300)

labels_binary = ['Nyers', 'Prediktív']
colors = ['#7f8c8d', '#2980b9']

bars1 = ax1.bar(labels_binary, [raw_avail, pred_avail], color=colors, width=0.5, edgecolor='black')
ax1.set_ylim(0, max(max([raw_avail, pred_avail]) * 1.2, 100))
ax1.set_ylabel('Rendelkezésre állás [%]', fontsize=10, fontweight='bold')
ax1.set_title('Sávfolytonosság', fontsize=11, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars1:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, y + max(max([raw_avail, pred_avail])*0.02, 1), f'{y:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

bars2 = ax2.bar(labels_binary, [raw_span_mean, pred_span_mean], color=colors, width=0.5, edgecolor='black')
ax2.set_ylim(0, max([raw_span_mean, pred_span_mean]) * 1.3)
ax2.set_ylabel('Ívhossz [képsor / px]', fontsize=10, fontweight='bold')
ax2.set_title('Összefüggő ívhossz', fontsize=11, fontweight='bold')
ax2.grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars2:
    y = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, y + max([raw_span_mean, pred_span_mean])*0.02, f'{y:.1f} px', ha='center', va='bottom', fontweight='bold', fontsize=9)

bars3 = ax3.bar(labels_binary, [raw_cells_mean, pred_cells_mean], color=colors, width=0.5, edgecolor='black')

min_cells = min([raw_cells_mean, pred_cells_mean])
max_cells = max([raw_cells_mean, pred_cells_mean])
y_min = max(0, min_cells * 0.9)
y_max = max_cells * 1.1

ax3.set_ylim(y_min, y_max)
ax3.set_ylabel('Aktív cellák száma [db]', fontsize=10, fontweight='bold')
ax3.set_title('Rácstérkép telítettség', fontsize=11, fontweight='bold')
ax3.grid(axis='y', linestyle='--', alpha=0.5)
for bar in bars3:
    y = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0, y + (y_max-y_min)*0.02, f'{y:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

fig.suptitle(f'Összesen áthidalt sávszakadások (vakfoltok) a mérés során: {total_bridged_gaps} db', 
             fontsize=11, fontweight='bold', color='#2c3e50', y=0.03)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("savkovetes_robosztussag.png")
print("savkovetes_robosztussag.png")
