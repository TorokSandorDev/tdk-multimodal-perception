import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

shutil.copyfile('/tmp/tdk_meres.csv', 'tdk_meres.csv')

df = pd.read_csv('tdk_meres.csv')
df_valid = df[df['ms_total'] > 0.1].copy().reset_index(drop=True)

p1 = df_valid[df_valid['phase'] == 0]
p2 = df_valid[df_valid['phase'] == 1]

if len(p1) == 0:
    p1 = df_valid.iloc[:len(df_valid)//2]
if len(p2) == 0:
    p2 = df_valid.iloc[len(df_valid)//2:]

df_obs = df_valid[df_valid['raw_obs_cells'] > 0]
total_filtered_cells = int((df_obs['raw_obs_cells'] - df_obs['moore_obs_cells']).sum())

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.8), gridspec_kw={'width_ratios': [1.35, 1.05, 0.9]})

cycles = np.arange(len(df_valid))
y_total = df_valid['ms_total'].to_numpy()
y_post = df_valid['ms_post'].to_numpy()
phases_arr = df_valid['phase'].to_numpy()

ax1.plot(cycles, y_total, color='#1f77b4', linewidth=1.4, label='Teljes ciklusidő ($T_{total}$)')
ax1.plot(cycles, y_post, color='#ff7f0e', linewidth=1.3, alpha=0.9, label='Potenciálmező szintézis ($T_{post}$)')

p2_indices = np.where(phases_arr == 1)[0]
if len(p2_indices) > 0:
    switch_idx = p2_indices[0]
    ax1.axvline(switch_idx, color='#d62728', linestyle='--', linewidth=1.8, label='Fázisváltás ($D_{th}$)')
    ax1.axvspan(0, switch_idx, color='#e5f5e0', alpha=0.3)
    ax1.axvspan(switch_idx, len(df_valid), color='#fee0d2', alpha=0.3)

ax1.set_title('Ciklusidő alakulása a fázisváltás során', fontsize=12, fontweight='bold')
ax1.set_xlabel('Fúziós ciklusok [db]', fontsize=11, fontweight='bold')
ax1.set_ylabel('Futásidő [ms]', fontsize=11, fontweight='bold')
ax1.set_ylim(0, max(np.percentile(y_total, 99) * 1.35, 5.0))
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.legend(loc='upper left', fontsize=9, framealpha=0.95)

phases_labels = ['Phase 1\n(Sávkövetés)', 'Phase 2\n(Szlalom)']
t_temp = np.array([p1['ms_temporal'].mean(), p2['ms_temporal'].mean()])
t_proj = np.array([p1['ms_proj'].mean(), p2['ms_proj'].mean()])
t_post = np.array([p1['ms_post'].mean(), p2['ms_post'].mean()])
t_rest = np.array([p1['ms_rest'].mean(), p2['ms_rest'].mean()])

x = np.arange(len(phases_labels))
w = 0.46

ax2.bar(x, t_temp, w, label='Log-Odds + Kamera', color='#7f8c8d', edgecolor='black')
ax2.bar(x, t_proj, w, bottom=t_temp, label='LiDAR + Sávprojekció', color='#2ca02c', edgecolor='black')
ax2.bar(x, t_post, w, bottom=t_temp+t_proj, label='Potenciálmező (Gauss)', color='#1f77b4', edgecolor='black')
ax2.bar(x, t_rest, w, bottom=t_temp+t_proj+t_post, label='Visszavetítés + Pub.', color='#9467bd', edgecolor='black')

totals = t_temp + t_proj + t_post + t_rest
for i, tot in enumerate(totals):
    ax2.text(i, tot + 0.08, f"{tot:.2f} ms", ha='center', va='bottom', fontweight='bold', fontsize=10.5)

ax2.set_title('Részfolyamatok ideje fázisonként', fontsize=12, fontweight='bold')
ax2.set_ylabel('Kumulált futásidő [ms]', fontsize=11, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(phases_labels, fontsize=10.5, fontweight='bold')
ax2.set_ylim(0, max(totals) * 1.32)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.legend(loc='upper left', fontsize=8.5, framealpha=0.95)

post_means = [p1['ms_post'].mean(), p2['ms_post'].mean()]
bars3 = ax3.bar(['Phase 1\n($k_{obs}=1,5$)', 'Phase 2\n($k_{obs}=18,0$)'], post_means, width=0.46, color=['#7f8c8d', '#1f77b4'], edgecolor='black')

for bar in bars3:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.03, f"{yval:.2f} ms", ha='center', va='bottom', fontweight='bold', fontsize=10.5)

ax3.set_title('Potenciálmező Gauss-konvolúció', fontsize=12, fontweight='bold')
ax3.set_ylabel('Szintézis futásideje [ms]', fontsize=11, fontweight='bold')
ax3.set_ylim(0, max(post_means) * 1.32)
ax3.grid(axis='y', linestyle='--', alpha=0.5)

plt.figtext(0.5, 0.01, f"20 Hz-es valós idejű korlát: 50,0 ms (Phase 2 kihasználtság: {(totals[1]/50.0)*100:.1f}%) | Kiszűrt fals pozitív LiDAR cellák: {total_filtered_cells} db", 
            ha="center", fontsize=11, fontweight='bold', color='#2c3e50')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('tdk_abra_29.png', dpi=300)
print("tdk_abra_29.png")
print("tdk_meres.csv")
