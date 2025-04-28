import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import colorsys

# constant
ELO = 400 / np.log(10)

# load data
df = pd.read_csv('boardlaw-paper-v2/board_source_data.csv')
boardsizes = sorted(df['boardsize'].unique())

# build ggplot2‐style hue palette
hues = np.linspace(15, 375, len(boardsizes)+1)[:-1] / 360.0
palette = [colorsys.hls_to_rgb(h, 0.4, 1.0) for h in hues]

fig, ax = plt.subplots(figsize=(10, 6))

for i, bs in enumerate(boardsizes):
    sub = df[df['boardsize'] == bs].sort_values('train_flops')
    col = palette[i]
    # solid line: elohat * ELO, labeled for legend
    ax.plot(sub['train_flops'], sub['elohat'] * ELO,
            '-', linewidth=2.0, color=col, label=f'{bs}×{bs}')
    # dashed line: elo * ELO (no legend entry)
    ax.plot(sub['train_flops'], sub['elo'] * ELO,
            '--', linewidth=3.0, color=col)

ax.set_xscale('log')
ax.set_ylim(top=0)
ax.set_xlabel('Training compute (FLOPS-seconds)')
ax.set_ylabel('Elo v. perfect play')

# simple legend
ax.legend()

# thicken axes
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
ax.tick_params(width=1.2, length=6)

plt.tight_layout()
# save or show
fig.savefig('frontiers.png', dpi=600, bbox_inches='tight')
plt.show()
