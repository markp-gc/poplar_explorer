import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.colorbar as cb
import argparse
import numpy as np


def log_ratio_min_max(x):
  log_x = np.log(x)
  log_max = max(log_x)
  log_min = min(log_x)
  ratios = [ (v - log_min)/log_max for v in log_x ]
  return ratios, min, max


def ratio_min_max(x):
  min_v = min(x)
  max_v = max(x)
  ratios = [ (v - min_v)/max_v for v in x ]
  return ratios, min_v, max_v


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Plotter.')
  parser.add_argument('--csv', required=True,
                    help='CSV file containing data to be plotteed.')
  args = parser.parse_args()

  df = pd.read_csv(args.csv, header=0)
  df = df.apply(pd.to_numeric, errors="coerce")
  #df = df.astype(dtype=np.float)
  headers = df.columns.values
  print(f"CSV Headers: {headers}")
  #df = df.fillna(value=0.0)

  time_ms = df['Cycles'] / 1.85e6
  df['Time (ms)'] = time_ms

  x_field = 'Input-size'
  y_field = 'Batch-size'
  color_field = 'Time (ms)'
  size_field = 'Radix-size'
  size_scale = 700
  size_exp = 2
  # Sort data by reverse of size field so all are visible:
  df = df.sort_values(by=[size_field], ascending=False)

  cmap = cm.get_cmap('magma')
  ratios_bs, log_min_bs, log_max_bs = log_ratio_min_max(df[color_field])

  ratios_rdx, _, _ = log_ratio_min_max(df[size_field])
  rdx_sizes = [ (size_scale * (0.1 + v)**size_exp) for v in ratios_rdx ]

  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,6), gridspec_kw={'width_ratios': [5, 1]})
  fig.suptitle("FFT-1D Sweep Summary Plot")

  plt1 = df.plot(ax=axes[0],
                kind='scatter', x=x_field, y=y_field,
                logx=True, logy=True,
                s=rdx_sizes,
                c=ratios_bs, cmap=cmap)
  plt1.set_xscale('log', base=2)
  plt1.set_yscale('log', base=2)

  norm = colors.Normalize(vmin=min(df[color_field]), vmax=max(df[color_field]))
  smap = cm.ScalarMappable(norm=norm, cmap=cmap)
  cb1 = fig.colorbar(mappable=smap, ax=axes[0])
  cb1.set_label(color_field)

  # Make a legend to help interpret marker sizes:
  marker_values = np.unique(df[size_field])
  marker_sizes = np.unique(rdx_sizes)
  axes[1].scatter(y=marker_values, x=np.zeros_like(marker_values), s=marker_sizes)
  axes[1].set_yscale('log', base=2)
  axes[1].yaxis.tick_right()
  axes[1].yaxis.set_label_position('right')
  axes[1].set_ylabel(size_field)
  axes[1].get_xaxis().set_visible(False)
  axes[1].set_title(f"Marker Sizes")
  # print(f"val: {marker_values}")
  # print(f"sizes: {marker_sizes}")

  fig.savefig('plot.png')
