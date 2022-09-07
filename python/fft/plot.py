import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.colorbar as cb
import matplotlib.ticker as ticker
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
  parser.add_argument('--csv', required=True, type=str,
                    help='CSV file containing data to be plotted.')
  parser.add_argument('--title', required=True, type=str,
                    help='Super-title for figure.')
  parser.add_argument('--cmap', default='magma', type=str, choices=plt.colormaps(),
                    help='Select the matplotlib colormap.')
  parser.add_argument('--marker-scale', default=650, type=float,
                    help='Set scale for marker sizes.')
  parser.add_argument('--clock-speed-ghz', default=1.85, type=float,
                    help='Set clock-speed in GHz used in conversion from cycles to milliseconds.')
  parser.add_argument('--x-field', default=None, type=str,
                    help='CSV column heading for x-axis data. If None, first column of CSV will be used.')
  parser.add_argument('--y-field', default=None, type=str,
                    help='CSV column heading for y-axis data. If None, second column of CSV will be used.')
  parser.add_argument('--color-field', default=None, type=str,
                    help='CSV column heading for colour data. If None, marker colour will not be data dependent.')
  parser.add_argument('--size-field', default=None, type=str,
                    help='CSV column heading for marker size data. If None, marker size will not be data dependent.')
  args = parser.parse_args()

  df = pd.read_csv(args.csv, header=0)
  df = df.apply(pd.to_numeric, errors="coerce")
  headers = df.columns.values
  print(f"Column Headers:\n{headers}\n")

  if 'Cycles' in df and not 'Time (ms)' in df:
    print(f"Computing timings based on clock speed of {args.clock_speed_ghz} GHz")
    df['Time (ms)'] = df['Cycles'] / (args.clock_speed_ghz * 1e6)

  x_field = args.x_field or headers[0] # 'Input-size'
  y_field = args.y_field or headers[1] # 'Batch-size'
  color_field = args.color_field # 'Time (ms)'
  size_field = args.size_field # 'Radix-size'

  plot_cols = 1
  grid_spec=None

  if size_field:
    size_scale = args.marker_scale
    size_exp = 2
    # Avoid NaNs:
    df[size_field] = df[size_field].astype(dtype=np.float)
    sz_min = min(df[size_field])
    sz_max = max(df[size_field])
    df[size_field] = df[size_field].fillna(value=sz_min)
    print(f"Size min/max: {sz_min} / {sz_max}")
    # Sort data by reverse of size field so all markers are visible:
    df = df.sort_values(by=[size_field], ascending=False)
    size_norm = colors.LogNorm(vmin=sz_min, vmax=sz_max)
    ns = size_norm(df[size_field])
    sizes = [ (size_scale * (0.1 + v)**size_exp) for v in ns ]
    plot_cols = 2
    grid_spec = {'width_ratios': [7, 1]}

  if color_field:
    col_min = min(df[color_field])
    col_max = max(df[color_field])
    print(f"Color min/max: {col_min} / {col_max}")
    color_norm = colors.LogNorm(vmin=col_min, vmax=col_max)
    nc = color_norm(df[color_field])
    cmap = cm.get_cmap(args.cmap)
  else:
    color_norm = None
    nc = None
    cmap = None

  fig, axes = plt.subplots(nrows=1, ncols=plot_cols, figsize=(9,6), gridspec_kw=grid_spec)
  fig.suptitle(args.title)

  if  size_field:
    axis1 = axes[0]
  else:
    axis1 = axes
    sizes = None

  plt1 = df.plot(ax=axis1,
                kind='scatter', x=x_field, y=y_field,
                loglog=True,
                s=sizes,
                c=nc, cmap=cmap, colorbar=False)
  plt1.set_xscale('log', base=2)
  plt1.set_yscale('log', base=2)

  if color_field:
    smap = cm.ScalarMappable(norm=color_norm, cmap=cmap)
    formatter = ticker.LogFormatterMathtext(10)
    cb1 = fig.colorbar(mappable=smap, ax=axis1, format=formatter, norm=color_norm)
    cb1.set_label(color_field)

  if size_field:
    # Make a legend to help interpret marker sizes:
    axis2 = axes[1]
    marker_values = np.unique(df[size_field])
    marker_sizes = np.unique(sizes)
    axis2.scatter(y=marker_values, x=np.zeros_like(marker_values), s=marker_sizes, color=[0.4,0.4,0.4])
    axis2.set_yscale('log', base=2)#
    if len(marker_values) < 20:
      axis2.yaxis.set_ticks(marker_values)
    axis2.yaxis.tick_right()
    axis2.yaxis.set_label_position('right')
    axis2.set_ylabel(size_field)
    axis2.get_xaxis().set_visible(False)
    axis2.set_title(f"Marker Size Legend", size=9)
    # print(f"val: {marker_values}")
    # print(f"sizes: {marker_sizes}")

  fig.savefig('plot.png')
