import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

f = 'execution_time'
df = pd.read_csv(f, sep=',', header=0, index_col=False)
df['time'] = df['time']/1000000000
df['size_image'] = df['name'].str.split('x').str[0] + 'x' + df['name'].str.split('x').str[1]
df['size_sample'] = df['name'].str.split('x').str[2]


df['size_window'] = df['size_window'].astype(str)
df['name_full'] = df['name'] + '_' + df['size_window'].astype(str) + '_' + df['cores'].astype(str)
mean = (df.groupby('name_full')["time"].mean())
std = (df.groupby('name_full')["time"].std())

stats = pd.DataFrame({'mean': mean, 'std': std})
print(stats)

# Plot the statistics in boxplots

# Plot results by window size

# Creating subplot axes
names = np.sort(df['name'].unique())[::-1]
print(names)
ncols = len(names)

fig = make_subplots(
    rows=1, cols=ncols, subplot_titles=names, horizontal_spacing=0.05
)

for name, ax in zip(names, range(1,ncols+1)):
    df_ax = df.where(df['name'] == name)
    fig.add_trace(go.Box(y=df_ax['time'], x= df_ax["size_window"], boxmean=True),
                row=1, col=ax)
    fig.update_xaxes(title_text="Window size (px)", row=1, col=ax)
    fig.update_yaxes(title_text="Execution time (s)", row=1, col=ax)

fig.update_layout(height=700, showlegend=False)
boxmean=True
fig.show()

# fig.write_html('result_simulation_time.html', include_mathjax='cdn')

# Plot results by image size
# Creating subplot axes
names = np.sort(df['size_window'].unique()).astype(str)
print(names)
ncols = len(names)

fig = make_subplots(
    rows=1, cols=ncols, subplot_titles=names, horizontal_spacing=0.05
)

for name, ax in zip(names, range(1,ncols+1)):
    df_ax = df.where(df['size_window'].astype(str) == name).sort_values(by='name', ascending=False)
    df_ax = df_ax.where(df_ax['size_sample'].astype(str) == '4')
    print(df_ax)
    fig.add_trace(go.Box(y=df_ax['time'], x= df_ax["name"], boxmean=True),
                row=1, col=ax)
    fig.update_xaxes(title_text="Sample size (px)", row=1, col=ax)
    fig.update_yaxes(title_text="Execution time (s)", row=1, col=ax)

fig.update_layout(height=700, showlegend=False)

fig.show()

# fig.write_html('result_simulation_time_window.html', include_mathjax='cdn')

# Plot results by all the parameters
fig = px.box(df.where(df['size_sample'] == '4').sort_values(by='name', ascending=False), x="size_image", y="time", color="size_window")
fig.update_xaxes(title_text="Image size (px)")
fig.update_yaxes(title_text="Execution time (s)")
fig.update_layout(legend_title_text='Window size (px)')

# fig.show()
fig.write_html('result_simulation_time_imagesize.html', include_mathjax='cdn')
