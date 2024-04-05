# Get statistics for each simulation parameter
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots

colors = ['red', 'green', 'blue']
colorsa = ['rgba(255, 0, 0, 0.1)', 'rgba(0, 255, 0, 0.1)', 'rgba(0, 0, 255, 0.1)']

def plot_cpu(df_field, size_image, size_window, i = 1, measurement='cpu', field='usage_idle', extra_measurement='cpu', extra_measurement_value='cpu-total', title=f'CPU usage', xaxis_title='Time (s)', yaxis_title='CPU usage (%)', legend_title='Window size (px)', yaxis_range=[0,100]):
    for t in range(1, 11): # for each simulation
        env = f"/home/verlyndem/Documents/Tests_change_detection/SAR-change-detection/results_simulation/{size_image}x4.{size_window}.12.{t}.metrics_output"
        with open(env, 'r') as file:
            df_list = file.read().split('#group')
            df_list.pop()
            
        file_name = env+f".{i}.csv"
        with open(file_name, 'w') as file:
            file.write(df_list[i])
                
        df = pd.read_csv(file_name, sep=',', skiprows=5, header=0, index_col=False).drop(columns=['Unnamed: 0','result','table'])

        df_fieldi = df.where((df['_field'] == field) & (df['_measurement'] == measurement)).dropna().reset_index()

        if extra_measurement:
            df_fieldi = df_fieldi.where(df_fieldi[extra_measurement] == extra_measurement_value).dropna().reset_index()
        
        print(df_fieldi.head())

        os.remove(file_name)
            
        df_field = df_field._append(df_fieldi[['_start', '_time', '_value']], ignore_index=False)

    df_field = df_field.reset_index()[['index', '_value']]
    df_field['_start'] = pd.to_datetime(df['_start'], format='ISO8601')
    df_field['_time'] = pd.to_datetime(df['_time'], format='ISO8601')
    df_field['time'] = (df_field['_time'] - df_field['_start']).dt.total_seconds()

    df_field_mean = df_field.groupby('time').mean().reset_index()
    df_field_std = 2*df_field.groupby('time').std().reset_index()

    scatter_fig = px.scatter(x=df_field['time'], y=df_field['_value'])
    trace = scatter_fig.data[0]
    trace.marker.color=colors[color_i]
    fig.add_traces(scatter_fig.data[0])

    line_fig = px.line(x=df_field_mean['time'], y=df_field_mean['_value'])
    trace = line_fig.data[0]
    trace.line.color=colors[color_i]
    trace.name=f'{size_window}x{size_window}'
    trace.showlegend=True
    fig.add_traces(trace)
        
    line_fig_up = px.line(x=df_field_mean['time'], y=df_field_mean['_value']+df_field_std['_value'])
    line_fig_down = px.line(x=df_field_mean['time'], y=df_field_mean['_value']-df_field_std['_value'])
        
    trace_up = line_fig_up.data[0]
    trace_up.line.color=colorsa[color_i]
    trace_up.opacity=0.2
        
    trace_down = line_fig_down.data[0]
    trace_down.fill='tonexty'
    trace_down.fillcolor=colorsa[color_i]
    trace_down.line.color=colorsa[color_i]
        
    fig.add_traces(trace_up)
    fig.add_traces(trace_down)

    fig.update_layout(title=title+f' for {size_image}x4 image', xaxis_title=xaxis_title, yaxis_title=yaxis_title, legend_title=legend_title, yaxis_range=yaxis_range)
    return fig

# for i in range(1,9): number of tables per simulation
# for size_image in ['500x500', '500x1000', '1000x1000']:
#     fig = make_subplots(rows=1, cols=1)
#     for color_i,size_window in enumerate([5, 11, 21]):
#         df_cpu = pd.DataFrame(columns=['_start', '_time', '_value'])
#         fig = plot_cpu(df_cpu, size_image, size_window, i = 1, measurement='cpu', field='usage_idle', extra_measurement='cpu', extra_measurement_value='cpu-total', title=f'CPU usage', xaxis_title='Time (s)', yaxis_title='CPU usage (%)', legend_title='Window size (px)')
#         # plot_cpu(df_cpu, size_image, size_window).write_html(f'cpu_usage_{size_image}x4.html', include_mathjax='cdn')
#     fig.show()

for size_image in ['500x500', '500x1000', '1000x1000']:
    fig = make_subplots(rows=1, cols=1)
    for color_i,size_window in enumerate([5, 11, 21]):
        df_mem = pd.DataFrame(columns=['_start', '_time', '_value'])
        fig = plot_cpu(df_mem, size_image, size_window, i = 6, measurement='mem', field='used_percent', extra_measurement=False, title=f'Memory usage', xaxis_title='Time (s)', yaxis_title='Memory usage (%)', legend_title='Window size (px)', yaxis_range=[0,100])
    # fig.write_html(f'cpu_usage_{size_image}x4.html', include_mathjax='cdn')
    fig.show()
