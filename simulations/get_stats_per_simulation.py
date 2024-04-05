# Get statistics for each simulation parameter
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots

colors = ['red', 'green', 'blue']
colorsa = ['rgba(255, 0, 0, 0.1)', 'rgba(0, 255, 0, 0.1)', 'rgba(0, 0, 255, 0.1)']

i = 1 # for i in range(1,9): number of tables per simulation
for size_image in ['500x500', '500x1000', '1000x1000']:
    fig = make_subplots(rows=1, cols=1)
    for color_i,size_window in enumerate([5, 11, 21]):
        df_cpu = pd.DataFrame(columns=['_start', '_time', '_value'])
        for t in range(1, 11): # for t in range(1, 11):
            env = f"/home/verlyndem/Documents/Tests_change_detection/SAR-change-detection/results_simulation/{size_image}x4.{size_window}.12.{t}.metrics_output"
            with open(env, 'r') as file:
                df_list = file.read().split('#group')
                df_list.pop()
                print(len(df_list))
            
            
            file_name = env+f".{i}.csv"
            with open(file_name, 'w') as file:
                file.write(df_list[i])
                
            df = pd.read_csv(file_name, sep=',', skiprows=5, header=0, index_col=False).drop(columns=['Unnamed: 0','result','table'])

            df_cpui = df.where((df['_field'] == 'usage_idle') & (df['_measurement'] == 'cpu') & (df['cpu'] == 'cpu-total')).dropna().reset_index()

            os.remove(file_name)
            
            # df_cpu = pd.concat([df_cpu, df_cpui[['_time', '_value']]], axis=0)
            df_cpu = df_cpu._append(df_cpui[['_start', '_time', '_value']], ignore_index=False)

        df_cpu = df_cpu.reset_index()[['index', '_value']]
        df_cpu['_start'] = pd.to_datetime(df['_start'], format='ISO8601')
        df_cpu['_time'] = pd.to_datetime(df['_time'], format='ISO8601')
        df_cpu['time'] = (df_cpu['_time'] - df_cpu['_start']).dt.total_seconds()

        df_cpu_mean = df_cpu.groupby('time').mean().reset_index()
        df_cpu_std = 2*df_cpu.groupby('time').std().reset_index()

        # plt.plot(df_cpu['time'], df_cpu['_value'], '.', alpha=0.8, color=colors[color_i])
        # plt.plot(df_cpu_mean.index, df_cpu_mean['_value'], alpha=0.8, color=colors[color_i], label=f'{size_window}x{size_window}')
        # plt.fill_between(df_cpu_mean.index, df_cpu_mean['_value']-df_cpu_std['_value'], df_cpu_mean['_value']+df_cpu_std['_value'], alpha=0.2, color=colors[color_i])
        # plt.xlabel('Time (s)')
        # plt.ylabel('CPU usage (%)')

        scatter_fig = px.scatter(x=df_cpu['time'], y=df_cpu['_value'])
        trace = scatter_fig.data[0]
        #set marker color
        trace.marker.color=colors[color_i]
        fig.add_traces(scatter_fig.data[0])

        line_fig = px.line(x=df_cpu_mean['time'], y=df_cpu_mean['_value'], labels={'_value': f'{size_window}x{size_window}'})
        trace = line_fig.data[0]
        trace.line.color=colors[color_i]
        trace.name=f'{size_window}x{size_window}'
        trace.showlegend=True
        fig.add_traces(trace)
        
        line_fig_up = px.line(x=df_cpu_mean['time'], y=df_cpu_mean['_value']+df_cpu_std['_value'])
        line_fig_down = px.line(x=df_cpu_mean['time'], y=df_cpu_mean['_value']-df_cpu_std['_value'])
        
        trace_up = line_fig_up.data[0]
        trace_up.line.color=colorsa[color_i]
        trace_up.opacity=0.2
        
        trace_down = line_fig_down.data[0]
        trace_down.fill='tonexty'
        trace_down.fillcolor=colorsa[color_i]
        trace_down.line.color=colorsa[color_i]
        
        fig.add_traces(trace_up)
        fig.add_traces(trace_down)

        fig.update_layout(title=f'CPU usage for {size_image}x4 sample', xaxis_title='Time (s)', yaxis_title='CPU usage (%)', legend_title='Window size (px)')

    fig.show()
    # fig.write_html(f'cpu_usage_{size_image}x4.html', include_mathjax='cdn')
    # plt.title(f'CPU usage for {size_image}x4 sample')
    # plt.legend(title='Window size', loc='lower right')
    # plt.ylim([0, 100])
    # plt.show()
        
    