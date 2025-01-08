# Get statistics for each simulation parameter
import os
import pandas as pd
import plotly.express as px
import argparse

colors = ['red', 'green', 'blue']
colorsa = ['rgba(255, 0, 0, 0.1)', 'rgba(0, 255, 0, 0.1)', 'rgba(0, 0, 255, 0.1)']

def plot_usage(env, i = 1, measurement='cpu', field='usage_idle', extra_measurement='cpu', extra_measurement_value='cpu-total', title=f'CPU usage', xaxis_title='Time (s)', yaxis_title='CPU usage (%)', yaxis_range=[0,100]):
    with open(env, 'r') as file:
        df_list = file.read().split('#group')
        df_list.pop()
            
    file_name = env+f".{i}.csv"
    with open(file_name, 'w') as file:
        file.write(df_list[i])
                
    df = pd.read_csv(file_name, sep=',', skiprows=5, header=0, index_col=False).drop(columns=['Unnamed: 0','result','table'])

    df_field = df.where((df['_field'] == field) & (df['_measurement'] == measurement)).dropna().reset_index()

    if extra_measurement:
        df_field = df_field.where(df_field[extra_measurement] == extra_measurement_value).dropna().reset_index()

    os.remove(file_name)
            
    df_field = df_field[['_start', '_time', '_value']]

    df_field = df_field.reset_index()[['index', '_value']]
    df_field['_start'] = pd.to_datetime(df['_start'], format='ISO8601')
    df_field['_time'] = pd.to_datetime(df['_time'], format='ISO8601')
    df_field['time'] = (df_field['_time'] - df_field['_start']).dt.total_seconds()


    fig = px.scatter(x=df_field['time'], y=df_field['_value'], title=title, labels={xaxis_title, yaxis_title}, range_y=yaxis_range)
    return fig


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_path", type=str, required=True)
    args = parser.parse_args()

    results_path = os.path.join(args.storage_path, "results.txt")

    fig = plot_usage(results_path, i = 1, measurement='cpu', field='usage_idle', extra_measurement=False, title=f'CPU usage', xaxis_title='Time (s)', yaxis_title='CPU usage (%)', yaxis_range=[0,100])

    fig.write_html(os.path.join(args.storage_path, "usage.png"), include_mathjax='cdn', include_plotlyjs='/home/verlyndem/Documents/cahier-labo-these/static/plotly.min.js')
    fig.show()
