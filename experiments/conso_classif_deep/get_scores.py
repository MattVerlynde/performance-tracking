import argparse
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def aggregate_scores(storage_path):
    scores = []
    for name in os.listdir(storage_path):
        if name[-5:] != '.yaml' and name[-4:] != '.csv' and name[-4:] != '.png' and name[-5:] != '.html':
            path = os.path.join(storage_path, name+"/output/losses_accuracies_all.csv")
            score = pd.read_csv(path)
            score['experiment'] = name.split('_')[1]
            scores.append(score)
    scores = pd.concat(scores)
    print(scores.head())
    groups = sorted([int(x) for x in scores['experiment'].unique()])
    print(groups)
    results_all = []
    for i in range(3):
        new_scores = []
        for j in range(i, len(groups), 3):
            print(groups[j])
            scores_j = scores[scores['experiment'] == str(groups[j])]
            print(scores_j.head())
            new_scores.append(scores_j)
            samples = scores_j['sample']
        if len(new_scores) == 0:
            new_scores = new_scores[0]
        else:
            new_scores = pd.concat(new_scores)
        means = new_scores.groupby('Unnamed: 0')[['loss','accuracy']].mean()
        means.columns = [f'{col}_mean' for col in means.columns]
        stds = new_scores.groupby('Unnamed: 0')[['loss','accuracy']].std()
        stds.columns = [f'{col}_std' for col in stds.columns]
        results = pd.concat([means, stds, samples], axis=1)
        # plt.plot(results['loss_mean'], label=f'loss_{i}')
        results_all.append(results)
    return results_all

def aggregate_performances(storage_path):
    scores = []
    for name in os.listdir(storage_path):
        if name[-5:] != '.yaml' and name[-4:] != '.csv' and name[-4:] != '.png' and name[-5:] != '.html':
            path = os.path.join(storage_path, name+"/output/scores.csv")
            score = pd.read_csv(path)
            score['experiment'] = name.split('_')[1]
            scores.append(score)
    scores = pd.concat(scores)
    # print(scores.head())
    groups = sorted([int(x) for x in scores['experiment'].unique()])
    # print(groups)
    results_all = []
    for i in range(3):
        new_scores = []
        lr = [0.0001, 0.001, 0.01]
        for j in range(i, len(groups), 3):
            scores_j = scores[scores['experiment'] == str(groups[j])]
            new_scores.append(scores_j)
        new_scores = pd.concat(new_scores)
        means = new_scores.groupby('Unnamed: 0').mean()
        means.columns = [f'{col}_mean' for col in means.columns]
        stds = new_scores.groupby('Unnamed: 0').std()
        stds.columns = [f'{col}_std' for col in stds.columns]
        results = pd.concat([means, stds], axis=1)
        # plt.plot(results['loss_mean'], label=f'loss_{i}')
        results['lr'] = lr[i]
        results_all.append(results)
    return pd.concat(results_all)

def plot_all(storage_path):
    all_scores = aggregate_scores(storage_path)
    print(all_scores[0].head())
    lr = [0.0001, 0.001, 0.01]
    for i in range(3):
        for metric in ['accuracy', 'loss']:   
            fig = go.Figure(layout_title_text=f'{metric} (lr = {lr[i]})')         
            for j,sample in enumerate(['train', 'val']):
                sample_scores = all_scores[i][all_scores[i]['sample'] == sample]
                color = ['blue', 'red']
                color_alpha = ['rgb(0,0,255,0.5)', 'rgb(255,0,0,0.5)']

                fig.add_trace(go.Scatter(x=sample_scores.index, y=sample_scores[f'{metric}_mean'], line_color=color[j], name=sample, mode ="lines+markers"))
                # fig.update_traces(line_color=color[j])
                fig.add_trace(go.Scatter(x=sample_scores.index, y = sample_scores[f'{metric}_mean'] - 1.96 * sample_scores[f'{metric}_std'] / 5**0.5,
                                    line = dict(color=color[j], width=0), mode ="lines", showlegend=False))
            
                fig.add_trace(go.Scatter(x=sample_scores.index, y = sample_scores[f'{metric}_mean'] + 1.96 * sample_scores[f'{metric}_std'] / 5**0.5,
                                    line = dict(color=color[j], width=0), mode ="lines", showlegend=False,
                                    fill='tonexty', opacity=0.001))
            fig.write_image(storage_path+f'/{metric}_{lr[i]}.png')
            # fig.write_html(storage_path+f'/{metric}_{lr[i]}.html', include_plotlyjs='/home/verlyndem/Documents/cahier-labo-these/static/plotly.min.js')
            fig.write_html(storage_path+f'/{metric}_{lr[i]}.html')
        # fig = px.line(all_scores[i], y='loss_mean', color='sample', title=f'Loss (lr = {lr[i]})')
        # fig.add_traces(go.Scatter(x=all_scores[i].index, y = all_scores[i]['loss_mean'] - all_scores[i]['loss_std'],
        #                       line = dict(color='sample')))
    
        # fig.add_traces(go.Scatter(x=all_scores[i].index, y = all_scores[i]['loss_mean'] + all_scores[i]['loss_std'],
        #                       line = dict(color='sample')),
        #                       fill='tonexty', 
        #                       fillcolor = dict(color='sample'))
        # fig.write_image(storage_path+f'/loss_{lr[i]}.png')
        # fig.write_html(storage_path+f'/loss_{lr[i]}.html', include_plotlyjs='/home/verlyndem/Documents/cahier-labo-these/static/plotly.min.js')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--storage_path', type=str, required=True)
    args = parser.parse_args()
    plot_all(args.storage_path)
    aggregate_performances(args.storage_path).to_csv(args.storage_path+'/performances.csv', float_format='%.3f')


   