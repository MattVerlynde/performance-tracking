from tensorflow.python.summary.summary_iterator import summary_iterator
import argparse
import matplotlib.pyplot as plt
import os
import plotly.express as px
import pandas as pd

def read_events(path_to_events_file, tag='train'):
    losses = []
    accuracies = []
    for e in summary_iterator(path_to_events_file):
        for v in e.summary.value:
            # print(v.tag)
            # print('loss/'+tag)
            if v.tag == 'loss/'+tag:
                losses.append(v.simple_value)
            elif v.tag == 'accuracy/'+tag:
                accuracies.append(v.simple_value)
    print(f"Read {len(losses)} losses and {len(accuracies)} accuracies from {path_to_events_file}")
    return losses, accuracies

def plot_losses(losses, save_path):
    os.makedirs(save_path, exist_ok=True)
    
    fig = px.scatter(x=range(len(losses)), y=losses, title='Losses')
    fig.write_html(save_path+'/losses.html', include_mathjax='cdn', include_plotlyjs='/home/verlyndem/Documents/cahier-labo-these/static/plotly.min.js')

    plt.plot(losses,"*")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(save_path+'/losses.png')
    print(f"Saved losses plot to {save_path}/losses.png")
    plt.show()
    plt.close()


def plot_accuracies(accuracies, save_path):
    os.makedirs(save_path, exist_ok=True)

    fig = px.scatter(x=range(len(accuracies)), y=accuracies, title='Accuracies')
    fig.write_html(save_path+'/accuracies.html', include_mathjax='cdn', include_plotlyjs='/home/verlyndem/Documents/cahier-labo-these/static/plotly.min.js')

    plt.plot(accuracies,"*")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.savefig(save_path+'/accuracies.png')
    # print(f"Saved accuracies plot to {save_path}/accuracies.png")
    plt.show()
    plt.close()

def plot_train_val(storage_path):
    save_path = storage_path + '/../plots'
    os.makedirs(save_path, exist_ok=True)
    for root, dirs, files in os.walk(storage_path):
        for name,tag in zip(files, ["train", "val"]):
            path = os.path.join(root, name)
            losses, accuracies = read_events(path,tag)
            plot_losses(losses,save_path+'/losses_'+tag)
            plot_accuracies(accuracies,save_path+'/accuracies_'+tag)

            losses, accuracies = read_events(os.path.join(root, files[0]), "train")
            epochs = len(losses)
            losses_val, accuracies_val = read_events(os.path.join(root, files[1]), "val")
            steps = epochs / len(losses_val)
            losses_all, accuracies_all, sample = [], [], []
            for i in range(epochs):
                if i % steps == 0:
                    j = int(i//steps)
                    losses_all.append(losses_val[j])
                    accuracies_all.append(accuracies_val[j])
                    sample.append("val")
                losses_all.append(losses[i])
                accuracies_all.append(accuracies[i])
                sample.append("train")
            plot_losses(losses_all,save_path+'/losses_all')
            plot_accuracies(accuracies_all,save_path+'/accuracies_all')
            pd.DataFrame({"loss":losses_all,"accuracy":accuracies_all,"sample":sample}).to_csv(save_path+'/../output/losses_accuracies_all.csv')
            
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--storage_path', type=str, required=True)
    args = parser.parse_args()
    model_path = args.storage_path + '/ShortCNN_RGB'
    plot_train_val(model_path)