from tensorflow.python.summary.summary_iterator import summary_iterator
import argparse
import matplotlib.pyplot as plt
import os
import plotly.express as px
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def read_events(path_to_events_file, tag='train'):
    losses = []
    epochs = []
    accuracies = []
    i=0
    for e in summary_iterator(path_to_events_file):
        for v in e.summary.value:
            # print(v.tag)
            # print('loss/'+tag)
            if v.tag == 'loss/'+tag or v.tag == 'accuracy/'+tag:
                epochs.append(i)
                if v.tag == 'loss/'+tag:
                    losses.append(v.simple_value)
                elif v.tag == 'accuracy/'+tag:
                    accuracies.append(v.simple_value)
                    i+=1
            print(f"EPOCH {i}")
    print(f"Read {len(losses)} losses and {len(accuracies)} accuracies from {path_to_events_file}")
    return losses, accuracies

def read_event_solo(path_to_events_file):
    losses = []
    accuracies = []
    sample = []
    for e in summary_iterator(path_to_events_file):
        for v in e.summary.value:
            # print(v.tag)
            # print('loss/'+tag)
            if v.tag == 'loss/train' or v.tag == 'loss/val':
                losses.append(v.simple_value)
                if v.tag == 'loss/train':
                    sample.append("train")
                else:
                    sample.append("val")
            elif v.tag == 'accuracy/train' or v.tag == 'accuracy/val':
                accuracies.append(v.simple_value)
    return losses, accuracies, sample

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
    print(f"Saved accuracies plot to {save_path}/accuracies.png")
    plt.show()
    plt.close()

def plot_train_val(storage_path):
    save_path = storage_path + '/../plots'
    os.makedirs(save_path, exist_ok=True)
    for root, dirs, files in os.walk(storage_path):
        for name in files:
            for tag in ["val", "train"]:
                path = os.path.join(root, name)
                losses, accuracies = read_events(path,tag)
                plot_losses(losses,save_path+'/losses_'+tag)
                plot_accuracies(accuracies,save_path+'/accuracies_'+tag)

                losses, accuracies, sample = read_event_solo(path)
                plot_losses(losses,save_path+'/losses_all')
                plot_accuracies(accuracies,save_path+'/accuracies_all')
                pd.DataFrame({"loss":losses, "accuracy":accuracies, "sample":sample}).to_csv(save_path+'/../output/losses_accuracies_all.csv')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--storage_path', type=str, required=True)
    args = parser.parse_args()
    model_path = args.storage_path + '/ShortCNN_RGB'
    plot_train_val(model_path)