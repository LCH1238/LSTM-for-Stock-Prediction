import matplotlib.pyplot as plt
import numpy as np


def plot_predictions(pred, target, method, show=True):
    target = np.array(target)
    pred = np.array(pred)
    
    MAE = np.abs(target - pred).mean()
    
    plt.figure(figsize=(13, 8))
    plt.title(f'{method} prediction results, MAE={MAE:.3f}')
    plt.plot(target, label='Ground Truth')
    plt.plot(pred, label='Prediction')
    plt.legend()
    
    path = f'images/{method}_results.png'
    plt.savefig(path)
    
    if show:
        plt.show()
    plt.close()
