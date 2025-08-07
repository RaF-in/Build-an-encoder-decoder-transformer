import matplotlib.pyplot as plt
import torch
from model import Model

param_grads = torch.load('param_grads.pt')
gradient_values = torch.load('activation_grads.pt')

def plot_gradients_activations():
    legends = []
    for name, grad in gradient_values.items():
        std, mean = grad.std().item(), grad.mean().item()
        hy, hx = torch.histogram(grad, density=True)
        legends.append(f"{name}: μ={mean:+.4f}, σ={std:.4e}")
        plt.plot(hx[:-1], hy)
        print(f"name: {name}, std: {std:.2f}, mean: {mean:.4e}")
    plt.title("Gradient Distributions for Activations (using named_modules())")
    # plt.legend(legends)
    plt.xlabel('gradient values')
    plt.ylabel('density')
    plt.show()



def plot_gradients_weights():
    legends = []
    plt.figure(figsize=(20, 4)) # width and height of the plot
    for name, grad in param_grads.items():
        std, mean = grad.std().item(), grad.mean().item()
        hy, hx = torch.histogram(grad, density=True)
        legends.append(f"{name}: μ={mean:+.4f}, σ={std:.4e}")
        plt.plot(hx[:-1], hy)
        print(f"name: {name}, std: {std:.2f}, mean: {mean:.4e}")
    plt.title("Gradient Distributions for Parameters")
    # plt.legend(legends)
    plt.xlabel('gradient values')
    plt.ylabel('density')
    plt.show()

if __name__ == '__main__':
    path = 'trained_model2.pth'
    plot_gradients_weights()
    # plot_gradients_activations()
    # print(gradient_values)

