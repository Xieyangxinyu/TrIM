import jax.numpy as jnp
from jax import grad, vmap
from scipy.stats import special_ortho_group
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import qr, svd, eig
from tqdm import tqdm
from Mondrian_RF.Mondrian_forest import MondrianForestTransformer
from Mondrian_RF.utils import evaluate_all_lifetimes, two_one_norm
from matplotlib.ticker import ScalarFormatter


def plot_trend(stats, x_axis, y_axis, xlabel = None, ylabel = None):
    df = pd.DataFrame(stats)
    # Prepare the data for plotting
    plot_data = [df[y_axis][i] for i in range(len(df))]

    # Create the boxplot
    plt.boxplot(plot_data, labels=[str(x[x_axis]) for x in stats])

    # Add titles and labels
    if xlabel is None:
        xlabel = x_axis
    if ylabel is None:
        ylabel = y_axis

    # set y range to 0-1.6
    plt.ylim(0, 1.6)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

def get_eigvectors(A):
    eigenValues, eigenVectors = eig(A)

    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return eigenVectors

def get_angle_distance(H, true_H, active):
    H = H / two_one_norm(H)
    true_H = true_H / two_one_norm(true_H)
    active = min(active, np.linalg.matrix_rank(true_H))
    eigenvectors = get_eigvectors(H)
    Q_H, _ = qr(eigenvectors[:, :active])
    
    eigenvectors = get_eigvectors(true_H)
    Q_t, _ = qr(eigenvectors[:, :active])

    D = np.matmul(np.transpose(Q_H), Q_t)
    _, S, _ = svd(D, full_matrices=True)
    S = np.minimum(S, 1)
    return np.max(np.arccos(S))

def plot_dist(forests, true_H, sample_range, active, tries):
    plot_data = []
    for i in range(len(sample_range)):
        n_sim = sample_range[i]
        dist = {'n': n_sim, 'dist':[]}
        for trial in range(tries):
            H_0 = forests[n_sim][trial].H
            dist['dist'].append(get_angle_distance(H_0, true_H, active))
        plot_data.append(dist)
    plot_trend(plot_data, 'n', 'dist', xlabel = 'Sample Size n', ylabel='Angle Distance from the True H')


class SimulatedData:
    def __init__(self, dim_in, active, rotation = None, fun = "poly", noise_sig2 = 0.01, seed = 0):
        self.dim_in = dim_in
        self.active = active
        self.noise_sig2 = noise_sig2
        self.seed = seed
        self.project = jnp.concatenate([jnp.ones(active), jnp.zeros(dim_in - active)])

        if rotation == None:
            self.rotation = np.identity(dim_in)
        elif rotation == "simple":
            self.rotation = np.identity(dim_in)
            self.rotation[1,0] = self.rotation[2,0] = self.rotation[3,1] = self.rotation[4,1] = 1
        elif rotation == "orth":
            rv = special_ortho_group(dim_in, seed = seed)
            self.rotation = rv.rvs()
        else:
            self.rotation = rotation

        if fun == "poly":
            self.fun = self.poly_fn
        elif fun == "max":
            self.fun = self.max_fn

    def add_noise(self, y):
        r_noise = np.random.RandomState(self.seed)
        noise = r_noise.randn(1)[0] * jnp.sqrt(self.noise_sig2)
        y = y + noise
        return y
    
    def poly_fn(self, x):
        res = x @ self.rotation
        y = res ** 4 @ self.project
        return self.add_noise(y)
    
    def max_fn(self, x):
        res = x @ self.rotation
        res = (res ** 2) * -0.25
        res = jnp.exp(res)
        y = jnp.max(res * self.project)
        return self.add_noise(y)
    
    def get_true_H(self, x_test):
        n_test = x_test.shape[0]
        W_grad = vmap(grad(self.fun), in_axes=0, out_axes=0)(x_test)
        true_H = jnp.matmul(jnp.transpose(W_grad), W_grad) / n_test
        return true_H
    

class Simulation():
    def __init__(self, dim_in, active, n_estimators, tries, lifetime, step_size, sample_range, x_train, x_test, y_train, y_test, true_H):
        self.dim_in = dim_in
        self.active = active
        self.n_estimators = n_estimators
        self.tries = tries
        self.lifetime = lifetime
        self.step_size = step_size
        self.sample_range = sample_range
        self.forests = {}
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.evaluation_results = []
        self.true_H = true_H

    def train(self):
        print("Training the Mondrian Forests:")
        for n_sim in self.sample_range:
            self.forests[n_sim] = []
            for trial in tqdm(range(self.tries)):
                forest = MondrianForestTransformer(n_estimators=self.n_estimators, lifetime=self.lifetime, 
                                                   step_size=self.step_size, random_state=(n_sim + 1) * (trial + 1))
                forest.fit(self.x_train[:n_sim], self.y_train[:n_sim])
                self.forests[n_sim].append(forest)
        plot_dist(self.forests, self.true_H, self.sample_range, self.active, self.tries)
    
    def evaluation_comparison(self):
        x_train = self.x_train
        x_test = self.x_test
        y_train = self.y_train
        y_test = self.y_test
        self.evaluation_results.append(
            evaluate_all_lifetimes(x_train, y_train, x_test, y_test, self.n_estimators, self.lifetime))
        
        normalized_true_H = self.true_H / two_one_norm(self.true_H)
        x_train_transformed = np.matmul(x_train, normalized_true_H)
        x_test_transformed = np.matmul(x_test, normalized_true_H)
        self.evaluation_results.append(
            evaluate_all_lifetimes(x_train_transformed, y_train, x_test_transformed, y_test, self.n_estimators, self.lifetime))
        
        forest = self.forests[self.sample_range[-1]][0]
        x_train_transformed = forest.transform(x_train)
        x_test_transformed = forest.transform(x_test)
        self.evaluation_results.append(
            evaluate_all_lifetimes(x_train_transformed, y_train, x_test_transformed, y_test, self.n_estimators, self.lifetime))
        
        plt.plot(self.evaluation_results[0][0]['times'], self.evaluation_results[0][0]['mse'], label = 'no transformation')
        plt.plot(self.evaluation_results[1][0]['times'], self.evaluation_results[1][0]['mse'], label = 'transform by true H')
        plt.plot(self.evaluation_results[2][0]['times'], self.evaluation_results[2][0]['mse'], label = 'transform by estimated H')
        # add labels 
        plt.xlabel('Lifetime', fontsize=15)
        plt.ylabel('MSE', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        # Change the font size of the legend
        plt.legend(fontsize=12)
        y_formatter = ScalarFormatter(useOffset=False)
        y_formatter.set_scientific(True)
        y_formatter.set_powerlimits((-1,1)) # Optional: adjust the range for using scientific notation

        # Apply the formatter to the y-axis
        plt.gca().yaxis.set_major_formatter(y_formatter)

    def reiterate(self):
        for n_sim in self.sample_range:
            for trial in range(self.tries):
                self.forests[n_sim][trial].reiterate()
        plot_dist(self.forests, self.true_H, self.sample_range, self.active, self.tries)

    def evaluation_comparison_reiterate(self):
        x_train = self.x_train
        x_test = self.x_test
        y_train = self.y_train
        y_test = self.y_test
        forest = self.forests[self.sample_range[-1]][0]
        x_train_transformed = forest.transform(x_train)
        x_test_transformed = forest.transform(x_test)
        self.evaluation_results.append(
            evaluate_all_lifetimes(x_train_transformed, y_train, x_test_transformed, y_test, self.n_estimators, self.lifetime))

        plt.plot(self.evaluation_results[0][0]['times'], self.evaluation_results[0][0]['mse'], label = 'no transformation')
        plt.plot(self.evaluation_results[1][0]['times'], self.evaluation_results[1][0]['mse'], label = 'transform by true H')
        plt.plot(self.evaluation_results[2][0]['times'], self.evaluation_results[2][0]['mse'], label = 'transform by estimated H - 1st')
        plt.plot(self.evaluation_results[3][0]['times'], self.evaluation_results[3][0]['mse'], label = 'transform by estimated H - 2nd')
        # add labels

        plt.xlabel('Lifetime', fontsize=15)
        plt.ylabel('MSE', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=12)
        y_formatter = ScalarFormatter(useOffset=False)
        y_formatter.set_scientific(True)
        y_formatter.set_powerlimits((-1,1)) # Optional: adjust the range for using scientific notation

        # Apply the formatter to the y-axis
        plt.gca().yaxis.set_major_formatter(y_formatter)