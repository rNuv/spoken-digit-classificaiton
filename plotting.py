import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde
from dataparser import get_data, extract_mfccs

def plot_mfccs_vs_analysis_window(tokens, plot_single=False):
    # grab relevant datapoints
    data_points = {}

    # plotting only the first occurrence of each digit
    for token in tokens:
        if token.token_index == 1:
            data_points[token.digit] = token
    
    # plot the first 3 MFCC coefficients
    if not plot_single:
        # one subplot for each digit
        fig, axes = plt.subplots(10, 1, figsize=(10, 15), sharex=True)
        fig.tight_layout(pad=3.0)
        for digit in data_points:
            token = data_points[digit]
            for i in range(3):
                axes[digit].plot(token.mfccs[:, i], label=f'MFCC {i+1}')
            axes[digit].set_title(f'Digit: {token.digit}')
            axes[digit].legend()
            axes[digit].set_xlabel('Analysis Window (Frame Number)')
            axes[digit].set_ylabel('Coefficient Value')
        # adjust the layout
        plt.subplots_adjust(hspace=0.5)
        plt.show()

    # plot all mfccs for single occuurence
    if plot_single:
        # create a plot for all 13 MFCCs for a single occurence
        digit = 7
        token = data_points[digit]
        plt.figure(figsize=(12, 7))
        for i in range(13):  # plot each of the 13 MFCC coefficients
            plt.plot(token.mfccs[:, i], label=f'MFCC {i+1}')
        plt.title(f'MFCCs vs. Analysis Window Index for Single Digit Sample (Digit: 7)')
        plt.legend()
        plt.xlabel('Analysis Window (Frame Number)')
        plt.ylabel('Coefficient Value')
        plt.show()
        # plt.savefig('images/mfccs_vs_window.png')

def plot_mfcc_scatter(tokens):
    # create a figure and a grid of subplots with 3 columns
    fig, axes = plt.subplots(10, 3, figsize=(15, 40))  # Adjust figsize as needed
    
    # plotting only the first occurrence of each digit
    for token in tokens:
        if token.token_index == 1:  # only plot the first token for each digit
            # MFCC1 vs MFCC2
            axes[token.digit, 0].scatter(token.mfccs[:, 0], token.mfccs[:, 1], label=f'Digit {token.digit}: MFCC1 vs MFCC2')
            axes[token.digit, 0].set_xlabel('MFCC1')
            axes[token.digit, 0].set_ylabel('MFCC2')
            axes[token.digit, 0].legend()

            # MFCC1 vs MFCC3
            axes[token.digit, 1].scatter(token.mfccs[:, 0], token.mfccs[:, 2], label=f'Digit {token.digit}: MFCC1 vs MFCC3')
            axes[token.digit, 1].set_xlabel('MFCC1')
            axes[token.digit, 1].set_ylabel('MFCC3')
            axes[token.digit, 1].legend()

            # MFCC2 vs MFCC3
            axes[token.digit, 2].scatter(token.mfccs[:, 1], token.mfccs[:, 2], label=f'Digit {token.digit}: MFCC2 vs MFCC3')
            axes[token.digit, 2].set_xlabel('MFCC2')
            axes[token.digit, 2].set_ylabel('MFCC3')
            axes[token.digit, 2].legend()
    
    plt.tight_layout()
    plt.show()

def plot_clusters_2D_kmeans(training_data, gmm, cluster_labels, digit, show_mean_and_cov=True):
    mfcc_frames = extract_mfccs(training_data, digit)
    # mfcc_frames = mfcc_frames[:, :2]
    n_clusters = max(list(set(cluster_labels))) + 1
    fig, ax = plt.subplots(1, 1)
    scatter = ax.scatter(mfcc_frames[:, 0], mfcc_frames[:, 1], c=cluster_labels)
    ax.set_title('Clusters for Digit ' + str(digit) + ": Diagonal Covariance")
    ax.set_xlabel('MFCC 1')
    ax.set_ylabel('MFCC 2')

    lp = lambda i: ax.plot([], [], color=scatter.cmap(scatter.norm(i)), ms=np.sqrt(20), mec="none",
                           label="Cluster {:g}".format(i), ls="", marker="o")[0]
    handles = [lp(i) for i in np.unique(cluster_labels)]
    ax.legend(handles=handles, loc='best')

    if show_mean_and_cov:
        x, y = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100), np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 100)
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T

        for i in range(n_clusters):
            cluster_frames = mfcc_frames[cluster_labels == i]
            cluster_mean = gmm.means_[i]
            cluster_cov = gmm.covariances_[i]

            if show_mean_and_cov:
                ax.plot(cluster_mean[0], cluster_mean[1], 'o', label=f'Mean of Cluster {i}')
                N = 200
                X = np.linspace(np.min(cluster_frames[:, 0].flatten())-0.5, np.max(cluster_frames[:, 0].flatten())+0.5, N)
                Y = np.linspace(np.min(cluster_frames[:, 1].flatten())-0.5, np.max(cluster_frames[:, 1].flatten())+0.5, N)
                X, Y = np.meshgrid(X, Y)
                pos = np.dstack((X, Y))
                rv = multivariate_normal(cluster_mean, cluster_cov)
                Z = rv.pdf(pos)
                ax.contour(X, Y, Z)

    # plt.show()
    plt.savefig('images/diag.png')

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def plot_clusters_2D_gmm(training_data, gmm, digit):
   # Extract the first two MFCCs
    mfcc_frames = extract_mfccs(training_data, digit)[:, :2]
    
    # Predict the cluster for each sample
    responsibilities = gmm.predict_proba(mfcc_frames)

    # Define primary colors for RGB
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])  # Red, Green, Blue, Yellow

    # Blend colors based on responsibilities
    rgb = np.dot(responsibilities, colors)

    # Plotting
    fig, ax = plt.subplots(1, 1)
    scatter = ax.scatter(mfcc_frames[:, 0], mfcc_frames[:, 1], c=rgb)
    ax.set_title(f'Clusters for Digit {digit}')
    ax.set_xlabel('MFCC 1')
    ax.set_ylabel('MFCC 2')

    # Plot GMM means and covariance contours
    for i, (mean, covar) in enumerate(zip(gmm.means_, gmm.covariances_)):
        ax.plot(mean[0], mean[1], 'o', label=f'Mean of Cluster {i}')
        # Create grid for contour
        x, y = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100), np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 100)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))
        rv = multivariate_normal(mean, covar)
        Z = rv.pdf(pos)
        ax.contour(X, Y, Z)

    # plt.show()
    plt.savefig('images/emclusters3.png')

def plot_variance(tokens):
    mfcc_frames = np.vstack([token.mfccs for token in tokens])
    
    # Calculate the variance of each MFCC across all frames
    mfcc_variances = np.var(mfcc_frames, axis=0)
    
    # Plotting the variances as a bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(mfcc_variances) + 1), mfcc_variances, color='skyblue')
    plt.xlabel('MFCC Components')
    plt.ylabel('Variance')
    plt.title('Variance of MFCC Components')
    plt.xticks(range(1, len(mfcc_variances) + 1))
    # plt.show()
    plt.savefig('images/variance.png')
    

def segregate_tokens_by_digit(all_tokens):
    """Segregates the SpeechToken objects by digit."""
    tokens_per_digit = {digit: [] for digit in range(10)}
    for token in all_tokens:
        tokens_per_digit[token.digit].append(token)
    return tokens_per_digit

def compute_likelihoods_under_gmm(tokens, gmm):
    """Computes the likelihood of each token's MFCC data under the provided GMM."""
    likelihoods = [gmm.score_samples(token.mfccs) for token in tokens]
    return np.concatenate(likelihoods)

def visualize_likelihoods_pdf(tokens, gmm):
    """Visualizes the pdf of likelihoods for each digit."""
    plt.figure(figsize=(10, 5))

    tokens_per_digit = segregate_tokens_by_digit(tokens)

    for digit in range(10):
        # Compute likelihoods of this digit's utterances under the GMM
        likelihoods = compute_likelihoods_under_gmm(tokens_per_digit[digit], gmm)
        
        # Apply Kernel Density Estimation
        kde = gaussian_kde(likelihoods)
        x_range = np.linspace(min(likelihoods), max(likelihoods), 1000)
        plt.plot(x_range, kde(x_range), label=f'Digit {digit}')

    plt.title('PDF of Likelihoods for Each Digit Under the GMM of Digit 0')
    plt.xlabel('Log-Likelihood')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()

def plot_gender(tokens):
     # grab relevant datapoints
    male_data_points = {}
    female_data_points = {}

    # plotting only the first occurrence of each digit
    for token in tokens:
        if token.token_index == 1 and token.speaker_gender == "male":
            male_data_points[token.digit] = token
        if token.token_index == 331 and token.speaker_gender == "female":
            female_data_points[token.digit] = token
    
    digit = 7
    token = female_data_points[digit]
    plt.figure(figsize=(12, 7))
    for i in range(13):  # plot each of the 13 MFCC coefficients
        plt.plot(token.mfccs[:, i], label=f'MFCC {i+1}')
    plt.title(f'MFCCs vs. Analysis Window Index for Single Digit Female Sample (Digit: 7)')
    plt.legend()
    plt.xlabel('Analysis Window (Frame Number)')
    plt.ylabel('Coefficient Value')
    plt.savefig('images/female.png')
    # plt.show()
    
