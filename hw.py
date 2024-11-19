import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np

def estimate_gmm_kmeans_parameters(data, n_components):
    """
    Estimates GMM parameters for the given data using K-means clustering.
    
    :param data: MFCC data for a specific digit.
    :param n_components: Number of components (clusters) for GMM.
    :return: GMM model.
    """
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_components, random_state=0).fit(data)

    # Initialize GMM parameters
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    gmm.means_ = kmeans.cluster_centers_
    gmm.fit(data)

    return gmm

def calculate_log_likelihoods(data, gmm):
    return gmm.score_samples(data)

blocks = read_data('spoken+arabic+digit/Train_Arabic_Digit.txt')
tokens = create_tokens(blocks)
# plot_mfccs(tokens)

# group tokens by their digit
# tokens_by_digit = {i: [] for i in range(10)}
# for token in tokens:
#     tokens_by_digit[token.digit].append(token)

# visualize gmm for each digit
# for digit, tokens_for_digit in tokens_by_digit.items():
#     visualize_kmeans_for_digit(digit, tokens_for_digit)

# get all of the speech tokens for digit 7
# mfcc_data_digit_7 = [token.mfccs for token in tokens if token.digit == 7]
# mfcc_data_digit_7_flat = np.vstack(mfcc_data_digit_7)

# estimate GMM parameters
# n_components = 4
# gmm_model = estimate_gmm_kmeans_parameters(mfcc_data_digit_7_flat, n_components)

# print parameters
# print("GMM Parameters:")
# print("Mixture Probabilities (pi):", gmm_model.weights_)
# print("Means (mu):", gmm_model.means_)
# print("Covariance Matrices (sigma):", gmm_model.covariances_)

# fig, axs = plt.subplots(2, 5, figsize=(20, 8))
# axs = axs.flatten()

# for digit in range(10):
#     # extract MFCC data for the current digit
#     mfcc_data_digit = [token.mfccs for token in tokens if token.digit == digit]
#     mfcc_data_digit_flat = np.vstack(mfcc_data_digit)

#     # calculate log-likelihoods
#     log_likelihoods = calculate_log_likelihoods(mfcc_data_digit_flat, gmm_model)

#     # Kernel Density Estimation for log-likelihoods
#     kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(log_likelihoods.reshape(-1, 1))
#     x_d = np.linspace(log_likelihoods.min(), log_likelihoods.max(), 1000)
#     log_dens = kde.score_samples(x_d.reshape(-1, 1))

#     # plotting
#     axs[digit].fill_between(x_d, np.exp(log_dens), alpha=0.5)
#     axs[digit].set_title(f'Digit {digit}')
#     axs[digit].set_xlabel('Log-Likelihood')
#     axs[digit].set_ylabel('Density')

# plt.tight_layout()
# plt.show()

def get_mfcc_data_for_digit(digit, tokens):
    # Extracts and combines MFCC data for a specific digit from all tokens
    return np.concatenate([token.mfccs for token in tokens if token.digit == digit])

def plot_gmm_contours(digit, mfcc_data, n_components=3):
    # Fit a Gaussian Mixture Model to the MFCC data
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(mfcc_data)

    # Get the mean of the features (excluding the two features for the contour plot)
    means = gmm.means_.mean(axis=0)

    # Create a meshgrid for plotting
    x = np.linspace(np.min(mfcc_data[:, 0]), np.max(mfcc_data[:, 0]), 100)
    y = np.linspace(np.min(mfcc_data[:, 1]), np.max(mfcc_data[:, 1]), 100)
    X, Y = np.meshgrid(x, y)

    # Create contour plots for each pair of MFCCs
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'GMM Contours for Digit {digit}')

    # Define the pairs of MFCCs for contour plotting
    mfcc_pairs = [(0, 1), (0, 2), (1, 2)]

    for i, (mfcc_x, mfcc_y) in enumerate(mfcc_pairs):
        # Create a full data array for GMM prediction
        full_data = np.zeros((X.ravel().shape[0], 13))
        full_data[:, mfcc_x] = X.ravel()
        full_data[:, mfcc_y] = Y.ravel()

        # Fill in the other MFCCs with their mean values
        for j in range(13):
            if j != mfcc_x and j != mfcc_y:
                full_data[:, j] = means[j]

        # Predict the log-likelihood for each point
        Z = -gmm.score_samples(full_data)
        Z = Z.reshape(X.shape)

        # Plot the contours
        axes[i].contour(X, Y, Z)
        axes[i].set_title(f'MFCC{mfcc_x+1} vs MFCC{mfcc_y+1}')
        axes[i].set_xlabel(f'MFCC{mfcc_x+1}')
        axes[i].set_ylabel(f'MFCC{mfcc_y+1}')

    plt.tight_layout()
    plt.show()

# Run the analysis for each digit
for digit in range(10):
    mfcc_data = get_mfcc_data_for_digit(digit, tokens)
    plot_gmm_contours(digit, mfcc_data)
