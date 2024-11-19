from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from dataparser import get_data, extract_mfccs, extract_mfccs_by_gender
from plotting import plot_clusters_2D_kmeans, plot_clusters_2D_gmm, visualize_likelihoods_pdf

class DigitModel:
    def __init__(self, training_data, testing_data):
        self.training_data = training_data
        self.testing_data = testing_data
        self.phoneme_count = {0: 4, 1: 5, 2: 5, 3: 6, 4: 6, 5: 5, 6: 4, 7: 6, 8: 6, 9: 5}
        self.gmm_list = []        
        self.test_predictions = []
        self.gmm_male = []
        self.gmm_female = []

    def fit_gmm_to_digit_kmeans(self, data, n_clusters=4):
        kmeans = KMeans(n_clusters=n_clusters).fit(data)
        cluster_labels = kmeans.labels_

        cluster_weights = []
        cluster_means = []
        cluster_covariances = []
        cluster_precisions = []

        for i in range(n_clusters):
            cluster_frames = data[cluster_labels == i]
            cluster_mean = np.mean(cluster_frames, axis=0, dtype=np.float64)

            # Calculate covariance
            cluster_cov = np.cov(cluster_frames, rowvar=False)

            # demeaned_cluster_frames = cluster_frames - cluster_mean
            # var = np.var(demeaned_cluster_frames.flatten())
            # cluster_cov = np.identity(cluster_mean.shape[0]) * var

            # cluster_cov = np.zeros((cluster_mean.shape[0], cluster_mean.shape[0]))
            # for j in range(cluster_mean.shape[0]):
            #     cluster_cov[j, j] = sum((val - cluster_mean[j]) ** 2 for val in cluster_frames[:, j]) / cluster_frames.shape[0]

            # Calculate weight of the cluster
            cluster_weight = len(cluster_frames) / len(data)

            cluster_weights.append(cluster_weight)
            cluster_means.append(cluster_mean)
            cluster_covariances.append(cluster_cov)
            cluster_precisions.append(np.linalg.inv(cluster_cov))


        cluster_weights, cluster_means, cluster_covariances, cluster_precisions = np.asarray(cluster_weights), \
                                                                                    np.asarray(cluster_means), \
                                                                                    np.asarray(cluster_covariances), \
                                                                                    np.asarray(cluster_precisions)

        # Create GMM by manually setting all of its parameters
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full')
        gmm.weights_ = cluster_weights
        gmm.covariances_ = cluster_covariances
        gmm.precisions_ = cluster_precisions
        gmm.means_ = cluster_means
        gmm.precisions_cholesky_ = _compute_precision_cholesky(cluster_covariances, "full")
        
        return gmm, cluster_labels
    
    def fit_gmm_to_digit_em(self, data, n_clusters=4):
        gmm = GaussianMixture(n_components=n_clusters, covariance_type="diag", random_state=2).fit(data)
        return gmm
    
    def train(self, gender=False):
        if not gender:
            for digit in range(10):
                data = extract_mfccs(self.training_data, digit)
                gmm = self.fit_gmm_to_digit_em(data, n_clusters=self.phoneme_count[digit])
                self.gmm_list.append(gmm)
        else:
            for digit in range(10):
                male, female = extract_mfccs_by_gender(self.training_data, digit)
                male = male[:, :12]
                female = female[:, :12]
                gmm_m = self.fit_gmm_to_digit_em(male, n_clusters=self.phoneme_count[digit])
                gmm_f = self.fit_gmm_to_digit_em(female, n_clusters=self.phoneme_count[digit])
                self.gmm_male.append(gmm_m)
                self.gmm_female.append(gmm_f)

    def predict(self):
        gmm_source = []
        for token in self.testing_data:
            # Initialize a variable to store the highest log likelihood and its corresponding digit
            max_log_likelihood = float('-inf')
            predicted_digit = None

            if len(self.gmm_list) != 0:
                gmm_source = self.gmm_list
            elif token.speaker_gender == "male":
                gmm_source = self.gmm_male
            elif token.speaker_gender == "female":
                gmm_source = self.gmm_female
            
            # Loop through each digit's GMM model
            for digit, gmm in enumerate(gmm_source):
                # Calculate the log likelihood of the token's MFCCs for the current GMM
                input = token.mfccs[:, :12]
                log_likelihood = gmm.score_samples(input)

                # Sum the log likelihoods as score_samples returns an array
                total_log_likelihood = np.sum(log_likelihood)

                # Check if this is the highest log likelihood so far
                if total_log_likelihood > max_log_likelihood:
                    max_log_likelihood = total_log_likelihood
                    predicted_digit = digit

            # Append the predicted digit to the test_predictions list
            self.test_predictions.append(predicted_digit)

    def evaluate(self):
         # Generate predictions
        self.predict()

        # Count the number of correct predictions
        correct_predictions = 0
        total_predictions = len(self.testing_data)

        for i, token in enumerate(self.testing_data):
            if token.digit == self.test_predictions[i]:
                correct_predictions += 1

        # Calculate the accuracy
        accuracy = correct_predictions / total_predictions
        print("Testing Accuracy: " + str(accuracy))

        return accuracy
    
    def plot_confusion_matrix(self):
        # Generate predictions if not already done
        if not self.test_predictions:
            self.predict()

        # Extract actual and predicted labels
        actual_labels = [token.digit for token in self.testing_data]
        predicted_labels = self.test_predictions

        # Compute the confusion matrix
        cm = confusion_matrix(actual_labels, predicted_labels)

        # Plotting the confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        # plt.show()
        plt.savefig('images/em_results.png')
        

if __name__ == "__main__":
    digit = 7
    training_data = get_data()
    testing_data = get_data(isTesting=True)
    model = DigitModel(training_data, testing_data)
    model.train(gender=True)
    model.evaluate()
    model.plot_confusion_matrix()
