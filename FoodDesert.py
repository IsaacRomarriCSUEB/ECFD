# Data manipulations
import pandas as pd
import numpy as np
# PCA and Kmeans
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns

class FoodDesertsData:
    def __init__(self, file_path="/content/drive/MyDrive/PCAandKMeansLab/Food_Deserts_Data.csv"):
        """
        Initializes the FoodDesertsData class by loading and processing the data.

        Args:
            file_path (str): The path to the Food_Deserts_Data.csv file.
        """
        self.file_path = file_path
        self.df = self._load_and_process_data()
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(self.df)
        self.pca = None
        self.pca_results = None
        self.cluster_labels = None
        self.cluster_centers = None
        self.inertia = None
        self.state_df = None

    def _load_and_process_data(self):
        """
        Loads the data from the CSV file and performs initial processing steps.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        food_deserts_df = pd.read_csv(self.file_path)
        state_df = food_deserts_df[food_deserts_df["State"] == "New Mexico"].copy(deep=True)
        self.state_df = state_df
        features_to_keep = [
            "PovertyRate", "MedianFamilyIncome", "lapophalfshare", "lapop1share",
            "TractLOWI", "TractKids", "TractSeniors", "TractWhite", "TractBlack",
            "TractAsian", "TractNHOPI", "TractAIAN", "TractOMultir", "TractHispanic",
            "TractHUNV", "TractSNAP", "Pop2010", "OHU2010"
        ]
        state_df = state_df[features_to_keep].copy(deep=True)

        features_to_convert = [
            "TractLOWI", "TractKids", "TractSeniors", "TractWhite",
            "TractBlack", "TractAsian", "TractNHOPI", "TractAIAN",
            "TractOMultir", "TractHispanic", "TractHUNV", "TractSNAP"
        ]

        for feature in features_to_convert:
            if feature == "TractHUNV" or feature == "TractSNAP":
                share = state_df[feature] / state_df["OHU2010"]
            else:
                share = state_df[feature] / state_df["Pop2010"]
            state_df[feature] = share
            rename = feature + "Share"
            state_df.rename(columns={feature: rename}, inplace=True)

        state_df.drop(columns=["OHU2010", "Pop2010"], inplace=True)

        for column in state_df.columns:
            column_mean = state_df[column].mean()
            state_df[column] = state_df[column].fillna(column_mean)

        return state_df

    def perform_pca(self, n_components=None):
        """
        Performs Principal Component Analysis on the scaled data.

        Args:
            n_components (int or None): The number of components to keep.
                                        If None, all components are kept.
        """
        self.pca = PCA(n_components=n_components)
        self.pca.fit(self.scaled_data)
        self.pca_results = self.pca.transform(self.scaled_data)

    def performed_kmeans(self, n_clusters=None, init="random"):
        """
        Performs k-means clustering on the PCA transformed data.
        """
        if n_clusters is None:
            n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, init=init)
        kmeans.fit(self.pca_results)
        self.cluster_labels = kmeans.labels_
        self.cluster_centers = kmeans.cluster_centers_
        self.inertia = kmeans.inertia_
    
    def get_df(self):
        """
        Returns the processed DataFrame.
        """
        return self.df.copy(deep=True)

    def get_scaled_data(self):
        """
        Returns the scaled data (NumPy array).
        """
        return self.scaled_data

    def get_pca_results(self):
        """
        Returns the PCA transformed data (NumPy array).
        Returns None if PCA has not been performed.
        """
        return self.pca_results

    def get_pca_components(self):
        """
        Returns the principal components.
        Returns None if PCA has not been performed.
        """
        if self.pca:
            return self.pca.components_
        else:
            return None

    def get_pca_explained_variance(self):
        """
        Returns the explained variance of each principal component.
        Returns None if PCA has not been performed.
        """
        if self.pca:
            return self.pca.explained_variance_
        else:
            return None

    def get_pca_explained_variance_ratio(self):
        """
        Returns the ratio of explained variance of each principal component.
        Returns None if PCA has not been performed.
        """
        if self.pca:
            return self.pca.explained_variance_ratio_
        else:
            return None
    
    def get_cluster_labels(self):
        """
        Returns the cluster labels.
        Returns None if k-means clustering has not been performed.
        """
        return self.cluster_labels
    
    def get_cluster_centers(self):
        """
        Returns the cluster centers.
        Returns None if k-means clustering has not been performed.
        """
        return self.cluster_centers
