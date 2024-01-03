import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import pandas as pd

class PCA:
    """
    Principal Component Analysis (PCA) Class

    Attributes
    ----------
    n_components : int
        Number of principal components to retain.
    feature_names : list
        Names of the raw features in the dataset.
    components : np.ndarray
        Principal components (eigenvectors of the covariance matrix).
    mean_vector : np.ndarray
        Mean of the features.
    explained_variance_ : np.ndarray
        Variance explained by each of the selected components.
    explained_variance_ratio_ : np.ndarray
        Percentage of variance explained by each of the selected components.

    References
    ----------
    * MLiS Notes: Garrahan JP, Gillman E, Mair JF. 2023. Machine Learning in Science Part 1 (PHYS4035) Lecture Notes.
    * Github: https://github.com/erdogant/pca
    * Textbook: Springer Series in Statistics 2002. Choosing a Subset of Principal Components or Variables. In:
    Principal Component Analysis. Springer, New York, NY. https://doi.org/10.1007/0-387-22440-8_6
    """

    def __init__(self, n_components):
        """
        Initialise the PCA instance with the number of components desired to retain.

        Parameters
        ----------
        n_components : int
            Number of principal components to retain.
        """
        self.feature_names = []
        self.n_components = n_components
        self.components = None
        self.mean_vector = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        self.mean_vector = np.mean(X, axis=0)
        X = X - self.mean_vector

        covariance_matrix = np.cov(X, rowvar=False)
        eigenvalues, eigenvectors = eigh(covariance_matrix)
        pos = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[pos]
        eigenvectors = eigenvectors[:, pos]

        if self.n_components is not None:
            self.components = eigenvectors[:, :self.n_components]
        else:
            self.components = eigenvectors

        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio()

        # Initialise feature names list
        self.feature_names = X.columns.tolist()


    def transform(self, X):
        X = X - self.mean_vector
        return np.dot(X, self.components)

    def explained_variance_ratio(self):
        variance_sum = sum(self.explained_variance_)
        self.explained_variance_ratio_ = self.explained_variance_ / variance_sum

    def plot_explained_variance(self, threshold_criteria, figsize=(6,6)):
        if self.explained_variance_ is None or self.explained_variance_ratio_ is None:
            raise ValueError("The PCA model must be fitted before plotting the explained variance.")

        cumulative_explained_variance = np.cumsum(self.explained_variance_ratio_)
        components_required = np.argmax(cumulative_explained_variance >= threshold_criteria) + 1

        # Prepare the figure
        xtick_idx = np.arange(1, len(self.explained_variance_) + 1)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(xtick_idx, cumulative_explained_variance[:self.n_components], 'o-', color='#0BEF2E', linewidth=1, label='Cumulative explained variance')
        ax.set_xticks(xtick_idx)
        ax.bar(xtick_idx, self.explained_variance_ratio_[:self.n_components], color='#3182bd', alpha=0.8, label='Individual explained variance')

        ax.axhline(y=threshold_criteria, color='red', linestyle='-', linewidth=0.5)
        ax.axvline(x=components_required, color='red', linestyle='-', linewidth=0.5)

        ax.set_ylabel('Explained Variance Ratio', color='black')
        ax.set_xlabel('Principal Component', color='black')
        ax.set_ylim([0, 1.05])
        ax.set_xlim([0, self.n_components + 1])

        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)

        title = f'{components_required} Principal Components Explain {cumulative_explained_variance[components_required - 1]:.2%} of the Variance Relative to the Chosen Components'
        ax.set_title(title)
        ax.set_facecolor("white")
        ax.grid(True, color='#A9A9A9', linewidth=0.5)
        ax.legend(loc='best')
        plt.show()

        return fig, ax

    def get_loadings(self, PC=(0,1)):
        if self.components is None:
            raise ValueError("The PCA model must be fitted before retrieving loadings.")

        for component in PC:
            if component < 0 or component >= self.n_components:
                raise ValueError(f"Invalid component: {component + 1} ({component}). Components must be between 1 and {self.n_components} (0,{self.n_components-1})")

        component_names = [f'PC{component+1}' for component in PC]
        loadings = pd.DataFrame(self.components[:, PC], columns=component_names, index=self.feature_names)

        return loadings

    def get_projected_data(self, X, PC=(0,1)):
        if self.components is None:
            raise ValueError("The PCA model must be fitted before retrieving projected data.")

        for component in PC:
            if component < 0 or component >= self.n_components:
                raise ValueError(f"Invalid component: {component + 1} ({component}). Components must be between 1 and {self.n_components} (0,{self.n_components-1})")

        # Projecting X
        projected_data = self.transform(X)
        component_names = [f'PC{component+1}' for component in PC]
        projected_data_df = pd.DataFrame(projected_data[:, PC], columns=component_names)

        return projected_data_df

    def biplot_2D(self, X, y, PC=(0,1), figsize=(9,5)):
        if self.components is None:
            raise ValueError("The PCA model must be fitted before creating the biplot.")

        for component in PC:
            if component < 0 or component >= self.n_components:
                raise ValueError(f"Invalid component: {component + 1} ({component}). Components must be between 1 and {self.n_components} (0,{self.n_components-1})")

        projected_data_df = self.get_projected_data(X)
        loadings = self.get_loadings(PC)

        # Scaling the PCs
        scales = [1.0 / (projected_data_df[col].max() - projected_data_df[col].min()) for col in projected_data_df.columns]

        # Prepare Plot
        fig, ax = plt.subplots(figsize=figsize)
        for i, feature in enumerate(self.feature_names):
            ax.arrow(0, 0, loadings.iloc[i, 0], loadings.iloc[i, 1], fc='red', ec='red', head_width=0.005, head_length=0.01, width=0.000005)
            ax.text(loadings.iloc[i, 0] * 1.05, loadings.iloc[i, 1] * 1.05, feature, fontsize=11)

        target_colors = np.where(y == 2, 2, 4)
        scatter = ax.scatter(projected_data_df.iloc[:,0] * scales[0],
                             projected_data_df.iloc[:, 1] * scales[1],
                             c=target_colors,
                             cmap='viridis')

        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.5)

        ax.set_xlabel(f'PC{PC[0]+1}', fontsize=15)
        ax.set_ylabel(f'PC{PC[1]+1}', fontsize=15)
        ax.set_title('Biplot', fontsize=15)
        ax.set_facecolor("white")
        ax.legend(*scatter.legend_elements(), loc="lower left", title="Groups")
        plt.show()

    def biplot_3D(self, X, y, PC=(0,1,2), width=700, height=600):
        if self.components is None:
            raise ValueError("The PCA model must be fitted before creating the biplot.")

        for component in PC:
            if component < 0 or component >= self.n_components:
                raise ValueError(f"Invalid component: {component + 1} ({component}). Components must be between 1 and {self.n_components} (0,{self.n_components-1})")

        projected_data_df = self.get_projected_data(X, PC)
        loadings = self.get_loadings(PC)

        # Scaling the PCs
        scales = [1.0 / (projected_data_df[col].max() - projected_data_df[col].min()) for col in projected_data_df.columns]

        y_values = y['Class'].values
        target_colors = np.where(y_values == 2, 2, 4)
        trace = go.Scatter3d(
            x=projected_data_df.iloc[:, 0] * scales[0],
            y=projected_data_df.iloc[:, 1] * scales[1],
            z=projected_data_df.iloc[:, 2] * scales[2],
            mode='markers',
            marker=dict(
                size=5,
                color=target_colors,
                opacity=0.8
            )
        )
        arrows = []
        texts = []
        for i, feature in enumerate(self.feature_names):
            arrows.append(
                go.Scatter3d(x=[0, loadings.iloc[i, 0]], y=[0, loadings.iloc[i, 1]], z=[0, loadings.iloc[i, 2]],
                    mode='lines',
                    line=dict(color='red'),
                )
            )
            texts.append(
                go.Scatter3d(x=[loadings.iloc[i, 0]*1.1], y=[loadings.iloc[i, 1]*1.1], z=[loadings.iloc[i, 2]*1.1],
                             mode='text',
                             text=[feature],
                             textfont=dict(color="black", size=10)
                )
            )
        data = [trace] + arrows + texts
        layout = go.Layout(
            title='3D Biplot',
            width=width,
            height=height,
            scene=dict(
                xaxis=dict(title=f'PC{PC[0]+1}'),
                yaxis=dict(title=f'PC{PC[1]+1}'),
                zaxis=dict(title=f'PC{PC[2]+1}')
            )
        )
        fig = go.Figure(data=data, layout=layout)
        iplot(fig)