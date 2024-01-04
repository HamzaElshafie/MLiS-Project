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
    components : numpy.ndarray
        Principal components (eigenvectors of the covariance matrix).
    mean_vector : numpy.ndarray
        Mean of the features.
    explained_variance_ : np.ndarray
        Variance explained by each of the selected components.
    explained_variance_ratio_ : numpy.ndarray
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
        Initialise the PCA object with the number of components desired to retain.

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
        """
        Fit the PCA model to the data

        The function computes the principal components of the provided design matrix. The function is normally fitted to
        the training set after splitting. It computes the mean vector, covariance matrix,eigenvalues, and eigenvectors
        of the covariance matrix. The components (eigenvectors) are then arranged in descending order by their associated
        eigenvalues. Only the top ``n_components`` principal components are kept if ``n_components`` is specified.

        Parameters
        ----------
        X : DataFrame
            The design matrix with the features.  Each column represents a feature, and each row represents a sample.

        Notes
        -----
        The method assumes that the input data 'X' is a pandas DataFrame with numerical values. Before using this
        method, missing values in 'X' should be addressed. The method centers the data by subtracting the mean vector
        from each sample but does not scale the features. Therefore, it's advisable to standardise the data before
        performing PCA if the features are on different scales.

        Examples
        -----
        >>> import pandas as pd
        >>> # Load example data
        >>> from sklearn.datasets import load_iris
        >>> data = load_iris()
        >>> X = pd.DataFrame(data.data, columns=data.feature_names)
        >>> pca = PCA(n_components=3)
        >>> pca.fit(X)
        """
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
        """
        Transform the data to the new principal components space.

        The function projects the data onto the principal components computed in the ``fit`` method. This transformation
        is achieved by centering the data again using the mean vector computed in ``fit`` and then multiplying it by the
        principal component vectors. The ``fit`` method must be called before using this method.

        Parameters
        ----------
        X : DataFrame
            The data to be transformed. It must have the same number of features as the dataset used in the ``fit``
            method.

        Returns
        -------
        numpy.ndarray
            The transformed data array, where each row is the projection of the corresponding sample onto the
            principal components.

        Notes
        -----
        This method does not check for missing values in ``X``. It is the responsibility of the user to ensure that ``X``
        is preprocessed appropriately before using this function. To appropriately apply PCA, the 'fit' method should
        be called on the training data to fit the model appropriately and then this function can be used for the
        transformation. It is crucial that the test data, however, is transformed directly using the previously fitted
        model and not fit a new model to prevent data leakage.

        Examples
        -----
        >>> import pandas as pd
        >>> # Load example data
        >>> from sklearn.datasets import load_iris
        >>> data = load_iris()
        >>> X = pd.DataFrame(data.data, columns=data.feature_names)
        >>> y = data.target
        >>> from sklearn.model_selection import train_test_split
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        >>> pca = PCA(n_components=3)
        >>> pca.fit(X_train)
        >>> X_train_transformed = pca.transform(X_train)
        >>> X_test_transformed = pca.transform(X_test)
        """

        X = X - self.mean_vector
        return np.dot(X, self.components)

    def explained_variance_ratio(self):
        """
        Calculate and store the explained variance ratio of each principal component.

        The method computes the ratio of variance that each principal component accounts for. It explains
        how much of the total variance in the data is explained by each principal component. The variance ratio
        is calculated as the variance explained by each principal component, ```self.explained_variance``` divided
        by the total variance explained by all the principal components specified by the user. The method then
        updates the ``explained_variance_ratio_`` attribute of the PCA object.

        Notes
        -----
        The 'explained_variance_ratio_' attribute will be a numpy array with the ratio of the variance explained by
        each principal component. These values are useful in understanding the contribution of each principal component
        to the total variance in the dataset, which can help us in making an informed decision on how many principal
        components to retain.
        """

        variance_sum = sum(self.explained_variance_)
        self.explained_variance_ratio_ = self.explained_variance_ / variance_sum

    def plot_explained_variance(self, threshold_criteria, figsize=(6,6)):
        """
        Plot the cumulative explained variance against the number of principal components specified.

        This function plots how much variance is explained by the principal components only specified in
        ```self.n_components```. It plots one line for the cumulative explained variance ratio among all the components
        and plots a bar chart, where the height of each bar corresponds to the variance ratio explained by each
        principal component. Furthermore, the function creates a horizontal and perpendicular line at a defined
        cumulative variance threshold., helping to determine an appropriate number of principal components to retain.

        Parameters
        ----------
        threshold_criteria : float
            Desired threshold for the cumulative explained variance.
        figsize : tuple, optional
            Size of the matplotlib figure. Defaults to (6, 6).

        Raises
        ------
        ValueError
            If the PCA model has not been fitted before calling this method.

        Examples
        -----
        >>> import pandas as pd
        >>> # Load example data
        >>> from sklearn.datasets import load_iris
        >>> data = load_iris()
        >>> X = pd.DataFrame(data.data, columns=data.feature_names)
        >>> y = data.target
        >>> from sklearn.model_selection import train_test_split
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        >>> pca = PCA(n_components=3)
        >>> pca.fit(X_train)
        >>> pca.plot_explained_variance(threshold_criteria=0.8, figsize=(9, 3.5))
        """

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


    def get_loadings(self, PC=(0,1)):
        """
        Gets the loadings for specified principal components.

        Loadings are the coefficients of the linear combination of the original raw variables from which the principal
        components are constructed. This method returns the loadings for the specified principal components.

        Parameters
        ----------
        PC : tuple, optional
            Indices of the principal components for which loadings are to be retrieved. The indices are start from zero not one.
            Defaults are set to (0, 1), i.e. the first two components.

        Returns
        -------
        pd.DataFrame
            A DataFrame where each column corresponds to a principal component and the rows are indexed by the original
            feature names.

        Raises
        ------
        ValueError
            If the PCA model has not been fitted before calling this method or if the specified principal component
            indices are invalid (outside the range of computed components).

        Examples
        --------
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> iris = load_iris()
        >>> X = pd.DataFrame(iris.data, columns=iris.feature_names)
        >>> pca = PCA(n_components=3)
        >>> pca.fit(X)
        >>> print(pca.get_loadings())
        """

        if self.components is None:
            raise ValueError("The PCA model must be fitted before retrieving loadings.")

        for component in PC:
            if component < 0 or component >= self.n_components:
                raise ValueError(f"Invalid component: {component + 1} ({component}). Components must be between 1 and {self.n_components} (0,{self.n_components-1})")

        component_names = [f'PC{component+1}' for component in PC]
        loadings = pd.DataFrame(self.components[:, PC], columns=component_names, index=self.feature_names)

        return loadings

    def get_projected_data(self, X, PC=(0,1)):
        """
        Gets the projected data onto the specified principal components.

        This function projects the data ``X`` onto the specified principal components and creates a DataFrame where the
        columns represent the principal components and the rows contain the axis of each projected sample in ``X``.

        Parameters
        ----------
        X : DataFrame
            The data to be projected. Should have the same P (features) dimensions of the data used in the ``fit`` function.
        PC : tuple, optional
            Indices of the principal components to project the data onto. The indices are start from zero not one.
            Defaults are set to (0, 1), i.e. the first two components.

        Returns
        -------
        pd.DataFrame
            A DataFrame containin the projected data.

        Raises
        ------
        ValueError
            If the PCA model has not been fitted before calling this method or if the specified principal component
            indices are invalid (outside the range of computed components).

        Examples
        --------
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> iris = load_iris()
        >>> X = pd.DataFrame(iris.data, columns=iris.feature_names)
        >>> pca = PCA(n_components=3)
        >>> pca.fit(X)
        >>> print(pca.get_projected_data(X))
        """
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
        """
        Generates a 2D biplot of the projected data.

        This method creates a 2D biplot that displays the projection of the original data ``X`` onto the specified
        principal components and the vector loadings of each feature on these components.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The data to be visualised. Should have the same P (features) dimensions of the data used in the ``fit`` function.
        y : pd.Series or np.ndarray
            The target vector with the corresponding target of each sample in ``X``. The target labels is used for color
             coding the observations in the plot.
        PC : tuple, optional
            Indices of the two principal components to plot against each other in the plot. Defaults to (0, 1).
        figsize : tuple, optional
            Size of the figure. Defaults to (9, 5).

        Returns
        -------
        None
            Does not return a value but shows the biplot.

        Raises
        ------
        ValueError
            If the PCA model has not been fitted before calling this method or if the specified component indices are
            invalid (outside the range of computed components).

        Examples
        --------
        >>> import pandas as pd
        >>> # Load example data
        >>> from sklearn.datasets import load_iris
        >>> data = load_iris()
        >>> X = pd.DataFrame(data.data, columns=data.feature_names)
        >>> y = data.target
        >>> from sklearn.model_selection import train_test_split
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        >>> pca = PCA(n_components=3)
        >>> pca.fit(X_train)
        >>> pca.biplot_2D(X_train, y_train)
        """

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
        """
        Generates a 3D interactive biplot of the projected data.

        This method creates a 3D interactive biplot that displays the projection of the original data ``X`` onto the specified
        principal components and the vector loadings of each feature on these components.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The data to be visualised. Should have the same P (features) dimensions of the data used in the ``fit`` function.
        y : pd.Series or np.ndarray
            The target vector with the corresponding target of each sample in ``X``. The target labels is used for color
             coding the observations in the plot.
        PC : tuple, optional
            Indices of the three principal components to plot against each other in the plot. Defaults to (0, 1,2). i.e
            first three components.
        width : int, optional
            Width of the plot.
        height : int, optional
            Height of the plot.

        Returns
        -------
        None
            Does not return a value but displays the biplot.

        Raises
        ------
        ValueError
            If the PCA model has not been fitted before calling this method or if the specified component indices are
            invalid (outside the range of computed components).

        Examples
        --------
        >>> import pandas as pd
        >>> # Load example data
        >>> from sklearn.datasets import load_iris
        >>> data = load_iris()
        >>> X = pd.DataFrame(data.data, columns=data.feature_names)
        >>> y = data.target
        >>> from sklearn.model_selection import train_test_split
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        >>> pca = PCA(n_components=3)
        >>> pca.fit(X_train)
        >>> pca.biplot_3D(X_train, y_train, PC=(0,1,2), width=1200, height=700)
        """
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