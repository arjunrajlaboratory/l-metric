import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from tqdm import tqdm

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
import plotly.graph_objects as go


class LMetricCalculator:
    def __init__(self, df):
        """
        Initialize the calculator with a DataFrame containing gene expression data.
        Parameters:
        df (pandas.DataFrame): DataFrame containing gene expression data
        """
        self.df = df
        self.l_metric_matrix = None
        self.thresholded_l_metric_matrix = None
        self.row_order = None
        self.col_order = None

        self.min_counts = 2

    def compute_l_metric(self, gene1, gene2, verbose=True):
        """
        Compute the L-metric between two genes.
        Parameters:
        gene1 (str): Name of the first gene
        gene2 (str): Name of the second gene
        Returns:
        tuple: (normalized_l_metric, raw_l_metric)
        """
        gene1_counts = self.df[gene1]
        gene2_counts = self.df[gene2]

        mask = (gene1_counts >= self.min_counts) | (
            gene2_counts >= self.min_counts)
        gene1_counts = gene1_counts[mask]
        gene2_counts = gene2_counts[mask]
        if verbose:
            print(
                f"Keeping {sum(mask)} cells out of {len(mask)} total cells")

        df_genes = pd.DataFrame({gene1: gene1_counts, gene2: gene2_counts})
        df_genes[f"equal_{gene2}"] = np.sum(
            df_genes[gene2]) / len(df_genes[gene2])
        df_sorted = df_genes.sort_values(
            by=gene1, ascending=False).reset_index(drop=True)
        df_sorted[f"negative_{gene2}"] = df_sorted[gene2].sort_values(
            ascending=True).values
        df_sorted[f"positive_{gene2}"] = df_sorted[gene2].sort_values(
            ascending=False).values

        cumsum_gene1 = df_sorted[gene1].cumsum().values
        cumsum_gene2 = df_sorted[gene2].cumsum().values
        cumsum_equal_gene2 = df_sorted[f"equal_{gene2}"].cumsum().values
        cumsum_negative_gene2 = df_sorted[f"negative_{gene2}"].cumsum().values
        cumsum_positive_gene2 = df_sorted[f"positive_{gene2}"].cumsum().values

        # This is stupid, I should just use the original X values
        deltaX = np.diff(cumsum_gene1)

        normalization_factor = cumsum_gene1[-1] * cumsum_gene2[-1]

        measured_area = np.sum(
            deltaX * (cumsum_gene2[:-1] + cumsum_gene2[1:]) / 2) / normalization_factor
        equal_area = np.sum(
            deltaX * (cumsum_equal_gene2[:-1] + cumsum_equal_gene2[1:]) / 2) / normalization_factor
        negative_area = np.sum(
            deltaX * (cumsum_negative_gene2[:-1] + cumsum_negative_gene2[1:]) / 2) / normalization_factor
        positive_area = np.sum(
            deltaX * (cumsum_positive_gene2[:-1] + cumsum_positive_gene2[1:]) / 2) / normalization_factor

        raw_l_metric = measured_area - equal_area

        if measured_area > equal_area:
            normalized_l_metric = (
                measured_area - equal_area) / (positive_area - equal_area)
        elif measured_area < equal_area:
            normalized_l_metric = - \
                (measured_area - equal_area) / (negative_area - equal_area)
        else:
            normalized_l_metric = 0.0

        return normalized_l_metric, raw_l_metric

    def compute_pairwise_l_metrics(self, genes=None):
        """
        Compute the full matrix of pairwise L-metrics.
        Parameters:
        genes (list, optional): List of genes to compute metrics for. If None, use all genes in the DataFrame.
        Returns:
        self: Returns the instance for method chaining
        """
        if genes is None:
            genes = self.df.columns.tolist()

        self.l_metric_matrix = pd.DataFrame(
            data=np.zeros((len(genes), len(genes))),
            index=genes,
            columns=genes
        )
        gene_pairs = list(permutations(genes, 2))
        for gene1, gene2 in tqdm(gene_pairs, desc="Computing Pairwise L-Metrics"):
            try:
                metric, _ = self.compute_l_metric(
                    gene1, gene2, verbose=False)
                self.l_metric_matrix.loc[gene1, gene2] = metric
            except Exception as e:
                print(f"Error computing L-metric for {gene1} and {gene2}: {e}")
                # 0.0  # Could do np.nan, but honestly that just makes more trouble
                self.l_metric_matrix.loc[gene1, gene2] = np.nan
        return self

    def threshold_l_metric_matrix(self, threshold=-0.5):
        self.thresholded_l_metric_matrix = self.l_metric_matrix.applymap(
            lambda x: 0 if x > threshold else 1)
        return self

    def perform_clustering(self, method='average', metric='euclidean', by_row=True, by_col=True):
        """
        Perform hierarchical clustering on the L-metric matrix.
        Parameters:
        method (str): The linkage method to use for clustering
        metric (str): The distance metric to use for clustering
        by_row (bool): If True, perform clustering by rows
        by_col (bool): If True, perform clustering by columns
        Returns:
        self: Returns the instance for method chaining
        """
        if self.l_metric_matrix is None:
            raise ValueError(
                "L-metric matrix has not been computed yet. Call compute_pairwise_l_metrics first.")

        matrix = self.l_metric_matrix.fillna(0)
        if by_row:
            dist_matrix = pdist(matrix)
            linkage_matrix = linkage(dist_matrix, method=method, metric=metric)
            dendrogram_row = dendrogram(linkage_matrix, no_plot=True)
            self.row_order = dendrogram_row['leaves']
        if by_col:
            dist_matrix = pdist(matrix.T)
            linkage_matrix = linkage(dist_matrix, method=method, metric=metric)
            dendrogram_col = dendrogram(linkage_matrix, no_plot=True)
            self.col_order = dendrogram_col['leaves']
        return self

    def plot_heatmap(self, interactive=True, cmap=None, use_ordering="row", **kwargs):
        """
        Plot the heatmap of the L-metric matrix.
        Parameters:
        interactive (bool): If True, create an interactive plotly heatmap. If False, create a static matplotlib heatmap.
        cmap (list of lists, optional): Custom color scale for interactive plot. If None, a default custom scale is used.
        use_ordering (str): If "row", use row ordering. If "col", use column ordering. If "both", use both row and column ordering.
        **kwargs: Additional keyword arguments to pass to the plotting function
        Returns:
        self: Returns the instance for method chaining
        """
        if self.l_metric_matrix is None:
            raise ValueError(
                "L-metric matrix has not been computed yet. Call compute_pairwise_l_metrics first.")
        matrix = self.l_metric_matrix

        if self.row_order is not None and self.col_order is not None:
            if use_ordering == "row":
                matrix = matrix.iloc[self.row_order, self.row_order]
            elif use_ordering == "col":
                matrix = matrix.iloc[self.col_order, self.col_order]
            elif use_ordering == "both":
                matrix = matrix.iloc[self.row_order, self.col_order]

        # Determine the maximum absolute value for symmetric color scaling
        vmax = np.abs(matrix.values).max()

        if interactive:
            if cmap is None:
                cmap = [
                    [0.0, 'blue'],
                    [0.25, 'lightblue'],
                    [0.5, 'white'],
                    [0.75, 'orange'],
                    [1.0, 'darkorange']
                ]

            heatmap = go.Heatmap(
                z=matrix.values,
                x=matrix.columns,
                y=matrix.index,
                colorscale=cmap,
                zmin=-vmax,
                zmax=vmax,
                colorbar=dict(title='L-Metric'),
                hoverongaps=False,
                reversescale=False
            )

            layout = go.Layout(
                title='Pairwise L-Metric Heatmap',
                xaxis=dict(tickangle=90),
                yaxis=dict(
                    scaleanchor="x",
                    autorange='reversed'  # This reverses the y-axis
                ),
                width=600,
                height=600,
                autosize=False
            )

            fig = go.Figure(data=[heatmap], layout=layout)

            fig.update_layout(
                xaxis_title="Genes",
                yaxis_title="Genes",
                title=dict(x=0.5),  # Center the title
            )

            fig.show()
        else:
            plt.figure(figsize=(10, 10))
            plt.imshow(matrix, cmap='RdBu_r', vmin=-vmax,
                       vmax=vmax, aspect='auto', **kwargs)
            plt.colorbar(label='L-Metric')
            plt.title('Pairwise L-Metric Heatmap')
            plt.xlabel('Genes')
            plt.ylabel('Genes')
            plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=90)
            plt.yticks(range(len(matrix.index)), matrix.index)
            plt.tight_layout()
            plt.show()
        return self

    def plot_hexbin(self, gene1, gene2):
        """
        Plot a hexbin scatter plot of two genes.

        Parameters:
        gene1 (str): Name of the first gene
        gene2 (str): Name of the second gene
        """
        gene1_counts = self.df[gene1]
        gene2_counts = self.df[gene2]

        mask = (gene1_counts >= self.min_counts) | (
            gene2_counts >= self.min_counts)
        gene1_counts_nozeros = gene1_counts[mask]
        gene2_counts_nozeros = gene2_counts[mask]

        plt.figure(figsize=(8, 6))
        plt.hexbin(gene1_counts_nozeros, gene2_counts_nozeros,
                   gridsize=50, cmap='Reds', mincnt=1)
        plt.colorbar(label='Number of Cells')
        plt.title(f"Hexbin Plot of {gene1} vs. {gene2}")
        plt.xlabel(gene1)
        plt.ylabel(gene2)
        plt.show()

    def plot_rank_orders(self, gene1, gene2):
        """
        Plot rank order plots and cumulative sum plots for two genes.

        Parameters:
        gene1 (str): Name of the first gene
        gene2 (str): Name of the second gene
        """
        gene1_counts = self.df[gene1]
        gene2_counts = self.df[gene2]

        mask = (gene1_counts >= self.min_counts) | (
            gene2_counts >= self.min_counts)
        gene1_counts = gene1_counts[mask]
        gene2_counts = gene2_counts[mask]

        df_genes = pd.DataFrame({gene1: gene1_counts, gene2: gene2_counts})
        df_genes[f"equal_{gene2}"] = np.sum(
            df_genes[gene2]) / len(df_genes[gene2])
        df_genes[f"random_{gene2}"] = np.random.permutation(
            df_genes[gene2].values)
        df_sorted = df_genes.sort_values(
            by=gene1, ascending=False).reset_index(drop=True)
        df_sorted[f"negative_{gene2}"] = df_sorted[gene2].sort_values(
            ascending=True).values
        df_sorted[f"positive_{gene2}"] = df_sorted[gene2].sort_values(
            ascending=False).values

        # Gene counts plot
        plt.figure(figsize=(10, 6))
        x_values = df_sorted.index
        plt.scatter(x_values, df_sorted[gene1],
                    color='blue', label=gene1, s=10, alpha=0.5)
        plt.scatter(x_values, df_sorted[gene2],
                    color='red', label=gene2, s=10, alpha=0.5)
        plt.scatter(x_values, df_sorted[f"random_{gene2}"],
                    color='green', label="Random", s=10, alpha=0.5)
        plt.title('Rank Order Plot')
        plt.xlabel('Index')
        plt.ylabel('Counts')
        plt.legend()
        plt.show()

        # Cumulative sum plot
        cumsum_gene1 = df_sorted[gene1].cumsum().values
        cumsum_gene2 = df_sorted[gene2].cumsum().values
        cumsum_equal_gene2 = df_sorted[f"equal_{gene2}"].cumsum().values
        cumsum_negative_gene2 = df_sorted[f"negative_{gene2}"].cumsum().values
        cumsum_positive_gene2 = df_sorted[f"positive_{gene2}"].cumsum().values

        plt.figure(figsize=(10, 6))
        plt.plot(cumsum_gene1, cumsum_gene2, color='red',
                 label=f'{gene1} vs. {gene2}')
        plt.plot(cumsum_gene1, cumsum_equal_gene2, color='green',
                 label=f'{gene1} vs. Equal {gene2}')
        plt.plot(cumsum_gene1, cumsum_negative_gene2, color='blue',
                 label=f'{gene1} vs. Negative {gene2}')
        plt.plot(cumsum_gene1, cumsum_positive_gene2, color='orange',
                 label=f'{gene1} vs. Positive {gene2}')
        plt.title(f'Cumulative Sum of {gene1} vs. {gene2}')
        plt.xlabel(f'Cumulative Sum of {gene1}')
        plt.ylabel(f'Cumulative Sum of {gene2}')
        plt.grid(True)
        plt.legend()
        plt.show()
