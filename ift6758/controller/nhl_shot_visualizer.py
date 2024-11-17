import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class NHLShotVisualizer:
    def __init__(self, dataframe):
        """
        Initialize the visualizer with a DataFrame containing shot and goal data.
        """
        self.dataframe = dataframe

    def plot_distance_histogram(self):
        """
        Plot a histogram of the number of shots (goals and non-goals) binned by distance.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=self.dataframe,
            x="distance_to_net",
            hue="is_goal",
            bins=30,
            kde=False,
            multiple="stack",
            palette="coolwarm",
            alpha=0.7
        )
        plt.title("Number of Shots Binned by Distance to Net")
        plt.xlabel("Distance to Net")
        plt.ylabel("Number of Shots")
        plt.legend(title="Is Goal", labels=[ "Goal","No Goal"])
        plt.grid(True)
        plt.show()

    def plot_angle_histogram(self):
        """
        Plot a histogram of the number of shots (goals and non-goals) binned by angle.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=self.dataframe,
            x="angle_to_net",
            hue="is_goal",
            bins=30,
            kde=False,
            multiple="stack",
            palette="viridis",
            alpha=0.7
        )
        plt.title("Number of Shots Binned by Angle to Net")
        plt.xlabel("Angle to Net (degrees)")
        plt.ylabel("Number of Shots")
        plt.legend(title="Is Goal", labels=["Goal","No Goal"])
        plt.grid(True)
        plt.show()


    def plot_2d_histogram_distance_angle(self):
        """
        Create a scatter plot of distance vs. angle with marginal histograms using seaborn.jointplot.

        Parameters:
        - dataframe: The input pandas DataFrame containing 'distance_to_net' and 'angle_to_net' columns.
        """
        sns.set(style="whitegrid")
        joint_plot = sns.jointplot(
            data=self.dataframe,
            x="distance_to_net",
            y="angle_to_net",
            kind="scatter",  # Scatter plot for the center
            marginal_kws=dict(bins=30, fill=True, color="blue")  # Histogram settings for the margins
        )
        # Add a title to the entire plot
        plt.suptitle("Scatter Plot with Marginal Histograms", fontsize=16, y=1.02)
        joint_plot.set_axis_labels("Distance to Net", "Angle to Net (degrees)", fontsize=12)
        plt.show()

    def calculate_goal_rate(self, x_column, num_bins):
        """
        Calculate goal rate as a function of a given column (distance or angle).

        Parameters:
        - x_column: The column to calculate goal rate against (e.g., 'distance_to_net' or 'angle_to_net').
        - num_bins: Number of bins to group the data.

        Returns:
        - A DataFrame with binned x_column values and corresponding goal rates.
        """
        # Work with a copy of the DataFrame to avoid SettingWithCopyWarning
        df = self.dataframe.copy()
        df['bin'] = pd.cut(df[x_column], bins=num_bins)

        # Group by the bins and calculate goals and totals, explicitly setting observed=False
        grouped = df.groupby('bin', observed=False).agg(
            goals=('is_goal', 'sum'),
            total=('is_goal', 'count')
        )
        grouped['goal_rate'] = grouped['goals'] / grouped['total']

        # Add the bin center for plotting
        grouped['bin_center'] = grouped.index.map(lambda x: x.mid)
        return grouped[['bin_center', 'goal_rate']]

    def plot_goal_rate(self, x_column, num_bins, xlabel, title):
        """
        Plot goal rate as a function of a given column (distance or angle).

        Parameters:
        - x_column: The column to calculate and plot goal rate against.
        - num_bins: Number of bins for grouping.
        - xlabel: Label for the x-axis.
        - title: Title of the plot.
        """
        # Correct call to calculate_goal_rate
        goal_rate_data = self.calculate_goal_rate(x_column=x_column, num_bins=num_bins)

        # Plotting the goal rate
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=goal_rate_data['bin_center'], y=goal_rate_data['goal_rate'], marker='o')
        plt.title(title, fontsize=14)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel("Goal Rate", fontsize=12)
        plt.grid(True)
        plt.show()





