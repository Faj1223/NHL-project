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
            x="shooting_distance",
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
            x="shot_angle",
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
            x="shooting_distance",
            y="shot_angle",
            kind="scatter",  # Scatter plot for the center
            marginal_kws=dict(bins=30, fill=True, color="blue")  # Histogram settings for the margins
        )
        # Add a title to the entire plot
        plt.suptitle("Scatter Plot with Marginal Histograms", fontsize=16, y=1.02)
        joint_plot.set_axis_labels("Distance to Net", "Angle to Net (degrees)", fontsize=12)
        plt.show()

    def calculate_goal_rate(self, column, num_bins):
        """
        Calculate goal rate as a function of a specified column (e.g., distance_to_net or angle_to_net).

        Parameters:
        - column: The column to calculate goal rate against (e.g., 'distance_to_net', 'angle_to_net').
        - num_bins: Number of bins to group the data.

        Returns:
        - A DataFrame with bin centers and goal rates.
        """
        # Work with a copy of the DataFrame to avoid modifying the original
        df = self.dataframe.copy()

        # Bin the specified column
        df['bin'] = pd.cut(df[column], bins=num_bins)

        # Group by the bins
        grouped = df.groupby('bin', observed=False).agg(
            total_shots=('is_goal', 'count'),
            goals=('is_goal', 'sum')
        )

        # Calculate goal rate
        grouped['goal_rate'] = grouped['goals'] / grouped['total_shots']

        # Add bin center for plotting
        grouped['bin_center'] = grouped.index.map(lambda x: x.mid)
        return grouped

    def plot_goal_rate(self, column, num_bins=20, xlabel="", title=""):
        """
        Plot goal rate as a function of a specified column (e.g., distance_to_net or angle_to_net).

        Parameters:
        - column: The column to calculate and plot goal rate against (e.g., 'distance_to_net', 'angle_to_net').
        - num_bins: Number of bins to group the data.
        - xlabel: Label for the x-axis.
        - title: Title of the plot.
        """
        # Calculate goal rate
        goal_rate_data = self.calculate_goal_rate(column, num_bins)

        # Plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            x=goal_rate_data['bin_center'],
            y=goal_rate_data['goal_rate'],
            marker='o',
            color='blue'
        )
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel("Goal Rate", fontsize=14)
        # Adjust x-axis range dynamically based on the column
        if column == "shooting_distance":
            plt.xlim(0, 100)  # most shots occur at closer ranges,therefore focus on shorter distances
        elif column == "shot_angle":
            plt.xlim(-180, 180)  # Full range of angles in degrees

        plt.grid(True)
        plt.show()

    def plot_goal_histogram(self):
        """
        Plot a histogram of goals by distance, separating empty net and non-empty net events.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=self.dataframe,
            x="shooting_distance",
            hue="empty_net",
            bins=30,
            kde=False,
            palette="viridis",
            alpha=0.7
        )
        plt.title("Goals Binned by Distance to Net (Empty Net vs Non-Empty Net)")
        plt.xlabel("Distance to Net")
        plt.ylabel("Number of Goals")
        plt.legend(title="Empty Net", labels=["Non-Empty Net", "Empty Net"])
        plt.grid(True)
        plt.show()





