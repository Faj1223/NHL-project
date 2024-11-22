import os
import datetime

import wandb
from matplotlib import pyplot as plt

from ift6758.controller.model_pipeline.base_model import BaseModel


class ModelComparator:
    def __init__(self, model_instances: list[BaseModel]):
        self.models = model_instances
        self.plot_cache = BaseModel.init_plot_cache()

        self.data_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

        # Create id based on actual date time
        self.comparison_id = ""
        date_now = datetime.datetime.now()
        self.comparison_id += date_now.strftime("%m")
        self.comparison_id += '_'
        self.comparison_id += date_now.strftime("%y")
        self.comparison_id += '_'
        self.comparison_id += date_now.strftime("%d")
        self.comparison_id += '-'
        self.comparison_id += date_now.strftime("%H")
        self.comparison_id += date_now.strftime("%H")
        self.comparison_id += date_now.strftime("%S")


    def __get_models_comparisons_path(self) -> str:
        path = os.path.join(self.data_dir_path, "models_comparisons", f"comparison_{self.comparison_id}")
        if not os.path.exists(path):
            os.makedirs(path)

        return path

    def __get_img_path(self, img_title):
        return os.path.join(self.__get_models_comparisons_path(), f"{img_title}.png")


    def evaluate_models(self):

        # Evaluate each model separately
        for model in self.models:
            model_plot_cache = model.evaluate_model()

            # Combine their plot
            for key, value in model_plot_cache.items():
                self.plot_cache[key].append(value)

    def plot_evaluation_together(self):
        # "roc": [],
        # "goal_rate_by_percentile": [],
        # "cumulative_goals_by_percentile": [],
        # "reliability_diagram": [],
        # "loss": [],
        # "validation_error": [],
        # "training_error": [],


        run_id = f"Comparison_{self.comparison_id}"
        run = wandb.init(
            # Set the project where this run will be logged
            project="IFT6758.2024-A09",
            id=run_id,
        )

        ############################################################
        ##           ROC
        ############################################################

        # namings
        title = "ROC Curve (Combined)"
        save_path = self.__get_img_path("ROC_Curve_Combined")

        axes_to_plot = self.plot_cache['roc']

        # Plotting
        fig, ax = plt.subplots(figsize=(8, 6))

        for eval in axes_to_plot:
            line = eval[0]
            plt.plot(line[0], line[1], label=line[2])

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        fig.savefig(save_path)  # save the figure to file # do it before 'show' as the show function clear cache
        img = plt.imread(save_path)

        # plt.show() # if not showing, close it
        plt.close(fig)

        wandb.log({"ROC Curve": [wandb.Image(img, caption="ROC Curve")]})
        print("--Comparison: ROC Curve sent to WANDB")

        ############################################################
        ##           Cumulative % of Goals
        ############################################################
        title = "Cumulative % of Goals (Combined)"
        save_path = self.__get_img_path("cumul_goals_by_percentile")

        axes_to_plot = self.plot_cache['cumulative_goals_by_percentile']

        # Plotting
        fig, ax = plt.subplots(figsize=(8, 6))

        for eval in axes_to_plot:
            line = eval[0]
            plt.plot(line[0], line[1], label=line[2])

        ax.invert_xaxis()  # Reverse x-axis direction
        ax.set_xlabel("Shot Probability Model Percentile")
        ax.set_ylabel("Cumulative % of Goals")
        ax.set_ylim(0, 100)  # Set y-axis scale to 0–100
        ax.set_title(title)
        ax.grid()
        ax.legend()

        fig.savefig(save_path)  # save the figure to file # do it before 'show' as the show function clear cache
        img = plt.imread(save_path)

        # plt.show() # if not showing, close it
        plt.close(fig)

        wandb.log(
            {"Cumulative % of Goals": [wandb.Image(img, caption="Cumulative % of Goals")]})
        print("--Comparison: Cumulative % of Goals sent to WANDB")

        ############################################################
        ##           Goal Rate per percentile
        ############################################################
        title = "Goal Rate per percentile (Combined)"
        save_path = self.__get_img_path("goal_rate_per_percentile")

        axes_to_plot = self.plot_cache['goal_rate_by_percentile']

        # Plotting
        fig, ax = plt.subplots(figsize=(8, 6))

        for eval in axes_to_plot:
            line = eval[0]
            plt.plot(line[0], line[1], label=line[2])

        ax.invert_xaxis()  # Reverse x-axis direction
        ax.set_xlabel("Shot Probability Model Percentile")
        ax.set_ylabel("Goals / (Shots + Goals) (%)")
        ax.set_ylim(0, 100)  # Set y-axis scale to 0–100
        ax.set_title(title)
        ax.grid()
        ax.legend()

        fig.savefig(save_path)  # save the figure to file # do it before 'show' as the show function clear cache
        img = plt.imread(save_path)

        # plt.show() # if not showing, close it
        plt.close(fig)

        wandb.log({f"Goal Rate per percentile": [wandb.Image(img, caption="Goal Rate per percentile")]})
        print("--Comparison: Goal Rate per percentile sent to WANDB")

        ############################################################
        ##           Reliability Diagram
        ############################################################
        title = "Reliability Diagram (Combined)"
        save_path = self.__get_img_path("reliability_diagram")

        axes_to_plot = self.plot_cache['reliability_diagram']

        # Plotting
        fig, ax = plt.subplots(figsize=(8, 6))

        for eval in axes_to_plot:
            line = eval[0]
            plt.plot(line[0], line[1], marker="o", label=line[2])

        # Add perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

        # Add plot details
        ax.set_title(title)
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.grid(alpha=0.3)

        fig.savefig(save_path)  # save the figure to file # do it before 'show' as the show function clear cache
        img = plt.imread(save_path)

        # plt.show() # if not showing, close it
        plt.close(fig)

        wandb.log(
            {f"Reliability Diagram": [wandb.Image(img, caption="Reliability Diagram")]})
        print("--Comparison: Reliability Diagram sent to WANDB")

        ############################################################
        run.finish()

