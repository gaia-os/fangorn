from os.path import abspath, dirname
import digital_gaia.fangorn as fangorn
from digital_gaia.fangorn.agents.AgentFactory import AgentFactory
from digital_gaia.fangorn.assessment.DataLoader import DataLoader
from digital_gaia.fangorn.visualisation.distributions import draw_beliefs
import matplotlib.pyplot as plt


def run_like_in_notebook():
    """
    Run the agent like presented in the notebook
    """
    # Load the data corresponding to the Roots & Culture model
    fangorn_dir = dirname(dirname(dirname(abspath(fangorn.__file__))))
    data_loader = DataLoader(f"{fangorn_dir}/data/projects/Roots-and-Culture/roots-indoor1.json")

    # Load the agent(s) compatible with the loaded data
    agents = AgentFactory.create(data_loader, verbose=True, debug=True)

    # Get the deterministic agent of Roots & Culture
    agent = next(filter(lambda a: a.name == "Roots-and-Culture.roots-indoor1.Deterministic", agents))

    # Predict the future using the deterministic agent
    prediction_samples = agent.predict(model=agent.model, num_samples=1)

    # Draw prior beliefs using the predictive samples
    draw_beliefs(
        prediction_samples,
        var_1={
            "soil_organic_matter": "Soil organic matter",
            "soil_water_status": "Soil water status",
            "growth_rate": "Growth rate",
            "plant_count": "Plant count",
            "evapotranspiration_rate": "Evapotranspiration rate"
        },
        var_2={
            "obs_soil_organic_matter": "Measured soil organic matter",
            "wilting": "Wilting",
            "plant_size": "Plant size",
            "obs_yield": "Yield",
            None: None
        },
        measured=[True, False, False, False, False],
        fig_size=(16, 4.5)
    )
    plt.show()
    plt.waitforbuttonpress()


if __name__ == '__main__':
    # Entry point performing project assessment
    run_like_in_notebook()
