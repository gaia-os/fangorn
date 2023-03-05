# Import required packages
from os.path import abspath, dirname
import matplotlib.pyplot as plt
import jax.numpy as jnp
from numpyro.optim import optax_to_numpyro
from numpyro.infer import Trace_ELBO
import optax
from numpyro import set_platform

# Retrieve the fangorn directory
import digital_gaia.fangorn as fangorn

# Import required classes from the natural_models package
from digital_gaia.fangorn.agents.AgentFactory import AgentFactory
from digital_gaia.fangorn.assessment.DataLoader import DataLoader
from digital_gaia.fangorn.kernels.impl.MCMCKernel import MCMCKernel
from digital_gaia.fangorn.kernels.impl.SVIKernel import SVIKernel
from digital_gaia.fangorn.visualisation.distributions import draw_beliefs
from digital_gaia.fangorn.visualisation.distributions import compare_posteriors

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Tell numpyro whether to use cpu or gpu
use_gpu = False
if use_gpu is True:
    set_platform('gpu')
else:
    set_platform('cpu')


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
    plt.waitforbuttonpress()

    # Provide the data to the deterministic agent
    data = {
        'obs_soil_organic_matter': prediction_samples['obs_soil_organic_matter'][0]
    }
    agent.condition_all(data=data)

    # The measurement interval is the number of time steps between two observations made by the agent
    measurement_interval = 4

    # Create the mask
    soil_organic_matter_mask = jnp.arange(len(data['obs_soil_organic_matter']))
    soil_organic_matter_mask = jnp.expand_dims(soil_organic_matter_mask, -1) % measurement_interval == 0
    mask = {
        'obs_soil_organic_matter': soil_organic_matter_mask
    }

    # Create the MCMC algorithm
    mcmc_args = {
        "num_warmup": 1000,
        "num_samples": 1000,
        "num_chains": 1
    }
    mcmc_algorithm = agent.inference_algorithm(MCMCKernel, kernel_args=mcmc_args)

    # Perform inference using MCMC
    inference_args = {
        "mask": mask,
        "data_level": 0
    }
    mcmc_samples = mcmc_algorithm.run_inference(inference_args=inference_args)

    # Create the SVI algorithm
    svi_args = {
        "optimiser": optax_to_numpyro(optax.adabelief(1e-3, eps=1e-8, eps_root=1e-8)),
        "loss": Trace_ELBO(num_particles=10),
        "num_steps": 1000
    }
    svi_algorithm = agent.inference_algorithm(SVIKernel, kernel_args=svi_args)

    # Perform inference using SVI
    inference_params = {
        "stable_update": True,
        "mask": mask,
    }
    svi_samples = svi_algorithm.run_inference(inference_params=inference_params)

    # Compare the initial prediction, the SVI posterior, and the MCMC posterior
    compare_posteriors(
        is_observed=mask['obs_soil_organic_matter'],
        mcmc_samples=mcmc_samples,
        svi_samples=svi_samples,
        prediction_samples=prediction_samples,
        var_names=[
            "soil_organic_matter", "soil_water_status", "growth_rate", "plant_count",
            "evapotranspiration_rate", "wilting", "plant_size"
        ],
        var_labels=[
            "Soil organic matter", "Soil water status", "Growth rate", "Plant count",
            "Evapotranspiration rate", "Wilting", "Plant size"
        ],
        fig_size=(16, 4.5)
    )
    plt.waitforbuttonpress()


if __name__ == '__main__':
    # Entry point performing project assessment
    run_like_in_notebook()
