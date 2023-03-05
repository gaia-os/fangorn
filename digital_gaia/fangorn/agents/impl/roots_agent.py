from jax.scipy.stats import norm
from digital_gaia.fangorn.ontology.v1.measurement.base.agriculture.Yield import HempYield
from digital_gaia.fangorn.ontology.v1.genetics.base.plant.Plant import PlantSpecies
from digital_gaia.fangorn.ontology.v1.management.base.agriculture.Harvest import HarvestCrops as HarvestCrops
from digital_gaia.fangorn.ontology.v1.management.base.agriculture.Planting import PlantingSeeds as PlantingSeeds
from digital_gaia.fangorn.ontology.v1.management.base.agriculture.Fertilizer import FertilizeSoil as FertilizeSoil
from digital_gaia.fangorn.ontology.v1.management.base.agriculture.Irrigation import IrrigateCrops as IrrigateCrops
from digital_gaia.fangorn.ontology.v1.management.base.agriculture.Pruning import PruneCrops as PruneCrops
from numpyro.infer.autoguide import AutoMultivariateNormal
from jax.tree_util import tree_map
from jax.numpy import stack, pad, array
from digital_gaia.fangorn.agents.AgentInterface import AgentInterface
import jax.numpy as jnp
import numpy as np
from numpyro import deterministic, plate
from numpyro.contrib.control_flow import scan
from numpyro.distributions import Normal
from numpyro.primitives import sample

# TODO this needs to be moved in the ontology, no sure where
from enum import IntEnum
class SoilType(IntEnum):
    """
    The variable representing the type of soil.
    """
    VeryCoarseSands = 0
    CoarseSands = 1
    FineSands = 2
    LoamySands = 3
    SandyLoams = 4
    FineSandyLoams = 5
    VeryFineSandyLoams = 6
    Loams = 7
    SiltLoams = 8
    ClayLoams = 9
    SiltyClayLoams = 10
    SandyClayLoams = 11
    SandyClays = 12
    SiltyClays = 13
    Clays = 14

# TODO this needs to be moved in the ontology, no sure where
from enum import IntEnum
class SoilOrganicMatter(IntEnum):
    """
    The variable representing the soil organic matter.
    """
    Continuous = 0


class RootsAndCultureAgent(AgentInterface):
    """
    # TODO is that correct?
    # TODO is there a better way to describe it?
    A class implementing an agent specialised for roots and culture.
    """

    # The agent's species
    species = [
        AgentInterface.ontology_name(PlantSpecies, "Hemp")
    ]

    # The agent's actions
    actions = [
        AgentInterface.ontology_name(PlantingSeeds, "HempSeeds"),
        AgentInterface.ontology_name(HarvestCrops, "Hemp"),
        AgentInterface.ontology_name(PruneCrops, "Yes"),
        AgentInterface.ontology_name(FertilizeSoil, "Yes"),
        AgentInterface.ontology_name(IrrigateCrops, "Yes")
    ]

    # The agent's name
    name = "Roots-and-Culture.roots-indoor1.Deterministic"

    def __init__(self, data):
        """
        Construct the roots and culture agent
        :param data: an instance of the data loader containing information about the available reports and lots.
        """
        # Call parent constructor
        obs_to_site = {
            AgentInterface.ontology_name(HempYield, "Continuous"): "obs_yield",
            AgentInterface.ontology_name(SoilOrganicMatter, "Continuous"): "obs_soil_organic_matter"
        }
        super().__init__(data, obs_to_site)

        # Store lot information
        self.n_lots = len(self.data.project.lots)
        # TODO this should be computed based on the polygon coordinates
        self.lot_area = jnp.array([100.] * self.n_lots)  # TODO change that to make it realistic

        # Store actions information
        self.action_names = ["irrigation", "fertilizer", "pruning", "planting", "harvesting"]
        self.n_actions = len(self.action_names)

        # Pre-process the default policies
        self.policy = stack([self.to_array(policy) for policy in data.policies], -2)

    def to_array(self, policy):
        """
        Convert the list of action names into the corresponding array of action indices
        :param policy: the list of action names
        :return: the array of action indices
        """

        # Replace action names by their corresponding indices
        policy = tree_map(lambda action: self.index_of(action) + 1 if action else action, policy)

        # Pad the actions to ensure they all have the same length
        return stack([pad(array(actions), (0, self.n_actions - len(actions))) for actions in policy])

    def add_reports(self, reports):
        """
        Provide new reports to the model
        :param reports: the new reports
        """

        # Extract reports content as numpy array
        np_reports = stack([self.extract_measurements(reports, lot) for lot in range(self.n_lots)])
        np_reports = jnp.expand_dims(np_reports, axis=0)

        # Merge new report to already existing reports
        self.reports = np_reports if self.reports is None else jnp.concatenate((self.reports, np_reports), axis=0)

    def extract_measurements(self, reports, lot_id):
        """
        Extract the measurements associated with a lot
        :param reports: the reports available
        :param lot_id: the lot's index whose measurements need to be retrieved
        :return: the lot's measurements
        """

        # Select only the measurements correspond to i-th lot
        reports = reports[reports['lot'] == lot_id]

        # Create the numpy array that will contain the measurements
        observations = np.zeros(self.number_of_measurements(reports))

        # Iterate over the observation names
        obs_id = 0
        for obs_name in self.observations:

            # Iterate over the measurements associated with these observations
            for measurements in reports[obs_name]:

                # Extract the measurements
                measurements = [measurements] if isinstance(measurements, float) else measurements
                for measurement in measurements:
                    observations[obs_id] = measurement
                    obs_id += 1

        return observations

    def number_of_measurements(self, reports):
        """
        Getter
        :param reports: the reports
        :return: the number of measurements in the reports
        """

        # Get the measurements in the first row of the reports
        measurements = reports[self.observations].iloc[0].tolist()

        # Iterate over the measurements
        n_reports = 0
        for measurement in measurements:

            # Process each measurement
            if isinstance(measurement, float):
                n_reports += 1
            elif isinstance(measurement, list):
                n_reports += len(measurement)

        # Divide the total number of measurement by the number of lots
        return int(n_reports)

    def guide(self, *args, **kwargs):
        """
        Getter
        :param args: the guide's arguments
        :param kwargs: the guide's keyword arguments
        :return: the guide
        """
        return AutoMultivariateNormal(self.model)()

    def model(self, *args, time_horizon=-1, mask=None, **kwargs):
        """
        Implement the generative model of the agent
        :param args: unused positional arguments
        :param time_horizon: the time horizon of planning, if it equals -1, the length of the current policy is used
        :param kwargs: unused keyword arguments
        """

        # Make sure the time horizon is value
        time_horizon = len(self.policy) if time_horizon == -1 else time_horizon

        # Create time indices from zero up to the time horizon
        time_indices = jnp.arange(0, time_horizon)

        # Create the model's parameters
        parameters = self.get_parameters(time_horizon)

        # Initialise the states at time zero
        plant_count = jnp.zeros(self.n_lots)
        plant_size = jnp.zeros(self.n_lots)
        soil_organic_matter = jnp.ones(self.n_lots) * parameters["soil_organic_matter"]
        growth_rate = jnp.zeros(self.n_lots)
        wilting = jnp.ones(self.n_lots) * 0.5 * (parameters["saturation_point"] + parameters["wilting_point"])

        # Create the initial states
        initial_states = (
            plant_count,
            plant_size,
            soil_organic_matter,
            growth_rate,
            wilting,
            parameters
        )

        # Call the scan function that unroll the model over time
        scan(self.model_dynamic, initial_states, (time_indices, self.policy[:time_horizon], mask))

    def get_parameters(self, time_horizon):
        """
        Getter
        :param time_horizon: the time horizon of planning
        :return: a dictionary containing the model's parameters
        """

        # TODO is there some uncertainty about the parameters?

        # TODO is there a need for different parameters for different lots?
        #  For example, different soil_types/max_root_depth/etc... in different lots
        # with plate('num_lots', self.n_lots):
        #     parameters['key'] = sample('key', Normal(0, 1))

        # Retrieve the default parameters
        parameters = {
            "growth_function_mean": self.get_growth_function_mean(time_horizon),
            "growth_function_std": self.get_growth_function_std(time_horizon),
            "max_growth_rate": 0.5,  # TODO change that to make it realistic
            "evaporation_rate": 0.5,  # TODO change that to make it realistic
            "soil_type": SoilType.Clays,  # TODO you may want to change that to fit your needs
            "lot_area": self.lot_area,
            "max_root_depth": 0.5,  # TODO change that to make it realistic
            "max_evapotranspiration_rate": 0.5,  # TODO change that to make it realistic
            "yield_potential": 0.5,  # TODO change that to make it realistic
            "saturation_points": self.get_saturation_point(),
            "wilting_points": self.get_wilting_point(),
            "soil_organic_matters": self.get_soil_organic_matters(),
            "n_seeds": 84,  # TODO you may want to change that to fit your needs
            "time_delta": 1,  # TODO change that to make it realistic, it is equal to 1 week and to compute plant size
            "weekly_irrigation": 2,  # TODO you may want to change that to make it realistic
            "obs_soil_organic_matter_std": 0.2,  # TODO you may want to change that to make it realistic
            "obs_yield_std": 0.2  # TODO you may want to change that to make it realistic
        }

        # Get the soil type and soil volume
        soil_type = parameters["soil_type"]
        soil_volume = self.lot_area * parameters["max_root_depth"]

        # Compute the soil organic matter
        parameters["soil_organic_matter"] = parameters["soil_organic_matters"][soil_type]

        # Get the conversion ration from cubic meter to liters
        m3_to_l = 1000

        # Compute lot saturation and wilting points
        parameters["saturation_point"] = soil_volume * parameters["saturation_points"][soil_type] * m3_to_l
        parameters["wilting_point"] = soil_volume * parameters["wilting_points"][soil_type] * m3_to_l

        return parameters

    @staticmethod
    def get_wilting_point():
        """
        Getter
        :return: a dictionary whose keys are soil types and values are the associated wilting point
        """
        return {  # TODO check that these values are correct
            SoilType.VeryCoarseSands: 3.333,
            SoilType.CoarseSands: 6.25,
            SoilType.FineSands: 6.25,
            SoilType.LoamySands: 6.25,
            SoilType.SandyLoams: 10.416,
            SoilType.FineSandyLoams: 10.416,
            SoilType.VeryFineSandyLoams: 12.5,
            SoilType.Loams: 12.5,
            SoilType.SiltLoams: 12.5,
            SoilType.ClayLoams: 14.583,
            SoilType.SiltyClayLoams: 14.583,
            SoilType.SandyClayLoams: 14.583,
            SoilType.SandyClays: 13.333,
            SoilType.SiltyClays: 13.333,
            SoilType.Clays: 13.333
        }

    @staticmethod
    def get_saturation_point():
        """
        Getter
        :return: a dictionary whose keys are soil types and values are the associated saturation point
        """
        return {  # TODO check that these values are correct
            SoilType.VeryCoarseSands: 6.25,
            SoilType.CoarseSands: 10.416,
            SoilType.FineSands: 10.416,
            SoilType.LoamySands: 10.416,
            SoilType.SandyLoams: 14.583,
            SoilType.FineSandyLoams: 14.583,
            SoilType.VeryFineSandyLoams: 19.166,
            SoilType.Loams: 19.166,
            SoilType.SiltLoams: 19.166,
            SoilType.ClayLoams: 20.833,
            SoilType.SiltyClayLoams: 20.833,
            SoilType.SandyClayLoams: 20.833,
            SoilType.SandyClays: 20.833,
            SoilType.SiltyClays: 20.833,
            SoilType.Clays: 20.833
        }

    @staticmethod
    def get_soil_organic_matters():
        """
        Getter
        :return: a dictionary whose keys are soil types and values are the associated soil organic matters
        """
        return {  # TODO change that to make it realistic
            SoilType.VeryCoarseSands: 2,
            SoilType.CoarseSands: 2,
            SoilType.FineSands: 2,
            SoilType.LoamySands: 2,
            SoilType.SandyLoams: 2,
            SoilType.FineSandyLoams: 2,
            SoilType.VeryFineSandyLoams: 2,
            SoilType.Loams: 2,
            SoilType.SiltLoams: 2,
            SoilType.ClayLoams: 2,
            SoilType.SiltyClayLoams: 2,
            SoilType.SandyClayLoams: 2,
            SoilType.SandyClays: 2,
            SoilType.SiltyClays: 2,
            SoilType.Clays: 2
        }

    @staticmethod
    def get_growth_function_mean(time_horizon):
        """
        Compute the mean of the Gaussian modelling the growth rate over time
        :param time_horizon: the time horizon of planning
        :return: the mean
        """
        # Inflection points are given by the Gaussian mean plus or minus its standard deviation, thus the mean is:
        return time_horizon / 2

    @staticmethod
    def get_growth_function_std(time_horizon):
        """
        Compute the standard deviation of the Gaussian modelling the growth rate over time
        :param time_horizon: the time horizon of planning
        :return: the standard deviation
        """
        # Inflection points are given by the Gaussian mean plus or minus its standard deviation (std), thus the std is:
        return time_horizon / 6

    def model_dynamic(self, states_t, values_t):
        """
        Implement the model dynamic
        :param states_t: the states of the lots at time t
        :param values_t: the values at time t
        :return: the states at time t + 1
        """

        # Unpack the states at time t
        plant_count_t, \
            plant_size_t, \
            soil_organic_matter_t, \
            growth_rate_t, \
            wilting_t, \
            params = states_t

        # Unpack the values at time t
        t, actions_performed, mask = values_t

        # Duplicate the model for each lot
        with plate('num_lots', self.n_lots):

            # Check if planting, fertilising, pruning, harvesting and irrigation are performed at time t
            plant = self.is_performed("planting", actions_performed)
            harvest = self.is_performed("harvesting", actions_performed)
            fertilize = self.is_performed("fertilizer", actions_performed)
            prune = self.is_performed("pruning", actions_performed)
            irrigate = self.is_performed("irrigation", actions_performed)

            # Compute the number of plants at time t + 1
            plant_count_t1 = self.compute_plant_count(plant_count_t, plant, harvest, params["n_seeds"])

            # Compute the plant size at time t + 1
            plant_size_t1 = self.compute_plant_size(
                plant_count_t,  plant_count_t1, plant_size_t, growth_rate_t, prune, params["time_delta"]
            )

            # Compute the soil organic matter at time t + 1
            soil_organic_matter_t1 = self.compute_soil_organic_matter(
                soil_organic_matter_t, plant_size_t, plant_count_t, params["lot_area"], prune, fertilize
            )

            # Compute the evapotranspiration rate at time t + 1
            evapotranspiration_rate_t1 = self.compute_evapotranspiration_rate(
                plant_count_t1, plant_size_t, plant_size_t1, params["max_evapotranspiration_rate"]
            )

            # Compute the soil water status at time t + 1
            wilting_t1 = self.compute_wilting(
                wilting_t, evapotranspiration_rate_t1, irrigate,
                params["weekly_irrigation"], params["evaporation_rate"], params["lot_area"], params["saturation_point"]
            )

            # Compute the soil water status at time t + 1
            soil_water_status_t1 = self.compute_soil_water_status(
                wilting_t1, params["wilting_point"], params["saturation_point"]
            )

            # Compute the growth rate at time t + 1
            growth_rate_t1 = self.compute_growth_rate(
                t, soil_water_status_t1, fertilize,
                params["growth_function_mean"], params["growth_function_std"], params["max_growth_rate"]
            )

            # Computed the expected observed yield at time t + 1
            self.compute_yield(
                plant_count_t1, plant_size_t1, harvest, params["yield_potential"], params["obs_yield_std"], mask
            )

            # Computed the expected observed soil organic matter at time t + 1
            self.compute_obs_soil_organic_matter(soil_organic_matter_t1, params["obs_soil_organic_matter_std"], mask)

        # Create the states at time t + 1
        states_t1 = (
            plant_count_t1,
            plant_size_t1,
            soil_organic_matter_t1,
            growth_rate_t1,
            wilting_t1,
            params
        )
        return states_t1, None

    def is_performed(self, action, actions_performed):
        """
        Check whether an action is performed
        :param action: the action for which the check is done
        :param actions_performed: all the performed actions
        :return: True if the action is performed, False otherwise
        """
        return jnp.any(self.index_of(action) + 1 == actions_performed, -1)

    def index_of(self, action_name):
        """
        Getter
        :param action_name: the name of the action whose index must be returned
        :return: the action index
        """
        return self.action_names.index(action_name)

    @staticmethod
    def compute_wilting(
        wilting_t, evapotranspiration_rate_t1, irrigate,
        weekly_irrigation, evaporation_rate, lot_area, saturation_point
    ):
        """
        Computed the wilting at time t + 1
        :param wilting_t: the wilting at time t
        :param evapotranspiration_rate_t1: the soil organic matter at time t + 1
        :param irrigate: whether irrigation is performed
        :param weekly_irrigation: the quantity of water used while irrigating
        :param evaporation_rate: a constant representing the evaporation rate
        :param lot_area: the lot's area
        :param saturation_point: the soil's saturation point
        :return: the wilting at time t + 1
        """
        evaporation_effect = evapotranspiration_rate_t1 + (evaporation_rate * lot_area)
        irrigation_effect = weekly_irrigation * irrigate
        wilting_t1 = wilting_t + irrigation_effect - evaporation_effect
        return deterministic('wilting', jnp.clip(wilting_t1, a_max=saturation_point))

    @staticmethod
    def compute_soil_water_status(wilting_t1, wilting_point, saturation_point):
        """
        Computed the soil water status at time t + 1
        :param wilting_t1: the wilting at time t + 1
        :param wilting_point: the soil organic matter at time t + 1
        :param saturation_point: a constant representing the evaporation rate
        :return: the soil water status at time t + 1
        """
        soil_water_status_t1 = (wilting_t1 - wilting_point) / (saturation_point - wilting_point)
        return deterministic('soil_water_status', jnp.clip(soil_water_status_t1, a_min=0, a_max=1))

    @staticmethod
    def compute_soil_organic_matter(soil_organic_matter_t, plant_size_t, plant_count_t, lot_area, prune, fertilize):
        """
        Computed the soil organic matter at time t + 1
        :param soil_organic_matter_t: the soil organic matter at time t
        :param plant_size_t: the plant size at time t
        :param plant_count_t: the plant count at time t
        :param lot_area: the lot's area
        :param prune: whether pruning is performed
        :param fertilize: whether fertilizing is performed
        :return: the soil organic matter at time t + 1
        """

        # Compute the effect of only pruning
        prune_effect = (plant_count_t * plant_size_t * 0.02) / lot_area
        only_prune = prune * (1 - fertilize)
        soil_organic_matter_t1 = soil_organic_matter_t + prune_effect * soil_organic_matter_t * only_prune

        # Compute the effect of only fertilizing
        only_fertilize = fertilize * (1 - prune)
        soil_organic_matter_t1 = soil_organic_matter_t1 + 0.01 * soil_organic_matter_t * only_fertilize

        # Compute the effect of both pruning and fertilizing
        prune_and_fertilize = fertilize * prune
        both_effect = (plant_count_t * plant_size_t * 0.04) / lot_area
        soil_organic_matter_t1 = soil_organic_matter_t1 + both_effect * soil_organic_matter_t * prune_and_fertilize

        return deterministic('soil_organic_matter', soil_organic_matter_t1)

    @staticmethod
    def compute_plant_size(plant_count_t, plant_count_t1, plant_size_t, growth_rate_t, pruning, time_delta):
        """
        Compute the plant size at time t + 1
        :param plant_count_t: the plant count at time t
        :param plant_count_t1: the plant count at time t + 1
        :param plant_size_t: the plant size at time t
        :param growth_rate_t: the growth rate at time t
        :param pruning: whether pruning is performed
        :param time_delta: the real-life duration between two time steps modeled by the agent
        :return: the plant size at time t + 1
        """

        # Apply the immediate reduction in plant size caused by pruning
        initial_planting_size = 0.1
        n_seeds_planted = jnp.clip(plant_count_t1 - plant_count_t, a_min=0)
        total_plant_count = jnp.clip(plant_count_t1, a_min=1)
        plant_size_t1 = (plant_size_t * plant_count_t + initial_planting_size * n_seeds_planted) / total_plant_count

        # Apply the immediate reduction in plant size caused by pruning
        pruning_effect = 1 - 0.2 * pruning
        plant_size_t1 = plant_size_t1 * pruning_effect

        # Apply the immediate reduction in plant size due to the effect of pruning
        pruning_effect = 1 + 0.1 * pruning
        plant_size_t1 = plant_size_t1 + plant_size_t1 * growth_rate_t * time_delta * pruning_effect
        return deterministic('plant_size', plant_size_t1)

    @staticmethod
    def compute_growth_rate(
        t, soil_water_status_t1, fertiliser, growth_function_mean, growth_function_std, max_growth_rate
    ):
        """
        Compute the growth rate at time t + 1
        :param t: the time step t
        :param soil_water_status_t1: the soil water status at time t + 1
        :param fertiliser: whether some fertiliser is applied
        :param growth_function_mean: the mean of the function defining the plant growth for each time step
        :param growth_function_std: the standard deviation of the function defining the plant growth for each time step
        :param max_growth_rate: the maximum growth rate of the plant
        :return: the growth rate at time t + 1
        """

        # Compute the effect of the fertiliser on the growth rate at time t + 1
        fertiliser_effect = 1 + 0.1 * fertiliser

        # Compute the growth rate at time t + 1
        growth_rate_t1 = norm.pdf(t + 1, loc=growth_function_mean, scale=growth_function_std)
        growth_rate_t1 = growth_rate_t1 * soil_water_status_t1 * fertiliser_effect
        return deterministic('growth_rate', jnp.clip(growth_rate_t1, a_min=0, a_max=max_growth_rate))

    @staticmethod
    def compute_evapotranspiration_rate(plant_count_t1, plant_size_t, plant_size_t1, max_evapotranspiration_rate):
        """
        Compute the new evapotranspiration rate
        :param plant_count_t1: the number of plants at time t + 1
        :param plant_size_t: the plant size at time t
        :param plant_size_t1: the plant size at time t + 1
        :param max_evapotranspiration_rate: the maximum evapotranspiration rate
        :return: the evapotranspiration rate at time t + 1
        """

        # Compute the average plant size over the week
        average_plant_size = 0.5 * (plant_size_t + plant_size_t1)

        # Compute the evapotranspiration rate at time t + 1
        evapotranspiration_rate_t1 = plant_count_t1 * average_plant_size * max_evapotranspiration_rate
        return deterministic('evapotranspiration_rate', evapotranspiration_rate_t1)

    @staticmethod
    def compute_plant_count(plant_count_t, plant, harvest, n_seeds):
        """
        Compute the new plant count
        :param plant_count_t: the plant count at time t
        :param plant: whether planting is performed
        :param harvest: whether harvesting is performed
        :param n_seeds: the number of plant
        :return: the plant count at time t + 1
        """
        return deterministic('plant_count', (plant_count_t + plant * n_seeds) * (1 - harvest))

    @staticmethod
    def compute_yield(plant_count, plant_size, harvest, yield_potential, obs_yield_standard_dev, mask):
        """
        Compute the yield based on the system states and model's parameters
        :param plant_count: the number of plants currently present
        :param plant_size: the average size of the plants
        :param harvest: whether harvesting is performed
        :param yield_potential: the yield potential of the plant
        :param obs_yield_standard_dev: the standard deviation of the yield measurement
        :param mask: the observation mask describing when the yield is observed
        :return: the expected yield
        """

        # Get the mask corresponding to the yield
        yield_mask = RootsAndCultureAgent.get_mask(mask, 'obs_yield')

        # Create the yield distribution
        obs_yield_mean = plant_count * plant_size * yield_potential * (1 - harvest)
        obs_yield_std = obs_yield_standard_dev * harvest
        yield_distribution = Normal(obs_yield_mean, obs_yield_std).mask(yield_mask)

        # Sample from the yield distribution
        return sample('obs_yield', yield_distribution)

    @staticmethod
    def compute_obs_soil_organic_matter(soil_organic_matter_t1, obs_soil_organic_matter_std, mask):
        """
        Computed the expected observed soil organic matter at time t + 1
        :param soil_organic_matter_t1: the true soil organic matter in the system at time t + 1
        :param obs_soil_organic_matter_std: the standard deviation of the soil organic matter measurement
        :param mask: the observation mask describing when the soil organic matter is observed
        :return: the expected observed soil organic matter at time t + 1
        """

        # Get the mask corresponding to the soil organic matter
        soil_organic_matter_mask = RootsAndCultureAgent.get_mask(mask, 'obs_soil_organic_matter')

        # Create the soil organic matter distribution
        som_distribution = Normal(soil_organic_matter_t1, obs_soil_organic_matter_std)
        som_distribution = som_distribution.mask(soil_organic_matter_mask)

        # Sample from the soil organic matter distribution
        return sample('obs_soil_organic_matter', som_distribution)

    @staticmethod
    def get_mask(mask, obs_name):
        """
        Getter
        :param mask: all the observation masks
        :param obs_name: the name of the observation whose mask needs to be retrieved
        :return: the mask corresponding to the observation name
        """
        return True if mask is None or obs_name not in mask.keys() else mask[obs_name]
