from datetime import timedelta
import jax.numpy as jnp
from copy import deepcopy
from geojson_pydantic import Point
from jax.numpy import DeviceArray
from jax import random
from numpyro.infer import Predictive
from numpyro.infer.util import log_likelihood
import numpy as np
import numpyro
import abc
from abc import ABC
from functools import partial
from digital_gaia.fangorn.pydantic.Report import Report
from digital_gaia.fangorn.pydantic.Observation import Observation
from digital_gaia.fangorn.kernels.impl.MCMCKernel import MCMCKernel
from digital_gaia.fangorn.kernels.impl.SVIKernel import SVIKernel
from random import randint


class AgentInterface(ABC):
    """
    An abstract class representing a model
    """

    def __init__(self, data, obs_to_site):
        """
        Constructor
        :param data: the data loader
        :param obs_to_site: a dictionary that maps observations name in the ontology to their sample site name
        """

        # The time horizon of planning
        self.time_horizon = 1

        # A dictionary mapping observations name to their sample sites name
        self.obs_to_site = obs_to_site

        # The reports observed so far
        self.reports = None

        # Random keys
        self.rng_key = random.PRNGKey(0)

        # The model and guide conditioned on some data
        self.conditioned_model = None
        self.conditioned_guide = None

        # Mapping from variable name to extrinsic function
        self.extrinsic = {
            "soil_organic_carbon": self.more_is_better,
            "obs_yield_density": self.more_is_better,
            "obs_plant_height": self.more_is_better,
            "obs_plant_density": self.more_is_better,
            "biomass_carbon_per_m2": self.more_is_better,
            "obs_yield": self.more_is_better
        }

        # Store the data loader
        self.data = data

        # Store the policy
        self.agent_policy = data.policies

    @staticmethod
    def to_point(location):
        """
        Convert a location to a (geojson_pydantic) point.
        :param location: the location to convert
        :return: the created point
        """
        return Point(coordinates=location.tolist() if isinstance(location, DeviceArray) else location)

    def export_reports(self, reports_dir):
        """
        Export the reports created by the agent to the file system
        :param reports_dir: the directory where the reports should be saved
        """
        reports = self.create_reports()
        for report in reports:
            with open(f"{reports_dir}/report-{report.datetime}.{report.id}.json", "w") as f:
                f.write(report.json(indent=2))

    def create_reports(self, domain_name=None):
        """
        Create reports using the agent's model
        :param domain_name: the domain name to use for the email address and provenance in the reports
        :return: a list of created reports
        """

        # Predict the observations value
        model = self.model if self.conditioned_model is None else self.conditioned_model
        predictions = self.predict(model=model, num_samples=1)

        # Create the list of reports and the first report date
        reports = []
        report_date = self.data.project.start_date
        domain_name = self.data.project.name.lower() if domain_name is None else domain_name

        # Iterate over all time steps
        for t in range(0, self.data.T):

            # Iterate over all lots
            for lot_index in range(len(self.data.lots)):

                # Collect the report's observations
                observations = self.create_report_observations(lot_index, t, predictions)

                # Create the report corresponding to a specific time and place, then add it to the list of reports
                report = Report(
                    datetime=report_date,
                    location=self.data.find_point_in(self.data.lots[lot_index]),
                    project_name=self.data.project.name,
                    reporter=f"rob@{domain_name}.com",
                    provenance=f"https://api.{domain_name}.com/get_data?{randint(10000, 99999)}",
                    observations=observations
                )
                reports.append(report)

                # Increase the report date by 7 days, i.e., a week
                report_date += timedelta(days=7)

        return reports

    def create_report_observations(self, lot_index, t, predictions):
        """
        Create the report's observations
        :param lot_index: the index of the lot whose observations need to be created
        :param t: the time step for which the observations need to be created
        :param predictions: the model predictions containing the observations value
        :return: the report's observations
        """

        # Create the report's observations
        observations = []
        for sample_site in self.sample_sites:

            # Create the observation corresponding to each sample site, and add it to the list of observations
            observation = Observation(**{
                "name": sample_site,
                "lot_name": self.data.lots[lot_index].name,
                "value": predictions[sample_site][0][t][lot_index]
            })
            observations.append(observation)

        return observations

    def set_time_horizon(self, t):
        """
        Set the new time horizon
        :param t: the new time horizon
        """
        self.time_horizon = t

    @abc.abstractmethod
    def add_reports(self, reports):
        """
        Provide new reports to the model
        :param reports: the new reports
        """
        ...

    def get_report_by_sample_site(self):
        """
        Gather the report data for each (numpyro) sample site
        :return: a python dictionary whose keys are sample site names and values are report data for these sites
        """
        return {
            self.to_sample_site(obs_name): jnp.expand_dims(self.reports[:, :, obs_id], axis=1)
            for obs_id, obs_name in enumerate(self.data.obs_names) if obs_name in self.observations
        }

    @abc.abstractmethod
    def model(self, *args, **kwargs):
        """
        Implement the generative model
        :param args: the model's arguments
        :param kwargs: the model's keyword arguments
        """
        ...

    @abc.abstractmethod
    def guide(self, *args, **kwargs):
        """
        Implement the guide
        :param args: the guide's arguments
        :param kwargs: the guide's keyword arguments
        :return: the guide
        """
        ...

    def condition_all(self, data):
        """
        Condition the model and the guide on some data
        :param data: the data
        :return: the conditioned model and guide
        """
        self.condition_model(data)
        self.condition_guide(self.conditioned_model)
        return self.conditioned_model, self.conditioned_guide

    def condition_model(self, data):
        """
        Condition the model on some data
        :param data: the data
        :return: the conditioned model
        """
        # Condition the model on the data
        self.conditioned_model = numpyro.handlers.condition(self.model, data=data)
        return self.conditioned_model

    def condition_guide(self, model):
        """
        Condition the guide on some data
        :param model: the model conditioned on the data
        :return: the conditioned guide
        """
        # Ensure that the agent has implemented the variational distribution
        if self.guide is None:
            return None

        # Condition the variational distribution on the data
        self.conditioned_guide = partial(self.guide, model)
        return self.conditioned_guide

    def inference_algorithm(self, kernel=None, **kwargs):
        """
        Getter
        :param kernel: the kernel to use, by default the SVIKernel if a guide is found otherwise the MCMCKernel
        :param kwargs: additional parameters that must be passed to the kernel
        :return: the kernel algorithm to use
        """
        # Retrieve the most informed model
        model = self.model if self.conditioned_model is None else self.conditioned_model

        # If no kernel specified and no guide, create the MCMC inference algorithm
        if kernel is None and self.no_guide_implemented():
            return MCMCKernel(model=model, **kwargs)

        # If no kernel specified but a guide is available, create the SVI inference algorithm
        if kernel is None:
            return SVIKernel(model=model, guide=self.conditioned_guide, **kwargs)

        # Create the inference algorithm requested by the user
        return kernel(model=model, guide=self.conditioned_guide, **kwargs)

    def no_guide_implemented(self):
        """
        Check whether no guide is implemented
        :return: True if no guide is implemented, False otherwise
        """
        try:
            return self.guide() is None
        except Exception as e:
            return False

    def predict(self, **kwargs):
        """
        Perform prediction
        :param kwargs: keyword parameters to send to the "Predictive" class
        :return: the predictions
        """
        self.rng_key, rng_key = random.split(self.rng_key)
        predict = Predictive(**kwargs)
        return predict(rng_key)

    def efe(self, samples, present_time):
        """
        Compute the expected free energy. This function assumes that :
           - the observation is called 'sr' which stands for state report
           - the log-probability of the observation is called 'log_sr'
        If these assumptions does not hold for a particular agent, this function must be overwritten
        :param samples: samples from the posterior distribution
        :param present_time: the present time step
        :return: the expected free energy
        """
        return self.expected_free_energy(samples, present_time, self.sample_sites)

    @staticmethod
    def more_is_better(sr):
        """
        Compute extrinsic value
        :param sr: the state report
        :return: the extrinsic value
        """
        return -sr.mean(axis=0).sum()

    @staticmethod
    def less_is_better(sr):
        """
        Compute extrinsic value
        :param sr: the state report
        :return: the extrinsic value
        """
        return sr.mean(axis=0).sum()

    def expected_free_energy(self, samples, present_time, obs_sites):
        """
        Compute the expected free energy. This function assumes that :
           - if an observation is called 'sr', then its log-probability is called 'log_sr'
        :param samples: samples from the posterior distribution
        :param present_time: the present time step
        :param obs_sites: the name of the observation sites for which the expected free energy must be computed
        :return: the expected free energy
        """
        # Compute the expected free energy
        efe = 0
        for obs_site in obs_sites:
            # Compute standard deviation over samples dimension
            sr = samples[obs_site]
            sr_std = sr.std(axis=0) + 0.000001

            # Compute extrinsic value
            extrinsic_value = self.extrinsic[obs_site](sr)

            # Compute negative posterior entropy
            sqrt_2_pi_e = 4.13273135412
            neg_entropy = -np.log10(sr_std * sqrt_2_pi_e).sum()

            # Compute ambiguity
            log_samples = log_likelihood(self.conditioned_model, samples)
            ambiguity = -log_samples[obs_site][:, present_time:].sum()

            # Add the expected free energy of the current observation site
            efe += extrinsic_value + neg_entropy + ambiguity

        return efe

    @property
    def policy(self):
        """
        Policy getter
        :return: the agent's policy
        """
        return self.agent_policy

    @policy.setter
    def policy(self, new_policy):
        """
        Policy setter
        :param new_policy: the new policy the agent should follow
        """
        self.agent_policy = deepcopy(new_policy)
        self.set_time_horizon(self.agent_policy.shape[0])

    @property
    def observations(self):
        """
        Observations getter
        :return: the name of the observations supported by the agent
        """
        return self.obs_to_site.keys()

    @property
    def sample_sites(self):
        """
        Sample site getter
        :return: the name of the sample sites corresponding to the agent's observations
        """
        return self.obs_to_site.values()

    def to_sample_site(self, obs_name):
        """
        Getter
        :param obs_name: the observation name whose sample site should be returned
        :return: the name of the sample site associated with the observation name
        """
        return self.obs_to_site[obs_name]

    @staticmethod
    def ontology_name(variable_enum, variable_value):
        """
        The name of the variable described by the parameters, i.e., the name of the ontology entry
        :param variable_enum: the enumeration corresponding to the variable
        :param variable_value: the value taken by the enumeration
        :return: the name of the ontology entry
        """
        prefix = "digital_gaia.fangorn.ontology."
        ontology_name = variable_enum.__module__ + "." + variable_enum.__name__ + "." + variable_value
        return ontology_name[len(prefix):]
