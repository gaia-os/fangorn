from digital_gaia.fangorn.assessment.DataLoader import DataLoader
from digital_gaia.fangorn.assessment import ArgumentParser
from digital_gaia.fangorn.agents.AgentFactory import AgentFactory
from jax.numpy import concatenate
import traceback
import os.path
import jax.numpy as jnp
from datetime import timedelta
from digital_gaia.fangorn.pydantic.Assessment import Assessment, Belief
import re


class Engine:
    """
    A class performing the assessment of a project
    """

    def __init__(self, verbose=False, debug=False):
        """
        Construct the assessment fangorn
        :param verbose: True if useful information should be displayed, False otherwise
        :param debug: True if debug information should be displayed, False otherwise
        """
        # Load arguments
        self.parser = ArgumentParser.ArgumentParser()
        args = self.parse_arguments()
        self.project_file = args.project
        self.reports_dir = args.reports_dir

        # Load the data and requested agents
        self.data = DataLoader(self.project_file, self.reports_dir, verbose=verbose)
        self.agents = AgentFactory.create(self.data, verbose=verbose)

        # Keep track of verbose and debug level
        self.verbose = verbose
        self.debug = debug

    def add_agent(self, agent):
        """
        Add an agent to the list of existing agents
        :param agent: the agent to be added
        """
        self.agents.append(agent)

    def parse_arguments(self):
        """
        Parse the arguments of the 'perform_assessment.py' script
        """
        return self.parser.parse()

    def perform_assessment(self):
        """
        Perform a project assessment
        """
        # Display functions arguments
        if self.verbose:
            print(f"[INFO] Perform the assessment of the following project:")
            print(f"[INFO]     - project_file = {self.project_file}")
            print(f"[INFO]     - report_dir = {self.reports_dir}")

        # Perform the project assessment
        results = {}
        ignored_agents = []
        for t, report in sorted(self.data.reports, key=lambda x: x[0]):

            # For each agent
            for agent in self.agents:  # TODO replace iterations over project by agent averaging

                # Ignore the agent if it has already failed to produce an assessment
                if agent.name in ignored_agents:
                    continue

                try:
                    # Perform the assessment of the project at time step t, by providing an agent with a new report
                    result = self.assess_project(t, agent, report)

                    # Keep track of results
                    if agent.name not in results.keys():
                        results[agent.name] = []
                    results[agent.name].append(result)

                except Exception as e:

                    # Report the assessment failure to the user and ignore the agent
                    self.report_assessment_failure(t, agent, report)
                    ignored_agents.append(agent.name)

        return results

    def report_assessment_failure(self, t, agent, report):
        """
        Report the assessment failure to the user
        :param t: the current time step
        :param agent: the agent used for the assessment
        :param report: the new report that was provided to the agent
        """
        # Inform the user that an error occurred, if requested
        if self.verbose:
            print(f"[ERROR] Could not assess project at time {t} using agent {agent.name}.")
            exclude_columns = {'report_id', 'datetime', 'location', 'reporter', 't', 'lot', 'index'}
            columns = set(report.columns) - exclude_columns
            print(f"[ERROR] The report contains the following columns:")
            for column in columns:
                print(f"[ERROR] - {column}")

        # Display debug information, if requested
        if self.debug is True:
            print(f"[DEBUG] \n[DEBUG] ", end='')
            print(traceback.format_exc().replace("\n", "\n[DEBUG] "))

        # Inform the user that the agent will be ignored from now on, if requested
        if self.verbose:
            print(f"[INFO] The agent {agent.name} will now be ignored.")

    def assess_project(self, t, agent, report):
        """
        Perform the assessment of a project
        :param t: the current time step
        :param agent: the agent to use for the assessment
        :param report: the new report to provide to the agent
        :return: the samples from the sample site and the expected free energy
        """

        # Let the user know that the assessment is in process, if requested
        if self.verbose:
            print(f"[INFO] ")
            print(f"[INFO] Performing assessment of {agent.name} agent at time step t={t}...")

        # Provide a new report to the agent
        agent.add_reports(reports=report)

        # Get reports information sorted by agent's sample sites
        reports = agent.get_report_by_sample_site()

        # Perform inference using available data
        cond_model, cond_guide = agent.condition_all(reports)
        inference_samples = agent.inference_algorithm().run_inference(
            model=cond_model,
            guide=cond_guide,
            inference_params={"time_horizon": t + 1},
        )

        # Perform prediction of future random variables
        prediction_samples = agent.predict(
            model=agent.conditioned_model,
            posterior_samples=inference_samples,
            return_sites=list(inference_samples.keys())
        )

        # Concatenate report information with predictive samples
        for site_name, data in reports.items():
            samples = prediction_samples[site_name].mean(axis=0)[data.shape[0]:]
            reports[site_name] = concatenate((data, samples), axis=0)

        # Compute the expected free energy
        agent.condition_model(reports)
        efe = agent.efe(prediction_samples, t)
        if self.verbose:
            print(f"Expected Free Energy of {agent.name} agent at time step {t}: {efe}")

        return inference_samples, prediction_samples, efe

    @staticmethod
    def compute_belief(sample_site, samples):
        """
        Compute the belief associated to the samples passed as parameters
        :param sample_site: the sample site name
        :param samples: the samples for which the belief need to be computed
        :return: the computed belief
        """

        # Ensure that the sample site has four dimensions
        if samples.ndim > 4:
            print(f"[WARNING] The sample site {sample_site} have too many dimensions.")
            return None
        while samples.ndim < 4:
            samples = jnp.expand_dims(samples, axis=1)

        # Compute the belief of the sample site
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)
        return Belief(**{
            "mean": mean.tolist(),
            "lower_limit": (mean - std).tolist(),
            "upper_limit": (mean + std).tolist()
        })

    def compute_beliefs(self, all_samples):
        """
        Compute the beliefs associated with the samples passed as parameters
        :param all_samples: the samples whose beliefs should be computed
        :return: the computed beliefs
        """
        beliefs = {
            sample_site: self.compute_belief(sample_site, samples)
            for sample_site, samples in all_samples.items()
        }
        return dict(filter(lambda pair: pair[1] is not None, beliefs.items()))

    def save(self, saving_directory, assessment):
        """
        Save the results on the filesystem
        :param saving_directory: the directory in which the results should be saved
        :param assessment: the project assessment
        """

        # Retrieve the project's starting date
        project_start_date = self.data.project.start_date

        # Iterates over all agents
        for agent_name, results in assessment.items():

            # Iterates over time steps
            for t, (inference_samples, prediction_samples, efe) in enumerate(results):

                # Create the pydantic assessment
                pydantic_assessment = Assessment(**{
                    "date": project_start_date + t * timedelta(days=7),
                    "efe": efe,
                    "beliefs": self.compute_beliefs(inference_samples),
                    "predictions": self.compute_beliefs(prediction_samples)
                })

                # Create the saving directory, if it does not exist
                full_saving_directory = f"{saving_directory}/{agent_name}/{t}/"
                os.makedirs(full_saving_directory, exist_ok=True)

                # Write the pydantic assessment on the file system
                with open(f"{full_saving_directory}/{agent_name}.json", "w") as f:
                    json_string = pydantic_assessment.json(indent=4)
                    json_string = re.sub("[\[]\n[ ]+", "[", json_string)
                    json_string = re.sub("\n[ ]+[\]]", "]", json_string)
                    json_string = re.sub("\n[ ]+[\[]", "[", json_string)
                    f.write(json_string)
