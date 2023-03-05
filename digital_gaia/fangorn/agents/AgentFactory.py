import abc
import copy
import os
from digital_gaia.fangorn.agents.AgentInterface import AgentInterface


class AgentFactory:
    """
    A class allowing the creation of all the agents compatible with the project configuration file.
    """

    @staticmethod
    def get_all_classes(path, package):
        """
        Retrieve all the classes within a directory or file
        :param path: the path to the directory or file
        :param package: the classes package
        :return: all the classes
        """
        # Iterate over all files
        classes = {}
        files = os.listdir(path) if os.path.isdir(path) else [os.path.basename(path)]
        for file in files:
            # Check that the file is a python file but not the init.py
            if not file.endswith('.py') or file == '__init__.py':
                continue

            # Get the class and module
            class_name = file[:-3]
            class_module = __import__(package + class_name, fromlist=[class_name])

            # Get the frames' class
            module_dict = class_module.__dict__
            for obj in module_dict:
                if isinstance(module_dict[obj], type) and module_dict[obj].__module__ == class_module.__name__:
                    classes[obj] = getattr(class_module, obj)
        return classes

    @staticmethod
    def get_all_agents():
        """
        Getter
        :return: all the agent classes
        """
        agent_dir = os.path.abspath(os.path.dirname(__file__) + "/impl/")
        agents = AgentFactory.get_all_classes(agent_dir, "digital_gaia.fangorn.agents.impl.").values()
        agents = [agent for agent in agents if abc.ABC not in agent.__bases__]
        return [agent for agent in agents if AgentInterface in agent.__bases__]

    @staticmethod
    def get_all_interventions(data):
        """
        Getter
        :param: the data loader containing the project's information
        :return: all the project's interventions
        """
        return {value for strategy in data.project.strategies for key, value in strategy.interventions.items()}

    @staticmethod
    def display_interventions_mismatch(agent, interventions):
        """
        Display the interventions that does not match with the agent
        :param agent: the agent
        :param interventions: the required interventions
        """

        # Iterate over all the interventions
        missing_intervention = False
        for intervention in interventions:

            # Warn the user if the interventions is not compatible with the agent
            if intervention not in agent.actions:
                missing_intervention = True
                print(f"[DEBUG] The intervention {intervention}")
                print(f"[DEBUG] is not compatible with the {agent.name} agent.")
                print(f"[DEBUG]")

        # Remind the user to restart the notebook after modifying the agent class
        if missing_intervention is True:
            print(f"[DEBUG] If the above is not normal, please modify the agent and restart any running notebook.")
            print(f"[DEBUG]")

    @staticmethod
    def agents_matching_interventions(agents, interventions, verbose=False, debug=False):
        """
        Getter
        :param agents: all the agents whose compatibility must be tested
        :param interventions: the interventions that the agents must be compatible with
        :param verbose: True if useful information should be displayed, False otherwise
        :param debug: True if debug information should be displayed, False otherwise
        :return: the agents compatible with the interventions
        """

        # Filter out the agents that does not have an attribute named actions or whose interventions do not fit
        filtered_agents = []
        for agent in agents:

            # Filter out the agents that does not have an attribute named actions
            if not hasattr(agent, 'actions'):

                # Inform the user that no actions attribute was found, if required
                if verbose is True:
                    print(type(agent))
                    print(f"[ERROR] {agent.name} has no attribute named 'actions'.")

                # Skip this agent
                continue

            # Check whether the agents interventions matches the project's interventions
            intervention_matches = interventions.issubset(set(agent.actions))
            if not intervention_matches and debug is True:
                AgentFactory.display_interventions_mismatch(agent, interventions)

            # Filter out the agents whose interventions do not fit the project's interventions
            if intervention_matches:
                filtered_agents.append(agent)

        return filtered_agents

    @staticmethod
    def get_all_species(data):
        """
        Getter
        :param: the data loader containing the project's information
        :return: all the project's species
        """
        return {value for strategy in data.project.strategies for value in strategy.species}

    @staticmethod
    def display_species_mismatch(agent, all_species):
        """
        Display the species that does not match with the agent
        :param agent: the agent
        :param all_species: the required species
        """

        # Iterate over all the species
        missing_species = False
        for species in all_species:

            # Warn the user if the interventions is not compatible with the agent
            if species not in agent.species:
                missing_species = True
                print(f"[DEBUG] The species {species}")
                print(f"[DEBUG] is not compatible with the {agent.name} agent.")
                print(f"[DEBUG]")

        # Remind the user to restart the notebook after modifying the agent class
        if missing_species is True:
            print(f"[DEBUG] If the above is not normal, please modify the agent and restart any running notebook.")
            print(f"[DEBUG]")

    @staticmethod
    def agents_matching_species(agents, species, verbose=False, debug=False):
        """
        Getter
        :param agents: all the agents whose compatibility must be tested
        :param species: the species that the agents must be compatible with
        :param verbose: True if useful information should be displayed, False otherwise
        :param debug: True if debug information should be displayed, False otherwise
        :return: the agents compatible with the species
        """

        # Filter out the agents that does not have an attribute named species or whose species do not fit
        filtered_agents = []
        for agent in agents:

            # Filter out the agents that does not have an attribute named species
            if not hasattr(agent, 'species'):

                # Inform the user that no species attribute was found, if required
                if verbose is True:
                    print(f"[ERROR] {agent.name} has no attribute named 'species'.")

                # Skip this agent
                continue

            # Check whether the agents' species matches the project's species
            species_matches = species.issubset(set(agent.species))
            if not species_matches and debug is True:
                AgentFactory.display_species_mismatch(agent, species)

            # Filter out the agents whose species do not fit the project's species
            if species_matches:
                filtered_agents.append(agent)

        return filtered_agents

    @staticmethod
    def print_list(list_name, elements):
        """
        Print the list name and elements
        :param list_name: the name of the list to display
        :param elements: the elements of the list to display
        """
        print(f"[INFO] {list_name}")
        for element in elements:
            print(f"[INFO] \t- {element}")
        print("[INFO] ")

    @staticmethod
    def create(data, verbose=False, debug=False):
        """
        Create the models corresponding to the project passed as parameters
        :param data: the data loader containing all the report, policy and project description
        :param verbose: True if useful information should be displayed, False otherwise
        :param debug: True if debug information should be displayed, False otherwise
        :return: the created models
        """

        # Get all non-abstract agent classes
        agents = AgentFactory.get_all_agents()
        if verbose:
            AgentFactory.print_list("Agents found:", agents)

        # Get the project interventions
        interventions = AgentFactory.get_all_interventions(data)
        if verbose:
            AgentFactory.print_list("Project interventions:", interventions)

        # Get all agents supporting project interventions
        agents = AgentFactory.agents_matching_interventions(agents, interventions, verbose, debug)
        if verbose:
            AgentFactory.print_list("Agents satisfying interventions:", agents)

        # Get all species associated to the project strategy
        species = AgentFactory.get_all_species(data)
        if verbose:
            AgentFactory.print_list("Project species:", species)

        # Get all models supporting project species
        agents = AgentFactory.agents_matching_species(agents, species, verbose, debug)
        if verbose:
            AgentFactory.print_list("Agents satisfying interventions and species:", agents)

        # Instantiate the compatible agents
        return [agent(copy.deepcopy(data)) for agent in agents]
