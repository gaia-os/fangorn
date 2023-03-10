from digital_gaia.fangorn.assessment.Engine import Engine as AssessmentEngine
import digital_gaia.fangorn as fangorn
from os.path import abspath, dirname
from datetime import timedelta
from digital_gaia.fangorn.pydantic.Assessment import Assessment


def save(project_start_date, assessment, saving_directory, saving_file):
    """
    Save the results on the filesystem
    :param project_start_date: the date at which the project started
    :param assessment: the project assessment
    :param saving_directory: the directory in which the results should be saved
    :param saving_file: the file in which the results should be saved
    """

    # Iterates over all agents
    for agent_name, results in assessment.items():

        # Iterates over time steps
        for t, (inference_samples, prediction_samples, efe) in enumerate(results):

            # Create the path where the project assessment will be stored
            full_saving_path = f"{saving_directory}/{agent_name}/{t}/{saving_file}"

            # Create the beliefs and predictions

            print(inference_samples.keys())
            print(type(inference_samples))
            print("hemp_size: ", inference_samples["hemp_size"].shape)
            print("hemp_size_0: ", inference_samples["hemp_size_0"].shape)
            print(prediction_samples.keys())
            print(type(prediction_samples))
            print("hemp_size: ", prediction_samples["hemp_size"].shape)
            print("hemp_size_0: ", prediction_samples["hemp_size_0"].shape)
            beliefs = {}  # TODO beliefs: Dict[str, Belief]
            predictions = {}  # TODO predictions: Dict[str, Belief]

            # Create the pydantic assessment
            pydantic_assessment = Assessment(**{
                "assessment_date": project_start_date + timedelta(days=7),
                "efe": efe,
                "beliefs": beliefs,
                "predictions": predictions
            })

            # Write the pydantic assessment on the file system
            with open(full_saving_path, "w") as f:
                f.write(pydantic_assessment.json(indent=2))


def perform_assessment():
    """
    Performing the project assessment
    :return: a dictionary mapping model name to the model results, where the model results is a list of tuple
    containing the samples and expected free energy of the assessment at each time step
    """

    # Create the assessment engine and perform the assessment
    engine = AssessmentEngine(verbose=True, debug=True)
    assessment = engine.perform_assessment()

    # Save the results in a json file
    fangorn_dir = dirname(dirname(dirname(abspath(fangorn.__file__))))
    project_start_date = engine.data.project.start_date
    save(project_start_date, assessment, f"{fangorn_dir}/data/results/", "roots-and-culture.json")


if __name__ == '__main__':
    # Entry point performing project assessment
    perform_assessment()
