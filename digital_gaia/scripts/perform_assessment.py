from digital_gaia.fangorn.assessment.Engine import Engine as AssessmentEngine
import digital_gaia.fangorn as fangorn
from os.path import abspath, dirname


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
    engine.save(f"{fangorn_dir}/data/assessments", assessment)


if __name__ == '__main__':
    # Entry point performing project assessment
    perform_assessment()
