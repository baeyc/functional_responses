
from typing import NamedTuple

import parametrization_cookbook.jax as pc
from parametrization_cookbook.functions.jax import expit

class EstimationDescription(NamedTuple):
    """
    NamedTuple Object that describes the estimation process. 
    Attributes : 
        indiv_model_parameters : tuple of string that describes the aprameter to consider variable between indivudals
        population_model_parameters : tuple of string that describes the population parameters
        outputs
    """
    indiv_model_parameters: tuple
    population_model_parameters: tuple    

    @property
    def nindiv(self):
        return len(self.indiv_model_parameters)

    @property
    def npop(self):
        return len(self.population_model_parameters)

random_eff = "lambda_h"

if random_eff == "lambda_h":
    estimation_description = EstimationDescription(
        indiv_model_parameters=("lambda","h"),
        population_model_parameters=(),    
    )
elif random_eff == "lambda":
    estimation_description = EstimationDescription(
        indiv_model_parameters=("lambda",),
        population_model_parameters=("h",),    
    )
elif random_eff == "h":
    estimation_description = EstimationDescription(
        indiv_model_parameters=("h",),
        population_model_parameters=("lambda",),    
    )
else:
    estimation_description = EstimationDescription(
        indiv_model_parameters=(),
        population_model_parameters=("lambda","h",),    
    )

