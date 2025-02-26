module MirrorVI


include(joinpath("posterior_utils.jl"))
include(joinpath("model_building", "training_utils.jl"))
include(joinpath("model_building", "utils.jl"))

include(joinpath("model_building", "my_optimisers.jl"))
export MyOptimisers

include(joinpath("model_building", "model_prediction_functions.jl"))

include(joinpath("model_building", "mirror_statistic.jl"))

include(joinpath("plot_utils.jl"))
include(joinpath("mixed_models_data_generation.jl"))

include(joinpath("model_building", "bijectors_extension.jl"))
include(joinpath("model_building", "normal_degenerate.jl"))
include(joinpath("model_building", "variational_distributions.jl"))
include(joinpath("model_building", "distributions_logpdf.jl"))


end # module MirrorVI
