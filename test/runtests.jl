# runtests
include(joinpath("test_model_building", "test_bijectors_extension.jl"))
include(joinpath("test_model_building", "test_distributions_logpdf.jl"))
include(joinpath("test_model_building", "test_mirror_statistic.jl"))
include(joinpath("test_model_building", "test_model_prediction_functions.jl"))
include(joinpath("test_model_building", "test_my_optimisers.jl"))
include(joinpath("test_model_building", "test_normal_degenerate.jl"))
include(joinpath("test_model_building", "test_variational_distributions.jl"))
