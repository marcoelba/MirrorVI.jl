using Documenter
using MirrorVI

makedocs(
    sitename = "MirrorVI",
    format = Documenter.HTML(),
    modules = [MirrorVI]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
