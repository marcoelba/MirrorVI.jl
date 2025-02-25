using Documenter
using MirrorVI

makedocs(
    sitename = "MirrorVI",
    format = Documenter.HTML(),
    modules = [MirrorVI],
    checkdocs = :exports
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/marcoelba/MirrorVI.jl",
    devbranch = "main"
)
