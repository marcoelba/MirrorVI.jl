using Documenter
using MirrorVI

push!(LOAD_PATH,"../src/")
makedocs(
    modules = [MirrorVI, MyOptimisers],
    sitename = "MirrorVI.jl",
    format = Documenter.HTML()
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/marcoelba/MirrorVI.jl.git",
    devbranch = "main"
)
