using Documenter
using MirrorVI

makedocs(
    sitename = "MirrorVI.jl Documentation",
    pages = ["Index" => "index.md"],
    format = Documenter.HTML(prettyurls = false),
    modules = [MirrorVI]
    # checkdocs = :exports
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/marcoelba/MirrorVI.jl.git",
    devbranch = "main"
)
