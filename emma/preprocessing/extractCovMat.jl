# This script extracts covariance matrices (files ..._covmats.npy)
# and associated labels (files ..._labels.npy) from selected motor imagery (MI)
# databases of the Federico II corpus (FII corpus, see:
# https://marco-congedo.github.io/Eegle.jl/dev/documents/BCI%20Databases%20Overview/).

# Databases are selected so as to comprise the chosen classes
# (for example "left_hand" and "right_hand") and a minimum number of trials per class.
# The files are saved in NumPy format (.npy).

# The covariance matrices are saved as a tensor nxnxk, where n is the number of electrodes
# and k the number of trials. The associated labels follows this convention:
# left_hand → 1, right_hand → 2, feet → 3, rest → 4, both_hands → 5, tongue → 6

# Before running this script execute the following line in the REPL:
# ]add NPZ Eegle


using NPZ
using Eegle

# $ cd /localdata/costamai/Apps/LibData/NY/Extracted
# $ julia
# $ include("/localdata/costamai/Apps/LibData/NY/extractCovMat.jl")
#They should be at /localdata/costamai/Apps/LibData/NY/Extracted
#(we can find them doing (without changing dir they will be at /nethome/costamai))
# $ find /localdata/costamai -type f -name '*_covmats.npy' -exec ls -lh {} \;

# PUT HERE path to the MI folder of the FII corpus on your PC
# MIDir = joinpath(@__DIR__, "MI") 
MIDir = "C:/Users/andrieue/Desktop/Données/FII_BCI_Corpus/MI"
classes = ["left_hand", "right_hand"]; # or for example ["left_hand", "right_hand"]; ["feet",]   # Chosen classes

# select MI databases comprising the given 'classes' and minimum number of trials
inclusion = (("tpc", x -> minimum(values(x)) > 24),)

DBs = selectDB(MIDir, :MI; classes, inclusion);

outDir = "C:/Users/andrieue/Desktop/Données/FII_BCI_Corpus/MatCov"              # output to save the covariance matrices

# Create and save all data
for DB ∈ DBs
    @info "\nwriting data for database $(DB.dbName)"

    for (d, file) in enumerate(DB.files)

        println("writing file $d of $(length(DB.files))")

        o = readNY(file; bandPass = (8, 32), upperLimit = 1.2, classes) # read session
        C = encode(o; covtype=SCM) # BCI Riemannian Encoding, see Eegle.jl

        npzwrite(joinpath(outDir,"$(DB.dbName)_$(d)_covmats.npy"), cat(C...; dims=3))
        npzwrite(joinpath(outDir,"$(DB.dbName)_$(d)_labels.npy"), o.y)

    end
end
@info "Finished!"
