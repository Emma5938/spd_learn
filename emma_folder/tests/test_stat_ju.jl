using Pkg                           # Installer le package
# Pkg.add("PermutationTests")
# Pkg.add("Plots")
using DelimitedFiles                # Charger les packages
using PermutationTests
using Plots
using Random
using Statistics

#%%
# Charger le fichier des résultats
y = readdlm("results_05_06_BNCI2015.csv", ',', skipstart=1)                    # skipstart=1 signifie qu'on enlève l'entête

# Vérificatione la taille de y (doit être 25xnbr de couches)
println("Taille de y : ", size(y))

# Transformer y en vecteur 
y_vec = vec(permutedims(y))
#y_vec = vcat(y...)
# Vérification : y doit etre de taille 25*nbr_couches
println("Longueur y_vec : ", length(y_vec)) 
println("Première ligne y :")
println(y[1,:])

println("Premier bloc y_vec :")
println(y_vec[1:4])

#%%
#---------------------------------------
# TEST OMNIBUS
#---------------------------------------
res = anovaTestRM(y_vec, (n=25, k=4))
println("\n=== RESULTAT ANOVA ===")
println(res)


#%%
#---------------------------------------
# TEST POST HOC
#---------------------------------------
# Paramètres 
N = size(y, 1)                                             # nombre de runs = 25
K = size(y, 2)                                             # nombre de couches = 4

# Construction des différences 
NK = N * K

println(NK)

d12 = y_vec[1:K:NK] .- y_vec[2:K:NK]                        # ReEig - Cosh [start, step, end]
d13 = y_vec[1:K:NK] .- y_vec[3:K:NK]                        # ReEig - CoshP
d14 = y_vec[1:K:NK] .- y_vec[4:K:NK]                        # ReEig - ExpAct
d23 = y_vec[2:K:NK] .- y_vec[3:K:NK]                        # Cosh - CoshP
d24 = y_vec[2:K:NK] .- y_vec[4:K:NK]                        # Cosh - ExpAct
d34 = y_vec[3:K:NK] .- y_vec[4:K:NK]                        # CoshP - ExpAct

# Test
pht = studentMcTestRM([d12, d13, d14, d23, d24, d34])

println("\n=== RESULTATS POST-HOC ===")
println(pht.p)
println(pht.obsstat)
println("moyenne de ReEig : ", mean(y[:,1]), " +/- ", std(y[:,1]))
println("moyenne de Cosh : ", mean(y[:,2]), " +/- ", std(y[:,2]))
println("moyenne de CoshP : ", mean(y[:,3]), " +/- ", std(y[:,3]))
println("moyenne de ExpAct : ", mean(y[:,4]), " +/- ", std(y[:,4]))

#%% Bar plot moyennes et écarts types

gr()

moyennes = [mean(y[:,1]), mean(y[:,2]), mean(y[:,3]), mean(y[:,4])]
ecarts   = [std(y[:,1]), std(y[:,2]), std(y[:,3]), std(y[:,4])]

labels = ["ReEig", "Cosh", "CoshP", "ExpT"]
x = 1:length(labels)

p = bar(
    x, moyennes,
    xticks=(x, labels),
    yerr=ecarts,
    legend=false,
    size=(700,400),
    color=:steelblue,
    guidefont=font(12, "Times"),
    tickfont=font(12, "Times")
)

ymin = minimum(moyennes .- ecarts)
ymax = maximum(moyennes .+ ecarts)
margin = 0.05 * (ymax - ymin)

ylims!(ymin - margin, ymax + margin)

offset = 0.03 * (ymax - ymin)

for i in x
    texte = string(round(moyennes[i], digits=3),
                   " ± ",
                   round(ecarts[i], digits=3))

    annotate!(
        p,
        i,
        moyennes[i] + ecarts[i] + offset,
        text(texte, 12, "Times")
    )
end

display(p)


#%%
#---------------------------------------
# DONNEES FICTIVES
#---------------------------------------
# Paramètres
N = 25
K = 3
n_iter = 10000

pvals = zeros(n_iter)

# Test pour H0 
for i in 1:n_iter

    # H0 vraie : aucune différence entre colonnes
    y = randn(N, K)

    # Format pour le test 
    y_vec = vec(permutedims(y))

    # ANOVA 
    res = anovaTestRM(y_vec, (n=N, k=K))

    # stocker p-value
    pvals[i] = res.p
end

# Taux de rejet 
# ---------------------------
println("Empirical rejection rate (alpha=0.05): ",
        mean(pvals .< 0.05))

# HISTOGRAMME
# ---------------------------
histogram(
    pvals,
    bins = 20,
    normalize = true,
    label = "Empirical p-values",
    xlabel = "p-value",
    ylabel = "Density",
    title = "Distribution of p-values under H0"
)

# ligne théorique uniforme
plot!([0,1], [1,1],lw=2, linestyle=:dash,label="Uniform(0,1)")

#%%

# Paramètres
N = 25
K = 3
n_iter = 10000

pvals = zeros(n_iter)

# Test pour H1 
for i in 1:n_iter

    # H1 vraie : différence entre colonnes
    y = randn(N, K)

    # une couche meilleure
    y[:,3] .+= 0.5

    # Format pour le test 
    y_vec = vec(permutedims(y))

    # ANOVA 
    res = anovaTestRM(y_vec, (n=N, k=K))

    # stocker p-value
    pvals[i] = res.p
end

# Taux de rejet 
# ---------------------------
println("Empirical power (alpha=0.05): ",
        mean(pvals .< 0.05))

# HISTOGRAMME
# ---------------------------
histogram(
    pvals,
    bins = 20,
    normalize = true,
    label = "H1 p-values",
    xlabel = "p-value",
    ylabel = "Density",
    title = "Distribution of p-values under H1"
)

