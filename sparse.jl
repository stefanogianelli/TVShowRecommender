include("functions.jl")

#=
PARAMETRI
=#

#permette di scegliere la cartella
dir = ".\\dataset"
#percorso dati di training
trainingPath = "$dir\\training.txt"
#percorso dati di testing
testingPath = "$dir\\testing.txt"
#percorso dataset
datasetPath = "$dir\\data.txt"
#tolleranza (SGD)
tol = 1e-6
#numero massimo iterazioni (SGD)
miter = 1000
#dimensione passo (SGD)
alpha = 0.0001
#incremento percentuale del learning rate (SGD)
deltaAlpha = 5
#numero item simili (calcolo R)
N = 5

#=
MAIN
=#

#ricavo il percorso base da cui caricare i dataset
cd(dirname(@__FILE__))

#carico la matrice con le informazioni sui programmi di training
println("Carico gli id dei programmi di training")
tic()
ids = loadProgramIds(trainingPath)
toc()

#carico la matrice con le informazioni sui programmi di testing
println("Carico gli id dei programmi di testing")
tic()
idTesting = loadProgramIds(testingPath)
toc()

#carico il dataset
println("Carico il dataset ...")
tic()
dataset = int(readdlm(datasetPath, ',', use_mmap=true))
toc()

#creo un vettore con la lista degli utenti
println("Salvo la lista degli utenti univoci ...")
tic()
users = unique(dataset[:,6])
toc()

URM = sparse(dataset[:,6], dataset[:,7], dataset[:,9])

#calcolo la matrice S tramite adjusted cosine similarity
println("Calcolo la matrice S ...")
tic()
S = buildS()
toc()

#calcolo la matrice C
println("Calcolo la matrice C")
tic()
C = computeItemItemSim(dataset, [ids,idTesting])
toc()

#eseguo i calcoli per il gradiente che non devono essere rifatti ogni volta
Q = transpose(C) * S * C
T = transpose(C) * C

#calcolo la matrice M
println("Calcolo la matrice M ottimale ...")
tic()
M = gradientDescent()
toc()

#costruisco la matrice con i ratings per i programmi futuri
println("Calcolo la matrice R dei ratings per i programmi futuri ...")
tic()
R = buildR()
toc()

println("Fine.")

R
