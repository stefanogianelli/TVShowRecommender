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
alpha = 0.001
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
#rimuovo i programId che sono già presenti in quelli di testing
idTesting = setdiff(idTesting, intersect(idTesting, ids))
toc()

#carico il dataset
println("Carico il dataset ...")
tic()
dataset = int(readdlm(datasetPath, ',', use_mmap=true))
toc()

#pulisco il dataset
println("Pulisco il dataset ...")
tic()
ratings = Dict()
programs = Dict()
#genres = Dict()
countProg = 1
users = Dict()
countUser = 1
for i = 1:size(dataset)[1]
  if (dataset[i,3] != 14 && dataset[i,3] != 19)
    if (in(dataset[i,7], ids))
      try
        ratings[dataset[i,6], dataset[i,7]] += dataset[i,9]
      catch
        ratings[dataset[i,6], dataset[i,7]] = dataset[i,9]
      end
      if (!in(dataset[i,6], keys(users)))
        users[dataset[i,6]] = countUser
        countUser += 1
      end
      if (!in(dataset[i,7], keys(programs)))
        programs[dataset[i,7]] = countProg
        countProg += 1
      end
    end
    #genres[dataset[i,7]] = (dataset[i,4], dataset[i,5])
  end
end
#aggiungo gli id dei programmi di test
for id in idTesting
  programs[id] = countProg
  countProg += 1
end
if (length(programs) != length(ids) + length(idTesting))
  println("ATTENZIONE: nel dataset non sono stati trovati tutti gli id dei programmi!")
end
toc()

#costruisco la URM
println("Costruisco la URM ...")
tic()
URM = spzeros(length(users), length(ids))
for r in ratings
  URM[users[r[1][1]], programs[r[1][2]]] = r[2]
end
toc()

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

rec1 = getRecommendation(2203)

println("Fine.")
