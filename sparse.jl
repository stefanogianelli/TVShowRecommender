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
#numero item simili
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
#rimuovo i programId che sono già presenti in quelli di training
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
#dizionari dei ratings (training => ratings, testing => testingRatings)
ratings = Dict()
testingRatings = Dict()
#dizionario degli utenti
users = Dict()
countUser = 1
#dizionario dei programmi
programs = Dict()
countProg = 1
#dizionario dei generi
#genres = Dict()
#scansiono tutto il dataset
for i = 1:size(dataset)[1]
  #elimino le settimne 14 e 19
  if (dataset[i,3] != 14 && dataset[i,3] != 19)
    #controllo se l id corrente è nell insieme degli id di training
    if (in(dataset[i,7], ids))
      try
        ratings[dataset[i,6], dataset[i,7]] += dataset[i,9]
      catch
        ratings[dataset[i,6], dataset[i,7]] = dataset[i,9]
      end
      #aggiungo l utente corrente
      if (!in(dataset[i,6], keys(users)))
        users[dataset[i,6]] = countUser
        countUser += 1
      end
      #aggiungo il programma corrente
      if (!in(dataset[i,7], keys(programs)))
        programs[dataset[i,7]] = countProg
        countProg += 1
      end
    #controllo se l id corrente è nell insieme degli id di testing
    elseif (in(dataset[i,7], idTesting))
      try
        testingRatings[dataset[i,6], dataset[i,7]] += dataset[i,9]
      catch
        testingRatings[dataset[i,6], dataset[i,7]] = dataset[i,9]
      end
      #aggiungo l utente corrente
      if (!in(dataset[i,6], keys(users)))
        users[dataset[i,6]] = countUser
        countUser += 1
      end
      #aggiungo il programma corrente
      if (!in(dataset[i,7], keys(programs)))
        programs[dataset[i,7]] = countProg
        countProg += 1
      end
    end
  end
  #creo un dizionario con i generi e sottogeneri dei programmi
  #=if (!in(dataset[i,7], keys(genres)))
    genres[dataset[i,7]] = (dataset[i,4], dataset[i,5])
  end=#
end
if (length(programs) != length(ids) + length(idTesting))
  println("ATTENZIONE: nel dataset non sono stati trovati tutti gli id dei programmi!")
end
toc()

#costruisco la URM di training
println("Costruisco la URM di Training ...")
tic()
URM = spzeros(length(users), length(programs))
for r in ratings
  URM[users[r[1][1]], programs[r[1][2]]] = r[2]
end
toc()

#costruisco la URM di testing
println("Costruisco la URM di Testing ...")
tic()
URMT = spzeros(length(users), length(programs))
for r in testingRatings
  URMT[users[r[1][1]], programs[r[1][2]]] = r[2]
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

#cerco le raccomandazioni per tutti gli utenti
for u in users
  rec = getRecommendation(u[2])
  println("$(u[1]) : $rec")
end

println("Fine.")
