include("functions.jl")

#=
PARAMETRI
=#

#permette di scegliere la cartella
dir = ".\\dataset"
#dir = "C:\\Users\\Stefano\\Desktop\\tvshow"
#percorso dati di training
trainingPath = "$dir\\training.txt"
#percorso dati di testing
testingPath = "$dir\\testing.txt"
#percorso dataset
datasetPath = "$dir\\data.txt"
#tolleranza (SGD)
tol = 1e-8
#numero massimo iterazioni (SGD)
miter = 1000
#dimensione passo (SGD)
alpha = 0.001
#incremento percentuale del learning rate (SGD)
deltaAlpha = 5
#numero item simili
N = [1, 5, 10, 20, 30, 40, 50]

#=
MAIN
=#

#ricavo il percorso base da cui caricare i dataset
cd(dirname(@__FILE__))

#carico la matrice con le informazioni sui programmi di testing
println("Carico gli id dei programmi di testing")
tic()
idTesting = load_program_ids(testingPath)
toc()

#carico la matrice con le informazioni sui programmi di training
println("Carico gli id dei programmi di training")
tic()
ids = load_program_ids(trainingPath)
#rimuovo i programId che sono già presenti in quelli di testing
ids = setdiff(ids, intersect(idTesting, ids))
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
testingUsers = Dict()
#dizionario dei programmi
programs = Dict()
clean_dataset1!(dataset, ids, idTesting, ratings, testingRatings, users, testingUsers, programs)
toc()

#costruisco la URM di training
println("Costruisco la URM di Training ...")
tic()
URM = spzeros(length(users), length(programs))
for r in ratings
  URM[users[r[1][1]], programs[r[1][2]]] = r[2]
end
toc()

#calcolo la matrice S tramite adjusted cosine similarity
println("Calcolo la matrice S ...")
tic()
S = build_similarity_matrix (programs, ids, URM)
toc()

#calcolo la matrice C
println("Calcolo la matrice C")
tic()
C = compute_item_item_similarity (dataset, ids)
toc()

#eseguo i calcoli per il gradiente che non devono essere rifatti ogni volta
Q = transpose(C) * S * C
T = transpose(C) * C

#calcolo la matrice M
println("Calcolo la matrice M ottimale ...")
tic()
M = gradient_descent1 (alpha, length(programs), S, C, Q, T)
toc()

function clean_dataset1! (dataset::Matrix, ids::Array, idTesting::Array, ratings::Dict, testingRatings::Dict, users::Dict, testingUsers::Dict, programs::Dict)
  #inizializzo contattori
  countUser = 1
  countTestUser = 1
  countProg = 1
  #scansiono tutto il dataset
  for i = 1:size(dataset)[1]
    #elimino le settimne 14 e 19
    if (dataset[i,3] != 14 && dataset[i,3] != 19)
      #controllo se l'id corrente è nell'insieme degli id di training
      if (in(dataset[i,7], ids))
        try
          ratings[dataset[i,6], dataset[i,7]] += dataset[i,9]
        catch
          ratings[dataset[i,6], dataset[i,7]] = dataset[i,9]
        end
        #aggiungo l'utente corrente
        if (!in(dataset[i,6], keys(users)))
          users[dataset[i,6]] = countUser
          countUser += 1
        end
        #aggiungo il programma corrente
        if (!in(dataset[i,7], keys(programs)))
          programs[dataset[i,7]] = countProg
          countProg += 1
        end
      #controllo se l'id corrente è nell'insieme degli id di testing
      elseif (in(dataset[i,7], idTesting))
        try
          testingRatings[dataset[i,6], dataset[i,7]] += dataset[i,9]
        catch
          testingRatings[dataset[i,6], dataset[i,7]] = dataset[i,9]
        end
        #aggiungo l'utente corrente
        if (!in(dataset[i,6], keys(testingUsers)))
          testingUsers[dataset[i,6]] = countTestUser = 1
          countTestUser += 1
        end
        #aggiungo il programma corrente
        #=if (!in(dataset[i,7], keys(programs)))
          programs[dataset[i,7]] = countProg
          countProg += 1
        end=#
      end
    end
  end
end

#Calcola la matrice M ottimale attraverso SGD
function gradient_descent1 (a::Number, MSize::Int, S::SparseMatrixCSC, C::SparseMatrixCSC, Q::SparseMatrixCSC, T::SparseMatrixCSC)
  #inizializzo la matrice M
  M = Mnew = spzeros(MSize, MSize)
  fval = object(M, S, C)
  gain = 1
  println("Start value = $fval")
  #calcolo la matrice M ottimale
  i = 1
  while fval > tol && gain > tol
    Mnew = M - (a / MSize) * grad(M, T, Q)
    f = object(Mnew, S, C)
    if f <= fval
      gain = abs(fval - f)
      M = Mnew
      fval = f
      a += (a * deltaAlpha / 100)
    else
      a /= 2
    end
    i += 1
  end
  println("End value = $fval")
  return M
end
