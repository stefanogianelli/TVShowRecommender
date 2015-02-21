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
N = 10

#=
MAIN
=#

#ricavo il percorso base da cui caricare i dataset
cd(dirname(@__FILE__))

#carico la matrice con le informazioni sui programmi di training
println("Carico gli id dei programmi di training")
tic()
ids = load_program_ids(trainingPath)
toc()

#carico la matrice con le informazioni sui programmi di testing
println("Carico gli id dei programmi di testing")
tic()
idTesting = load_program_ids(testingPath)
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
testingUsers = Dict()
#dizionario dei programmi
programs = Dict()
clean_dataset!(dataset, ids, idTesting, ratings, testingRatings, users, testingUsers, programs)
toc()

#mostro un avviso nel caso ci siano discrepanze tra il numero di programmi trovati
if length(programs) != length(ids) + length(idTesting)
  println("ATTENZIONE: nel dataset non sono stati trovati tutti gli id dei programmi!")
end

#mostro un avviso nel caso non esistano utenti in comune tra training e testing
if length(intersect(keys(users), keys(testingUsers))) == 0
  println("ATTENZIONE: non esistono utenti confrontabili!")
end

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
URMT = spzeros(length(testingUsers), length(programs))
for r in testingRatings
  URMT[testingUsers[r[1][1]], programs[r[1][2]]] = r[2]
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
C = compute_item_item_similarity (dataset, [ids,idTesting])
toc()

#eseguo i calcoli per il gradiente che non devono essere rifatti ogni volta
Q = transpose(C) * S * C
T = transpose(C) * C

#calcolo la matrice M
println("Calcolo la matrice M ottimale ...")
tic()
M = gradient_descent (alpha, length(programs), S, C, Q, T)
toc()

#cerco le raccomandazioni per tutti gli utenti
println("Valuto l'efficienza dell'algoritmo ...")
tic()
totPrec = totRec = 0
count = 0
for u in testingUsers
  #verifico che l'utente esista nella lista degli utenti di training
  if in(u[1], keys(users))
    count += 1
    #genero lista ordinata degli spettacoli in base ai ratings dati dall utente
    ratings = vec(dense(URMT[u[2],:]))
    orderedItems = sortperm(ratings, rev=true)
    #genero lista ordinata delle raccomandazioni per l utente corrente
    rec = get_recommendation(users[u[1]], idTesting, programs, URM, C, M)
    recvet = vec(full(rec))
    orderedRec = sortperm(recvet, rev=true)
    #limito i risultati ai top-N
    if length(orderedItems) > N
      orderedItems = orderedItems[1:N]
      orderedRec = orderedRec[1:N]
    end
    #calcolo gli insiemi True Positive, False Positive e False Negative
    #reference: http://www.kdnuggets.com/faq/precision-recall.html
    TP = length(intersect(orderedItems, orderedRec))
    FP = length(setdiff(orderedRec, orderedItems))
    FN = length(setdiff(orderedItems, orderedRec))
    #calcolo precision
    #reference: http://en.wikipedia.org/wiki/Precision_and_recall#Definition_.28classification_context.29
    totPrec += TP / (TP + FP)
    #calcolo recall
    #reference: http://en.wikipedia.org/wiki/Precision_and_recall#Definition_.28classification_context.29
    totRec += TP / (TP + FN)
  end
end

#normalizzo i calcoli della precision e recall
endPrec = totPrec / count
endRec = totRec / count
toc()

#stampo statistiche
println("-----------------------------------------------------------------------------")
println("Numero programmi di training: $(length(ids))")
println("Numero programmi di testing: $(length(idTesting))")
println("Numero di utenti: $(length(users))")
println("Numero utenti di testing: $(length(testingUsers))")
println("Numero di utenti in comune: $count")

#Stampo Risultati
println("-----------------------------------------------------------------------------")
println("Precision@$N = $endPrec\nRecall@$N = $endRec")
