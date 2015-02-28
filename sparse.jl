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

#carico e pulisco il dataset
println("Carico e pulisco il dataset ...")
tic()
#dizionari dei ratings (training => ratings)
ratings = Dict()
#dizionario degli utenti
users = Dict()
#dizionario dei programmi
programs = Dict()
#dizionario dei generi
genres = Dict()
clean_dataset!(datasetPath, ids, idTesting, ratings, users, programs, genres)
toc()

#mostro un avviso nel caso ci siano discrepanze tra il numero di programmi trovati
if length(programs) != length(ids) + length(idTesting)
  println("ATTENZIONE: nel dataset non sono stati trovati tutti gli id dei programmi!")
end

#costruisco la URM
println("Costruisco la User-Ratings Matrix ...")
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
C = compute_item_item_similarity ([ids,idTesting], programs, genres)
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
test_number = length(N)
user_number = length(users)
precision = zeros(test_number)
recall = zeros(test_number)
for i = 1:test_number
  totPrec = totRec = 0
  for u in users
    #genero lista ordinata degli spettacoli in base ai ratings dati dall utente
    ratings = spzeros(1, length(idTesting))
    for j = 1:length(idTesting)
       ratings[1,j] = URM[u[2], programs[idTesting[j]]]
    end
    orderedItems = sortperm(vec(full(ratings)), rev=true)
    #genero lista ordinata delle raccomandazioni per l utente corrente
    rec = get_recommendation(users[u[1]], idTesting, programs, URM, C, M)
    recvet = vec(full(rec))
    orderedRec = sortperm(recvet, rev=true)
    #limito i risultati ai top-N
    if length(orderedRec) > N[i]
      orderedRec = orderedRec[1:N[i]]
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

  #normalizzo i calcoli della precision e recall
  precision[i] = totPrec / user_number
  recall[i] = totRec / user_number
end
toc()

#stampo statistiche
println("-----------------------------------------------------------------------------")
println("Numero programmi di training: $(length(ids))")
println("Numero programmi di testing: $(length(idTesting))")
println("Numero di utenti: $user_number")

#Stampo Risultati
println("-----------------------------------------------------------------------------")
for i = 1:test_number
  println("Precision@$(N[i]) = $(precision[i])\nRecall@$(N[i]) = $(recall[i])")
end
