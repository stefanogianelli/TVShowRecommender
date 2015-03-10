include("functions.jl")

#=
PARAMETRI
=#

#permette di scegliere la cartella
#dir = ".\\dataset"
dir = "C:\\Users\\Stefano\\Desktop\\tvshow"
#percorso dati di training
trainingPath = "$dir\\training.txt"
#percorso dati di testing
testingPath = "$dir\\testing.txt"
#percorso dataset
datasetPath = "$dir\\data.txt"
#tolleranza (SGD)
tol = 1e-5
#numero massimo iterazioni (SGD)
miter = 1000
#dimensione passo (SGD)
alpha = 0.001
#incremento percentuale del learning rate (SGD)
deltaAlpha = 5
#numero item simili
N = [1, 5, 10, 20]
#soglia affinchè un rating sia considerato rilevante
relevant_threshold = 0.6

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
#dizionari dei ratings
ratings = Dict()
#dizionario degli utenti
users = Dict()
test_users = Int64[]
#dizionario dei programmi
programs = Dict()
#dizionario dei generi
genres = Dict()
clean_dataset!(datasetPath, ids, idTesting, ratings, users, test_users, programs, genres)
toc()

#mostro un avviso nel caso ci siano discrepanze tra il numero di programmi trovati
if length(programs) != length(ids) + length(idTesting)
  println("ATTENZIONE: nel dataset non sono stati trovati tutti gli id dei programmi! Correggo il problema ...")
  ids = intersect(ids, keys(programs))
  idTesting = intersect(idTesting, keys(programs))
  println("Problema risolto!")
end

#stampo statistiche
println("-----------------------------------------------------------------------------")
println("Numero programmi di training: $(length(ids))")
println("Numero programmi di testing: $(length(idTesting))")
println("Numero di utenti: $(length(users))")
println("Numero di utenti di test: $(length(test_users))")
println("-----------------------------------------------------------------------------")

#costruisco la URM di training
println("Costruisco la User-Rating Matrix di training ...")
tic()
URM = spzeros(length(users), length(programs))
for r in ratings
  if in(r[1][2], ids)
    URM[users[r[1][1]], programs[r[1][2]]] = r[2]
  end
end
toc()

#costruisco la URM di testing
println("Costruisco la User-Rating Matrix di testing ...")
tic()
URM_Test = spzeros(length(users), length(programs))
for r in ratings
  if in(r[1][2], idTesting)
    URM_Test[users[r[1][1]], programs[r[1][2]]] = r[2]
  end
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
user_number = length(test_users)
precision = zeros(test_number)
recall = zeros(test_number)
stop = false
for i = 1:test_number
  #non ripeto il test se non necessario
  if !stop
    println("Test #$i @ $(N[i]) ...")
    #inizializzo variabili
    avg_rating = 0
    count_ratings = 0
    totPrec = totRec = 0
    #esegue il calcolo per tutti gli utenti
    for u in test_users
      #genero lista degli spettacoli in base ai ratings dati dall'utente (dalla URM)
      items = Dict()
      for p in idTesting
        val = URM_Test[users[u], programs[p]]
        items[p] = val
        avg_rating += val
        count_ratings += 1
      end
      #genero lista delle raccomandazioni per l'utente corrente
      rec_vec = get_recommendation(users[u], idTesting, programs, URM, C, M)
      ordered_rec = sortperm(rec_vec, rev=true)
      #limito i risultati ai top-N
      rec_size = min(length(rec_vec), N[i])
      #imposto blocco nel caso il vettore sia più corto della N corrente
      if rec_size == length(rec_vec)
        stop = true
      end
      rec = Dict()
      for j = 1:rec_size
        index = ordered_rec[j]
        rec[idTesting[index]] = rec_vec[index]
        avg_rating += rec_vec[index]
        count_ratings += 1
      end
      #creo una soglia di rilevanza
      avg_rating /= count_ratings
      threshold = avg_rating * relevant_threshold
      #creo insiemi di elementi rilevanti
      cond_positive = Int64[]
      cond_negative = Int64[]
      for k in keys(items)
        if items[k] >= threshold
          push!(cond_positive, k)
        else
          push!(cond_negative, k)
        end
      end
      #creo insiemi di raccomandazioni rilevanti
      test_positive = Int64[]
      test_negative = Int64[]
      for k in keys(rec)
        if rec[k] >= threshold
          push!(test_positive, k)
        else
          push!(test_negative, k)
        end
      end
      #calcolo gli insiemi True Positive, False Positive e False Negative
      #reference: http://www.kdnuggets.com/faq/precision-recall.html
      TP = length(intersect(cond_positive, test_positive))
      FP = length(intersect(cond_negative, test_positive))
      FN = length(intersect(cond_positive, test_negative))
      PPV = TP + FP
      TPR = TP + FN
      #calcolo precision
      #reference: http://en.wikipedia.org/wiki/Precision_and_recall#Definition_.28classification_context.29
      if PPV != 0
        totPrec += TP / PPV
      end
      #calcolo recall
      #reference: http://en.wikipedia.org/wiki/Precision_and_recall#Definition_.28classification_context.29
      if TPR != 0
        totRec += TP / TPR
      end
    end

    #normalizzo i calcoli della precision e recall
    precision[i] = totPrec / user_number
    recall[i] = totRec / user_number
  end
end
toc()

#Stampo Risultati
println("-----------------------------------------------------------------------------")
println("Evaluation Results")
for i = 1:test_number
  println("Precision@$(N[i]) = $(precision[i])\nRecall@$(N[i]) = $(recall[i])")
end
println("-----------------------------------------------------------------------------")
