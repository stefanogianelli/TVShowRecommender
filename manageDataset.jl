#ricavo il percorso base da cui caricare i dataset
#N.B.: pwd ritorna la cartella utente!
path = string(pwd(),"\\TVShowRecommender")

#carico la matrice con le informazioni sui programmi di training
trainingInfo = readdlm("$path\\dataset\\training.txt", '\t', use_mmap=true)

#considero solo le colonne programId e start
trainingInfo = trainingInfo[:,[2,4]]

#cerco e salvo i programId di 8 giorni consecutivi
#lo script legge tutti i programmi presenti nel file, è necessario dunque filtrare gli eventi di interesse manualmente
ids = Int64[]

for i=1:size(trainingInfo)[1]
  #verifico se il programId corrente non sia già stato inserito
  if !in(trainingInfo[i,1], ids)
    push!(ids, trainingInfo[i,1])
  end
end

#carico il dataset
dataset = readdlm("$path\\dataset\\data.txt", ',', use_mmap=true)

#creo una copia per la tabella dei ratings
ratingsTable = dataset

#considero solo le colonne: weekIdx, userIdx, programIdx, duration
ratingsTable = ratingsTable[:,[3,6,7,9]]

#rimuovo i programIdx duplicati e sommo le loro durate
#considero solo i programmi selezionati nel vettore ids
#vengono inoltre rimosse le settimane 14 e 19
ratingsTableSize = size(ratingsTable)[1]
i = 1
while i <= ratingsTableSize
  #non considero il programma se siamo nella 14esima o 19esima settimana
  if ratingsTable[i,1] != 14 && ratingsTable[i,1] != 19
    #verifico che il programId corrente sia presente nel vettore ids
    if in(ratingsTable[i,3], ids)
      #verifico se la coppia (userId, programId) non sia già presente
      index = findExistingRating(ratingsTable[i,2], ratingsTable[i,3], i-1)
      if index != -1
        #se la coppia (userId, programId) esiste viene sommato il valore della duration
        ratingsTable[index,4] += ratingsTable[i,4]
        ratingsTable = ratingsTable[[1:(i-1), (i+1):end], :]
        ratingsTableSize -= 1
      else
        #altrimenti passo alla riga successiva
        i += 1
      end
    else
      #rimuovo la riga
      ratingsTable = ratingsTable[[1:(i-1), (i+1):end], :]
      ratingsTableSize -= 1
    end
  else
    #rimuovo la riga
    ratingsTable = ratingsTable[[1:(i-1), (i+1):end], :]
    ratingsTableSize -= 1
  end
end

#rimuovo la colonna della settimana, non più necessaria
ratingsTable = ratingsTable[:,[2,3,4]]

#creo un vettore con la lista degli utenti
users = Int64[]

for i=1:size(ratingsTable)[1]
  #verifico se l'userId corrente non sia già stato inserito
  if !in(ratingsTable[i,1], users)
    push!(users, ratingsTable[i,1])
  end
end

#preparo la URM
URM = zeros(size(users)[1], size(ids)[1])
for i=1:size(ratingsTable)[1]
  #cerco la riga corrispondente all'utente corrente
  row = findElem(ratingsTable[i,1], users, length(users))
  #cerco la colonna corrispondente al programma corrente
  col = findElem(ratingsTable[i,2], ids, length(ids))
  #inserisco la durata nella posizione (row,col)
  URM[row,col] = ratingsTable[i,3]
end

#calcolo la matrice S tramite adjusted cosine similarity
S = ones(length(ids), length(ids))
for i=1:size(URM)[2]
  for j=i+1:size(URM)[2]
    #sfrutto la simmetria della matrice S per il calcolo della similarità
    res = cosineSimilarity(URM[:,i], URM[:,j])
    S[i,j] = res
    S[j,i] = res
  end
end

#calcolo la matrice C
#considero solo le colonne: genreIdx, subGenreIdx, programIdx
genreTable = dataset
genreTable = genreTable[:,[4,5,7]]

#Inizializzo matrice A = (programmi,genere+sottogenere)
idsSize = size(ids)[1]
matrixA=ones( idsSize , 2 )

genreTableSize = size(genreTable)[1]
i = 1
while i <= idsSize
  j=1
  while j < genreTableSize && genreTable[j,3] != ids[i]
    j += 1
  end
  matrixA[i,1]=genreTable[j,1]
  matrixA[i,2]=genreTable[j,2]
  i += 1
end

A_rows = size(matrixA)[1]
C = zeros(A_rows,A_rows)
for i = 1:(A_rows-1)
  for l = i+1:(A_rows)
    k=0.0
     if matrixA[i,1]==matrixA[l,1]!=1
      k=0.5
      if matrixA[i,2]==matrixA[l,2]!=1
       k=1.0
      end
     end
    C[i,l]=k
    C[l,i]=k
  end
end

#calcolo la matrice M tramite l'SGD

#esporto il nuovo dataset di training?
#writecsv("$path\\dataset\\training.csv", dataset)

#=
Controlla se l'id esiste già nel vettore "array", nell'intervallo da 1 a size
Ritorna il numero di riga in cui è stato trovato l'id, -1 altrimenti
=#
function findElem (id, array, size)
  for i = 1:size
    if array[i] == id
      return i
    end
  end
  return -1
end

function findExistingRating (user, program, size)
  for i=1:size
    if ratingsTable[i,2] == user && ratingsTable[i,3] == program
      return i
    end
  end
  return -1
end

#calcola la cosine similarity tra due vettori
function cosineSimilarity (a, b)
  dim = length(a)
  #calcolo il numeratore
  num = 0
  #calcolo il denominatore
  den = 0
  den1 = 0
  den2 = 0
  #eseguo tutti i conti in un unico ciclo
  for n=1:dim
    #controllo che l'utente n abbia valutato entrambi gli item
    if a[n] != 0 && b[n] != 0
      #calcolo la media dei rating dati dall'utente corrente
      avg = userAverage(n)
      num += (a[n] - avg)*(b[n] - avg)
      den1 += (a[n] - avg)^2
      den2 += (b[n] - avg)^2
    end
  end
  den = sqrt(den1)*sqrt(den2)
  #restituisco la similarità totale
  return num / den
end

#calcola la media dei ratings dati dall'utente n
function userAverage (n)
  tot = 0
  count = 0
  for j=1:size(URM)[2]
    if URM[n,j] != 0
      tot += URM[n,j]
      count += 1
    end
  end
  return tot / count
end
