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

#considero solo le colonne: userIdx, programIdx, duration
ratingsTable = ratingsTable[:,[6,7,9]]

#rimuovo i programIdx duplicati e sommo le loro durate
#considero solo i programmi selezionati nel vettore ids
#vengono inoltre rimosse le settimane 14 e 19
ratingsTableSize = size(training)[1]
i = 1
while i <= ratingsTableSize
  #non considero il programma se siamo nella 14esima o 19esima settimana
  if ratingsTable[i,3] != 14 && ratingsTable[i,3] != 19
    #verifico che il programId corrente sia presente nel vettore ids
    if in(ratingsTable[i,2], ids)
      #verifico se il programId non sia già presente
      index = findElem(ratingsTable[i,2], ratingsTable[:,2], i-1)
      if index != -1
        #se il programId esiste viene sommato il valore della duration
        ratingsTable[index,3] += ratingsTable[i,3]
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
    S[i,j] = S[j,i] = cosineSimilarity(URM[:,i], URM[:,j])
  end
end

#calcolo la matrice C
#importare qui il codice del file buildMatrixC

#esporto il nuovo dataset di training?
#writecsv("$path\\dataset\\training.csv", dataset)

#=
Controlla se l'id esiste già nel vettore "array", nell'intervallo da 1 a size
Ritorna il numero di riga in cui è stato trovato l'id, -1 altrimenti
=#
function findElem (id, array, size)
  if size > length(array)
    size = length(array)
  end
  for i = 1:size
    if array[i] == id
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
