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

#creo una copia per la matrice di training
training = dataset

#considero solo le colonne: userIdx, programIdx, duration
training = training[:,[6,7,9]]

#rimuovo i programIdx duplicati e sommo le loro durate
#considero solo i programmi selezionati nel vettore ids
#vengono inoltre rimosse le settimane 14 e 19
trainingSize = size(training)[1]
i = 1
while i <= trainingSize
  #non considero il programma se siamo nella 14esima o 19esima settimana
  if training[i,3] != 14 && training[i,3] != 19
    #verifico che il programId corrente sia presente nel vettore ids
    if in(training[i,2], ids)
      #verifico se il programId non sia già presente
      index = findElem(training[i,2], training[:,2], i-1)
      if index != -1
        #se il programId esiste viene sommato il valore della duration
        training[index,3] += training[i,3]
        training = training[[1:(i-1), (i+1):end], :]
        trainingSize -= 1
      else
        #altrimenti passo alla riga successiva
        i += 1
      end
    else
      #rimuovo la riga
      training = training[[1:(i-1), (i+1):end], :]
      trainingSize -= 1
    end
  else
    #rimuovo la riga
    training = training[[1:(i-1), (i+1):end], :]
    trainingSize -= 1
  end
end

#creo un vettore con la lista degli utenti
users = Int64[]

for i=1:size(training)[1]
  #verifico se l'userId corrente non sia già stato inserito
  if !in(training[i,1], users)
    push!(users, training[i,1])
  end
end

#preparo la URM
URM = zeros(size(users)[1], size(ids)[1])
for i=1:size(training)[1]
  #cerco la riga corrispondente all'utente corrente
  row = findElem(training[i,1], users, size(users)[1])
  #cerco la colonna corrispondente al programma corrente
  col = findElem(training[i,2], ids, size(ids)[1])
  #posizione la durata nella posizione (row,col)
  URM[row,col] = training[i,3]
end

#calcolo la matrice S tramite adjusted cosine similarity
S = ones(size(ids)[1], size(ids)[1])
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
  if size > size(array)[1]
    size = size(array)[1]
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
  dim = size(a)[1]
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
