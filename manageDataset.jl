#ricavo il percorso base da cui caricare i dataset
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

#carico il dataset di training
dataset = readdlm("$path\\dataset\\data.txt", ',', use_mmap=true)

#creo una copia per la matrice di training
training = dataset

#considero solo le colonne: userIdx, programIdx, duration
training = training[:,[6,7,9]]

#rimuovo i programIdx duplicati e sommo le loro durate
#considero solo i programmi selezionati nel vettore ids
trainingSize = size(training)[1]
i = 1
while i <= trainingSize
  #non considero il programma se siamo nella 14esima o 19esima settimana
  if training[i,3] != 14 && training[i,3] != 19
    #verifico che il programId corrente sia presente nel vettore ids
    if in(training[i,2], ids)
      #verifico se il programId non sia già presente
      index = exixstProgramId(training[i,2], training[:,2], i)
      if index != -1
        training[index,3] += training[i,3]
        training = training[[1:(i-1), (i+1):end], :]
        trainingSize -= 1
      else
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

#calcolo la matrice S tramite cosine similarity

#calcolo la matrice C
#importare qui il codice del file buildMatrixC

#esporto il nuovo dataset di training
#writecsv("$path\\dataset\\training.csv", dataset)

#=
Controlla se l'id esiste già nel vettore "array", nell'intervallo da 1 a size
Ritorna il numero di riga in cui è stato trovato l'id, -1 altrimenti
=#
function exixstProgramId (id, array, size)
  for i = 1:(size-1)
    if array[i] == id
      return i
    end
  end
  return -1
end

#calcola la cosine similarity tra due vettori?
function cosineSimilarity ()
end
