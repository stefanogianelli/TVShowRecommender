#ricavo il percorso base da cui caricare i dataset
path = string(pwd(),"\\TVShowRecommender")

#carico la matrice con le informazioni sui programmi
eventInfo = readdlm("$path\\dataset\\eventLookup.txt", '\t', header=true, use_mmap=true)

#considero solo le colonne programId e start
eventInfo = eventInfo[1][:,[2,4]]

#cerco e salvo i programId di 8 giorni consecutivi
#lo script legge tutti i programmi presenti nel file, è necessario dunque filtrare gli eventi di interesse manualmente
ids = Int64[]

for i=1:size(eventInfo)[1]
  #verifico se il programId corrente non sia già stato inserito
  index = exixstProgramId(eventInfo[i,1], eventInfo[:,1], i)
  if index != -1
    push!(ids, eventInfo[i,1])
  end
end

#carico il dataset
dataset = readdlm("$path\\dataset\\data.txt", ',', use_mmap=true)

#rimuovo la 14esima e 19esima settimana
datasetSize = size(dataset)[1]
i = 1
while i <= datasetSize
  if dataset[i,3] == 14 || dataset[i,3] == 19
    dataset = dataset[[1:(i-1), (i+1):end], :]
    datasetSize -= 1
  else
    i += 1
  end
end

#esporto il nuovo dataset
#writecsv("C:\\Users\\Stefano\\Desktop\\dataset_new.csv", dataset)

#=
Costruisco la User-Rating Matrix a partire dal dataset
Vengono inoltre gestiti i duplicati
Ritorna la URM creata
=#
function buildURM (dataset)
  #considero solo le colonne: userIdx, programIdx, duration
  dataset = dataset[:,[6,7,9]]
  #rimuovo i programIdx duplicati e sommo le loro durate
  datasetSize = size(dataset)[1]
  i = 1
  while i <= datasetSize
    #verifico se il programId non sia già presente
    index = exixstProgramId(dataset[i,2], dataset[:,2], i)
    if index != -1
      dataset[index,3] += dataset[i,3]
      dataset = dataset[[1:(i-1), (i+1):end], :]
      datasetSize -= 1
    else
      i += 1
    end
  end
  return dataset
end

#=
Controlla se l'id esiste già nel dataset nell'intervallo da 1 a size
Ritorna il numero di riga in cui è stato trovato l'id, -1 altrimenti
=#
function exixstProgramId (id, dataset, size)
  for i = 1:(size-1)
    if dataset[i] == id
      return i
    end
  end
  return -1
end
