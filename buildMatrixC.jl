#function buildMatrixC (matrixA)
#  A_rows, A_cols = size(matrixA)
#  matrixC = ones(A_rows,A_rows)
#  for i = 1:(A_rows-1)
#    for l = i+1:(A_rows)
#      k=0
#      for j=1:A_cols
#        if matrixA[i,j]==matrixA[l,j]==1
#          k=k+0.5
#        end
#      end
#      matrixC[i,l]=k
#      matrixC[l,i]=k
#    end
#  end
#  return matrixC
#end


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

ids

#carico il dataset di training
training = readdlm("$path\\dataset\\data.txt", ',', use_mmap=true)

#considero solo le colonne: genreIdx, subGenreIdx, programIdx
training = training[:,[4,5,7]]

#Inizializzo matrice A = (programmi,genere+sottogenere)
idsSize = size(ids)[1]
matrixA=ones( idsSize , 2 )

trainingSize = size(training)[1]
i = 1
while i <= idsSize
  j=1
  while j <= trainingSize
    if training[j,3] == ids[i]
      matrixA[i,1]=training[j,1]
      matrixA[i,2]=training[j,2]
    end
    j += 1
  end
  i += 1
end

matrixA

A_rows = size(matrixA)[1]
matrixC = zeros(A_rows,A_rows)
for i = 1:(A_rows-1)
  for l = i+1:(A_rows)
    k=0.0
     if matrixA[i,1]==matrixA[l,1]!=1
      k=0.5
      if matrixA[i,2]==matrixA[l,2]!=1
       k=1.0
      end
     end
    matrixC[i,l]=k
    matrixC[l,i]=k
  end
end

training

ids
matrixC
