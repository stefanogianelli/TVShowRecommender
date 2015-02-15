#Carica i programId univoci dal file specificato. Usa come delimitatore la tabulazione
function loadProgramIds (filename::String)
  programInfo = readdlm(filename, '\t', use_mmap=true)
  return int(unique(programInfo[:,2]))
end

#Calcola la matrice di similarità tra item S
function buildS ()
  lengthS = length(programs)
  S = spzeros(lengthS, lengthS)
  for i=1:length(ids)
    for j=i:length(ids)
      p1 = programs[ids[i]]
      p2 = programs[ids[j]]
      if (i == j)
        S[p1, p2] = 1
      else
        #sfrutto la simmetria della matrice S per il calcolo della similarità
        S[p1, p2] = S[p2, p1] = cosineSimilarity(URM[:,p1], URM[:,p2])
      end
    end
  end
  return S
end

#calcola la cosine similarity tra due vettori
function cosineSimilarity (a::SparseMatrixCSC, b::SparseMatrixCSC)
  indexes = intersect(rowvals(a), rowvals(b))
  #calcolo il numeratore
  num = 0
  #calcolo il denominatore
  den = 0
  den1 = 0
  den2 = 0
  #eseguo tutti i conti in un unico ciclo
  for n in indexes
    #calcolo la media dei rating dati dall'utente corrente
    avg = userAverage(n)
    num += (a[n] - avg)*(b[n] - avg)
    den1 += (a[n] - avg)^2
    den2 += (b[n] - avg)^2
  end
  den = sqrt(den1)*sqrt(den2)
  #restituisco la similarità totale
  if den == 0
    return 0
  else
    return num / den
  end
end

#calcola la media dei ratings dati dall'utente n
function userAverage (n::Int)
  ratings = nonzeros(URM[n,:])
  size = length(ratings)
  if (size != 0)
    return sum(ratings) / size
  else
    return 0
  end
end

#=function calcC ()
  C = spzeros(length(programs))
  for i = 1:length(ids)
    id1 = programs[ids[i]]
    for j = i:length(ids)
      id2 = programs[ids[j]]
      if (genres[ids[i]][1] == genres[ids[j]][1])
        if (genres[ids[i]][2] == genres[ids[j]][2])
          C[id1,id2] = C[id2,id1] = 1
        else
          C[id1,id2] = C[id2,id1] = 0.5
        end
      end
    end
  end
  return C
end=#

#Restituisce la matrice di similarità rispetto ai contenuti di un certo set di programId
function computeItemItemSim (genreTable::Matrix, ids::Vector)
  #considero solo le colonne: genreIdx, subGenreIdx, programIdx
  genreTable = genreTable[:,[4,5,7]]

  #Inizializzo matrice A = (programmi,genere+sottogenere)
  idsSize = length(ids)
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
  C = spzeros(A_rows,A_rows)
  for i = 1:(A_rows-1)
    for l = i+1:(A_rows)
      k=0.0
       if matrixA[i,1]==matrixA[l,1]!=1
        k=0.5
        if matrixA[i,2]==matrixA[l,2]!=1
         k=1.0
        end
       end
      C[i,l ]= C[l,i] = k
    end
  end
  return C
end

#Calcola la matrice M ottimale attraverso SGD
function gradientDescent ()
  a = alpha
  #inizializzo la matrice M
  MSize= length(programs)
  M = Mnew = spzeros(MSize, MSize)
  fval = object(M)
  @printf "Start value: %f\nStart alpha: %f\n" fval a
  #calcolo la matrice M ottimale
  i = 1
  while i <= miter && object(M) > tol
    Mnew = M - (a / size(M)[1]) * grad(M)
    if (object(Mnew) <= fval)
      M = Mnew
      fval = object(M)
      a += (a * deltaAlpha / 100)
    else
      a /= 2
    end
    i += 1
  end
  @printf "End value: %f\nEnd alpha: %f\n" object(M) a
  return M
end

#definisco la funzione obiettivo
function object (X::SparseMatrixCSC)
  (vecnorm(S - C * X * transpose(C)))^2
end

#restituisce il gradiente della funzione obiettivo
function grad(X::SparseMatrixCSC)
  2 * T * X * T - 2 * Q
end

#Restituisce gli spettacoli consigliati all utente "user"
function getRecommendation (userIndex::Int)
  ratings = Dict()
  for prog in idTesting
    progIndex = programs[prog]
    p = getTau(userIndex, progIndex)
    num = 0
    den = 0
    for k in p
      s = computeSimilarity(k,progIndex)
      num += URM[userIndex, k] * s
      den += s
    end
    if (den != 0)
      res = num / den
    else
      res = 0
    end
    ratings[prog] = res
  end
  return ratings
end

#Restituisce l insieme tau dei programmi trasmessi simili a quello futuro preso in considerazine per un utente
function getTau (u::Int, f::Int)
  set = transpose(C[f,:])
  userRated = transpose(URM[u,:])
  intersect(rowvals(set), rowvals(userRated))
end

#Calcola la similarità tra uno spettacolo passato ed uno futuro
function computeSimilarity (p::Int, f::Int)
    (C[p,:] * M * transpose(C[f,:]))[1]
end
