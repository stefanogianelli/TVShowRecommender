#=
PARAMETRI
=#

#permette di scegliere la cartella
dir = ".\\dataset"
#percorso dati di training
trainingPath = "$dir\\training.txt"
#percorso dati di testing
testingPath = "$dir\\testing.txt"
#percorso dataset
datasetPath = "$dir\\data.txt"
#tolleranza (SGD)
tol = 1e-6
#numero massimo iterazioni (SGD)
miter = 1000
#dimensione passo (SGD)
alpha = 0.0001
#incremento percentuale del learning rate (SGD)
deltaAlpha = 5
#numero item simili (calcolo R)
N = 5

#=
MAIN
=#

#ricavo il percorso base da cui caricare i dataset
cd(dirname(@__FILE__))

#carico la matrice con le informazioni sui programmi di training
println("Carico gli id dei programmi di training")
tic()
ids = loadProgramIds(trainingPath)
toc()

#carico la matrice con le informazioni sui programmi di testing
println("Carico gli id dei programmi di testing")
tic()
idTesting = loadProgramIds(testingPath)
toc()

#carico il dataset
println("Carico il dataset ...")
tic()
dataset = int(readdlm(datasetPath, ',', use_mmap=true))
toc()

#creo un vettore con la lista degli utenti
println("Salvo la lista degli utenti univoci ...")
tic()
users = unique(dataset[:,6])
toc()

URM = sparse(dataset[:,6], dataset[:,7], dataset[:,9])

#calcolo la matrice S tramite adjusted cosine similarity
println("Calcolo la matrice S ...")
tic()
S = buildS()
toc()

#calcolo la matrice C
println("Calcolo la matrice C")
tic()
C = computeItemItemSim(dataset, [ids,idTesting])
toc()

#eseguo i calcoli per il gradiente che non devono essere rifatti ogni volta
Q = transpose(C) * S * C
T = transpose(C) * C

#calcolo la matrice M
println("Calcolo la matrice M ottimale ...")
tic()
M = gradientDescent()
toc()

#costruisco la matrice con i ratings per i programmi futuri
println("Calcolo la matrice R dei ratings per i programmi futuri ...")
tic()
R = buildR()
toc()

println("Fine.")

R

#=
FUNZIONI
=#

#Carica i programId univoci dal file specificato. Usa come delimitatore la tabulazione
function loadProgramIds (filename::String)
  programInfo = readdlm(filename, '\t', use_mmap=true)
  return int(unique(programInfo[:,2]))
end

#Calcola la matrice di similarità tra item S
function buildS ()
  lengthS = max(ids[indmax(ids)], idTesting[indmax(idTesting)])
  S = spzeros(lengthS, lengthS)
  for i=1:length(ids)
    for j=i:length(ids)
      if (i == j)
        S[ids[i], ids[j]] = 1
      else
        #sfrutto la simmetria della matrice S per il calcolo della similarità
        S[ids[i], ids[j]] = S[ids[j], ids[i]] = cosineSimilarity(URM[:,ids[i]], URM[:,ids[j]])
      end
    end
  end
  return S
end

#calcola la cosine similarity tra due vettori
function cosineSimilarity (a, b)
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
      C[i,l ]= C[l,i] = k
    end
  end
  return C
end

#Calcola la matrice M ottimale attraverso SGD
function gradientDescent ()
  a = alpha
  #inizializzo la matrice M
  M = Mnew = zeros(length(ids) + length(idTesting),length(ids) + length(idTesting))
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
function object (X::Matrix)
  return (vecnorm(S - C * X * transpose(C)))^2
end

#restituisce il gradiente della funzione obiettivo
function grad(X::Matrix)
  return 2 * T * X * T - 2 * Q
end

#Costruisce la matrice R
function buildR ()
  R = zeros(length(users), length(idTesting))
  for i=1:size(R)[1]
    for j=1:size(R)[2]
      p = getTau(i,j,N)
      num = den = 0
      for k=1:length(p)
        pIndex = findElem(p[k], ids, length(ids))
        s = computeSimilarity(pIndex,j)
        num += URM[i, pIndex] * s
        den += s
      end
      if (den != 0)
        R[i,j] = num / den
      else
        R[i,j] = 0
      end
    end
  end
  return R
end

#Restituisce linsieme tau dei programmi trasmessi simili a quello futuro preso in considerazine per un utente
function getTau (u::Int, f::Int, N::Int)
  fIndex = length(ids) + f
  set = C[fIndex,:]
  userRated = URM[u,:]
  common = Int64[]
  for i=1:length(userRated)
    if (set[i] != 0 && userRated[i] != 0)
      push!(common, ids[i])
    end
  end
  if (length(common) > N)
    sort!(common, rev=true)
    common = common[1:N]
  end
  return common
end

#Calcola la similarità tra uno spettacolo passato ed uno futuro
function computeSimilarity (p::Int, f::Int)
  fIndex = length(ids) + f
  return (C[p,:] * M * transpose(C[fIndex,:]))[1]
end

#=
Controlla se l'id esiste già nel vettore "array", nell'intervallo da 1 a size
Ritorna il numero di riga in cui è stato trovato l'id, -1 altrimenti
=#
function findElem (id::Int, array::Vector, size::Int)
  for i = 1:size
    if array[i] == id
      return i
    end
  end
  return -1
end

#=
Controlla se l'utente "user" ha già dato un rating al programma "progra", nell'intervallo da 1 a size
Ritorna il numero di riga in cui è stato trovato il rating, -1 altrimenti
=#
function findExistingRating (user::Int, program::Int, size::Int)
  for i=1:size
    if ratingsTable[i,2] == user && ratingsTable[i,3] == program
      return i
    end
  end
  return -1
end
