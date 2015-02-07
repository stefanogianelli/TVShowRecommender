#=
PARAMETRI
=#

#permette di scegliere se usare la cartella di test o quella con i dati completi
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
alpha = 0.00017
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
dataset = readdlm(datasetPath, ',', use_mmap=true)
toc()

#considero solo le colonne: weekIdx, userIdx, programIdx, duration
ratingsTable = dataset[:,[3,6,7,9]]

#rimuovo i programIdx duplicati e sommo le loro durate
#considero solo i programmi selezionati nel vettore ids
#vengono inoltre rimosse le settimane 14 e 19
println("Rimuovo i duplicati ...")
tic()
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
toc()

#rimuovo la colonna della settimana, non più necessaria
ratingsTable = ratingsTable[:,[2,3,4]]

#creo un vettore con la lista degli utenti
users = Int64[]

println("Salvo la lista degli utenti univoci ...")
tic()
for i=1:size(ratingsTable)[1]
  #verifico se l'userId corrente non sia già stato inserito
  if !in(ratingsTable[i,1], users)
    push!(users, ratingsTable[i,1])
  end
end
toc()

#preparo la URM
println("Costruisco la URM ...")
tic()
URM = buildURM()
toc()

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
R = zeros(length(users), length(idTesting))

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
  #considero solo le colonne programId e start
  programInfo = programInfo[:,[2,4]]
  #cerco e salvo i programId
  ids = Int64[]
  for i=1:size(programInfo)[1]
    #verifico se il programId corrente non sia già stato inserito
    if !in(programInfo[i,1], ids)
      push!(ids, programInfo[i,1])
    end
  end
  return ids
end

#Costruisce la User-Ratings Matrix
function buildURM ()
  URM = zeros(size(users)[1], size(ids)[1])
  for i=1:size(ratingsTable)[1]
    #cerco la riga corrispondente all'utente corrente
    row = findElem(ratingsTable[i,1], users, length(users))
    #cerco la colonna corrispondente al programma corrente
    col = findElem(ratingsTable[i,2], ids, length(ids))
    #inserisco la durata nella posizione (row,col)
    URM[row,col] = ratingsTable[i,3]
  end
  return URM
end

#Calcola la matrice di similarità tra item S
function buildS ()
  S = zeros(length(ids) + length(idTesting), length(ids) + length(idTesting))
  for i=1:size(URM)[2]
    for j=i:size(URM)[2]
      if (i == j)
        S[i,j] = 1
      else
        #sfrutto la simmetria della matrice S per il calcolo della similarità
        S[i,j] = S[j,i] = cosineSimilarity(URM[:,i], URM[:,j])
      end
    end
  end
  return S
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
  if den == 0
    return 0
  else
    return num / den
  end
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
  if count == 0
    return 0
  else
    return tot / count
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

function gradientDescent ()
  #inizializzo la matrice M
  M = Mnew = zeros(length(ids) + length(idTesting),length(ids) + length(idTesting))
  fval = object(M)

  #calcolo la matrice M ottimale
  i = 1
  while i <= miter && object(M) > tol
    index = rand(1:size(M)[2])
    Mnew[index,:] = M[index,:] - alpha * grad(M)[index,:]
    if (object(Mnew) < fval)
      #@printf "-----------------------\niter = %d\nF = %f\n" i object(Mnew)
      M = Mnew
      fval = object(M)
    end
    i += 1
  end
  return M
end

#definisco la funzione obiettivo
function object (X::Matrix)
  return vecnorm(S - C * X * transpose(C));
end

#restituisce il gradiente della funzione obiettivo
function grad(X::Matrix)
  return 2 * T * X * T - 2 * Q
end

#Costruisce la matrice R
function buildR ()
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
function getTau (u, f, N)
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
function computeSimilarity (p, f)
  fIndex = length(ids) + f
  return (C[p,:] * M * transpose(C[fIndex,:]))[1]
end

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

#=
Controlla se l'utente "user" ha già dato un rating al programma "progra", nell'intervallo da 1 a size
Ritorna il numero di riga in cui è stato trovato il rating, -1 altrimenti
=#
function findExistingRating (user, program, size)
  for i=1:size
    if ratingsTable[i,2] == user && ratingsTable[i,3] == program
      return i
    end
  end
  return -1
end
