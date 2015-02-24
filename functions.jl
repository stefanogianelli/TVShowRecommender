#Carica i programId univoci dal file specificato. Usa come delimitatore la tabulazione
function load_program_ids (filename::String)
  programInfo = readdlm(filename, '\t', use_mmap=true)
  return int(unique(programInfo[:,2]))
end

function clean_dataset! (dataset::Matrix, ids::Array, idTesting::Array, ratings::Dict, users::Dict, programs::Dict, genres::Dict)
  #inizializzo contattori
  countUser = 1
  countProg = 1
  #scansiono tutto il dataset
  for i = 1:size(dataset)[1]
    #elimino le settimne 14 e 19
    if (dataset[i,3] != 14 && dataset[i,3] != 19)
      #controllo se l'id corrente è nell'insieme degli id di training
      if (in(dataset[i,7], ids) || in(dataset[i,7], idTesting))
        try
          ratings[dataset[i,6], dataset[i,7]] += dataset[i,9]
        catch
          ratings[dataset[i,6], dataset[i,7]] = dataset[i,9]
        end
        #aggiungo l'utente corrente
        if (!in(dataset[i,6], keys(users)))
          users[dataset[i,6]] = countUser
          countUser += 1
        end
        #aggiungo il programma corrente
        if (!in(dataset[i,7], keys(programs)))
          programs[dataset[i,7]] = countProg
          countProg += 1
        end
        #creo un dizionario con i generi e sottogeneri dei programmi
        if (!in(dataset[i,7], keys(genres)))
          genres[dataset[i,7]] = (dataset[i,4], dataset[i,5])
        end
      end
    end
  end
end

#funzione di test
function clean_dataset1! (datasetPath::String, ids::Array, idTesting::Array, ratings::Dict, users::Dict, programs::Dict, genres::Dict)
  #inizializzo contattori
  countUser = 1
  countProg = 1
  #scansiono tutto il dataset
  file = open(datasetPath)
  for line in eachline(file)
    #leggo la riga corrente
    dataset = int(split(line, ","))
    #elimino le settimne 14 e 19
    if (dataset[3] != 14 && dataset[3] != 19)
      #controllo se l'id corrente è nell'insieme degli id di training
      if (in(dataset[7], ids) || in(dataset[7], idTesting))
        try
          ratings[dataset[6], dataset[7]] += dataset[9]
        catch
          ratings[dataset[6], dataset[7]] = dataset[9]
        end
        #aggiungo l'utente corrente
        if (!in(dataset[6], keys(users)))
          users[dataset[6]] = countUser
          countUser += 1
        end
        #aggiungo il programma corrente
        if (!in(dataset[7], keys(programs)))
          programs[dataset[7]] = countProg
          countProg += 1
        end
        #creo un dizionario con i generi e sottogeneri dei programmi
        if (!in(dataset[7], keys(genres)))
          genres[dataset[7]] = (dataset[4], dataset[5])
        end
      end
    end
  end
  flush(file)
  close(file)
end

#Calcola la matrice di similarità tra item S
function build_similarity_matrix (programs::Dict, ids::Array, URM::SparseMatrixCSC)
  lengthS = length(programs)
  lengthId = length(ids)
  S = spzeros(lengthS, lengthS)
  for i = 1:lengthId
    for j = i:lengthId
      p1 = programs[ids[i]]
      p2 = programs[ids[j]]
      if (i == j)
        S[p1, p2] = 1
      else
        #sfrutto la simmetria della matrice S per il calcolo della similarità
        S[p1, p2] = S[p2, p1] = cosine_similarity(p1, p2, URM)
      end
    end
  end
  return S
end

#calcola la cosine similarity tra due vettori
function cosine_similarity (i1::Int, i2::Int, URM::SparseMatrixCSC)
  a = URM[:,i1]
  b = URM[:,i2]
  #cerco gli utenti che hanno dato un voto ad entrambi i programmi
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
    avg = user_average(n, URM)
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
function user_average (userIndex::Int, URM::SparseMatrixCSC)
  ratings = 0
  count = 0
  for i = 1:length(ids)
    if URM[userIndex, programs[ids[i]]] != 0
      ratings += URM[userIndex, programs[ids[i]]]
      count += 1
    end
  end
  if count != 0
    return ratings / count
  else
    return 0
  end
end

#Restituisce la matrice di similarità rispetto ai contenuti di un certo set di programId
function compute_item_item_similarity (ids::Array, programs::Dict, genres::Dict)
  id_number = length(ids)
  program_number = length(programs)
  C = spzeros(program_number, program_number)
  for i = 1:id_number
    id1 = programs[ids[i]]
    for j = i:id_number
      id2 = programs[ids[j]]
      if genres[ids[i]][1] == genres[ids[j]][1]
        if genres[ids[i]][2] == genres[ids[j]][2]
          C[id1,id2] = C[id2,id1] = 1
        else
          C[id1,id2] = C[id2,id1] = 0.5
        end
      end
    end
  end
  return C
end

#Calcola la matrice M ottimale attraverso SGD
function gradient_descent (a::Number, MSize::Int, S::SparseMatrixCSC, C::SparseMatrixCSC, Q::SparseMatrixCSC, T::SparseMatrixCSC)
  #inizializzo la matrice M
  M = Mnew = spzeros(MSize, MSize)
  fval = object(M, S, C)
  gain = 1
  println("Start value = $fval")
  #calcolo la matrice M ottimale
  i = 1
  while fval > tol && gain > tol
    Mnew = M - (a / MSize) * grad(M, T, Q)
    f = object(Mnew, S, C)
    if f <= fval
      gain = abs(fval - f)
      M = Mnew
      fval = f
      a += (a * deltaAlpha / 100)
    else
      a /= 2
    end
    i += 1
  end
  println("End value = $fval")
  return M
end

#definisco la funzione obiettivo
function object (X::SparseMatrixCSC, S::SparseMatrixCSC, C::SparseMatrixCSC)
  (vecnorm(S - C * X * transpose(C)))^2
end

#restituisce il gradiente della funzione obiettivo
function grad (X::SparseMatrixCSC, T, Q)
  2 * T * X * T - 2 * Q
end

#Restituisce gli spettacoli consigliati all'utente "user"
function get_recommendation (userIndex::Int, idTesting::Array, programs::Dict, URM::SparseMatrixCSC, C::SparseMatrixCSC, M::SparseMatrixCSC)
  test_items = length(idTesting)
  ratings = spzeros(1, test_items)
  for i = 1:test_items
    progIndex = programs[idTesting[i]]
    p = get_tau(userIndex, progIndex, C, URM)
    num = 0
    den = 0
    for k in p
      s = compute_similarity(k,progIndex, C, M)
      num += URM[userIndex, k] * s
      den += s
    end
    if den != 0
      res = num / den
    else
      res = 0
    end
    ratings[1, i] = res
  end
  return ratings
end

#Restituisce l'insieme tau dei programmi trasmessi simili a quello futuro preso in considerazine per un utente
function get_tau (userIndex::Int, futureIndex::Int, C::SparseMatrixCSC, URM::SparseMatrixCSC)
  set = transpose(C[futureIndex,:])
  userRated = transpose(URM[userIndex,:])
  intersect(rowvals(set), rowvals(userRated))
end

#Calcola la similarità tra uno spettacolo passato ed uno futuro
function compute_similarity (pastIndex::Int, futureIndex::Int, C::SparseMatrixCSC, M::SparseMatrixCSC)
    (C[pastIndex,:] * M * transpose(C[futureIndex,:]))[1]
end
