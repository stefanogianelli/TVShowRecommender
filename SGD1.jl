#---- parte di supporto, da cancellare
ids = [1,2,3]

S = ones(length(ids),length(ids))
C = ones(length(ids),length(ids))

#---- fine parte di supporto

#eseguo i calcoli per il gradiente che non devono essere rifatti ogni volta
Q = transpose(C)*S*C
T = transpose(C)*C

#numero massimo iterazioni
miter = 100

#dimensione passo
#ottimizzazione: non fissarlo, ma dedurlo ad ogni passaggio
alpha = 0.2

#inizializzo la matrice M a 1
M = ones(length(ids),length(ids))

#calcolo la matrice M ottimale
for i=1:miter
  M = M - alpha*grad(M)
end

M

#definisco la funzione obiettivo
function object (X::Matrix)
  #TODO: scrivere funzione obiettivo
  return 0;
end

#restituisce il gradiente della funzione obiettivo
function grad(X::Matrix)
  return 2*T*X*T-2*Q
end
