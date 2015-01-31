#---- parte di supporto, da cancellare
ids = [1,2,3]

S = ones(length(ids),length(ids))
C = ones(length(ids),length(ids))

#---- fine parte di supporto

#eseguo i calcoli per il gradiente che non devono essere rifatti ogni volta
Q = transpose(C)*S*C
T = transpose(C)*C

#tolleranza
tol = 1e-6

#numero massimo iterazioni
miter = 10000

#dimensione passo
#ottimizzazione: non fissarlo, ma dedurlo ad ogni passaggio
alpha = 0.002

#inizializzo la matrice M a 1
M = zeros(length(ids),length(ids))

#calcolo la matrice M ottimale
i = 1
while i <= miter && object(M) > tol
  @printf "-----------------------\niter = %d\nF = %f\n" i object(M)
  M = M - alpha * grad(M)
  i += 1
end

M

#definisco la funzione obiettivo
function object (X::Matrix)
  return vecnorm(S - C * X * transpose(C))^2;
end

#restituisce il gradiente della funzione obiettivo
function grad(X::Matrix)
  return 2*T*X*T-2*Q
end
