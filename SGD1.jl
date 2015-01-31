#---- parte di supporto, da cancellare
ids = [1,2,3]

S = rand(-1:1, length(ids),length(ids))
C = rand(0:1, length(ids),length(ids))

#---- fine parte di supporto

#eseguo i calcoli per il gradiente che non devono essere rifatti ogni volta
Q = transpose(C)*S*C
T = transpose(C)*C

#tolleranza
tol = 1e-6

#numero massimo iterazioni
miter = 1000

#dimensione passo
alpha = 0.002

#inizializzo la matrice M
M = zeros(length(ids),length(ids))

#calcolo la matrice M ottimale
i = 1
while i <= miter && object(M) > tol
  fval = object(M)
  @printf "-----------------------\niter = %d\nF = %f\n" i fval
  Mnew = M - alpha * grad(M)
  if (object(Mnew) < fval)
    M = Mnew
  else
    return
  end
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
