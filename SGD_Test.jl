#=
Testo lalgoritmo tramite la funzione descritta nelle seguenti pagine:
https://github.com/JuliaOpt/Optim.jl
http://en.wikipedia.org/wiki/Rosenbrock_function
Utilizzo a = 1 e b = 100
Ha un minimo in circa 1
=#

#tolleranza
tol = 1e-6

#numero massimo iterazioni
miter = 100

#dimensione passo
alpha = alphaPrec = 0
deltaAlpha = 0.01

#inizializzo il vettore x
x = xnew = [0,0]
fval = rosenbrock(x)

#calcolo il vettore x ottimale
i = 1
while i <= miter && rosenbrock(x) > tol
  @printf "F = %f\nalpha = %f\nx = [%f , %f]\n" rosenbrock(x) alpha x[1] x[2]
  xnew = x - alpha * grad(x)
  if (rosenbrock(xnew) <= fval)
    alphaPrec = alpha
    alpha = alpha + deltaAlpha
    x = xnew
  else
    alpha = alphaPrec - 10 * deltaAlpha
  end
  fval = rosenbrock(x)
  i += 1
end

x

function rosenbrock(x::Vector)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function grad(x::Vector)
    return [-2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1], 200.0 * (x[2] - x[1]^2)]
end
