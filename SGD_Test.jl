#=
Testo lalgoritmo tramite la funzione descritta nelle seguenti pagine:
https://github.com/JuliaOpt/Optim.jl
http://en.wikipedia.org/wiki/Rosenbrock_function
Utilizzo a = 1 e b = 100
Ha un minimo in circa 1
=#

#tolleranza
tol = 1e-5

#numero massimo iterazioni
miter = 10000

#dimensione passo
alpha = 0.002

#inizializzo il vettore x
x = [0,0]

#calcolo il vettore x ottimale
i = 1
while i <= miter && rosenbrock(x) > tol
  @printf "----------\niter = %d\nF = %f\nx = [%f , %f]\n" i rosenbrock(x) x[1] x[2]
  x = x - alpha * grad(x)
  i += 1
end

x

function rosenbrock(x::Vector)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function grad(x::Vector)
    return [-2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1], 200.0 * (x[2] - x[1]^2)]
end