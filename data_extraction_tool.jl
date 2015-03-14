cd(dirname(@__FILE__))
dir = ".\\dataset"
#input
path = "$dir\\list.txt"
#output
trainingPath = "$dir\\training.txt"
testingPath = "$dir\\testing.txt"
#parametri
test_number = 20
percentage = 0.2

dataset = readdlm(path, '\t', use_mmap=true)
j = size(dataset)[1]
i = 1
while i <= j
  if in(dataset[i,2], dataset[1:(i-1), 2])
    dataset = dataset[[1:(i-1), (i+1):end],:]
    j -= 1
  else
    i += 1
  end
end

total = test_number / percentage
dim = size(dataset)[1]

if dim >= total
  testing = dataset[1:test_number,:]
  training = dataset[(test_number+1):total,:]

  writedlm(trainingPath, training, "\t")
  writedlm(testingPath, testing, "\t")
  println("File creati!")
  println("Programmi di training: $(total - test_number)")
  println("Programmi di testing: $test_number")
else
  println("Limite non raggiunto! (dim = $dim)")
end
