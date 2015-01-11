#load the dataset
dataset = readdlm("C:\\Users\\Stefano\\Desktop\\auditel.txt", ',', use_mmap=true)

#remove the 14th and the 19th weeks
datasetSize = size(dataset)[1]
i = 1
while i <= datasetSize
  if dataset[i,3] == 14 || dataset[i,3] == 19
    dataset = dataset[[1:(i-1), (i+1):end], :]
    datasetSize -= 1
  else
    i += 1
  end
end

#export the new dataset
writecsv("C:\\Users\\Stefano\\Desktop\\auditel_new.txt", dataset)

#=
Build the User-Rating Matrix from the original dataset
It take care of duplicate program ids
Return the URM
=#
function buildURM (dataset)
  #retain only the columns: userIdx, programIdx, duration
  dataset = dataset[:,[6,7,9]]
  #removing duplicates programIdx and sum their durations
  datasetSize = size(dataset)[1]
  i = 1
  while i <= datasetSize
    index = exixstProgramId(dataset[i,2], dataset, i)
    if index != -1
      dataset[index,3] += dataset[i,3]
      dataset = dataset[[1:(i-1), (i+1):end], :]
      datasetSize -= 1
    else
      i += 1
    end
  end
  return dataset
end

#=
Check if the id already exists in the dataset in the rows from 1 to size
and return the first index where the id is found, -1 otherwise
=#
function exixstProgramId (id, dataset, size)
  for i = 1:(size-1)
    if dataset[i,2] == id
      return i
    end
  end
  return -1
end
