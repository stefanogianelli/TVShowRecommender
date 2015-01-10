dataset = readdlm("C:\\Users\\Stefano\\Desktop\\auditel_txt\\data.txt", ',', use_mmap=true)

buildURM(dataset)

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

function exixstProgramId (id, dataset, size)
  for i = 1:(size-1)
    if dataset[i,2] == id
      return i
    end
  end
  return -1
end
