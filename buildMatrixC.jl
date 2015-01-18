function buildMatrixC (matrixA)
  A_rows, A_cols = size(matrixA)
  matrixC = ones(A_rows,A_rows)
  for i = 1:(A_rows-1)
    for l = i+1:(A_rows)
      k=0
      for j=1:A_cols
        if matrixA[i,j]==matrixA[l,j]==1
          k=k+0.5
        end
      end
      matrixC[i,l]=k
      matrixC[l,i]=k
    end
  end
  return matrixC
end
