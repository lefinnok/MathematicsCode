#T
This function returns a Transposed `Matrix` object of the input `Matrix` object.

##--Arguments--

###matrix | `Matrix`
The original matrix wished to be transposed.

##--Logic--

By definition, a transpose of a matrix is for element e(i,j) with coordinates i,j, the transpose will
have an element e(j,i) with coordinates i,j.

Or in other words, rows will be column, and columns will be rows.

Therefore, a simple list comprehension can be performed:

	[[row[cidx] for row in matrix.matrix] for cidx in range(matrix.c)]
	
where expanded:

	resultMatrix = []
	for cidx in range(matrix.c): #for every index of columns in the input matrix
		resultRow = []
		for row in matrix.matrix: #for every row in the matrix
			resultRow.append(row[cidx]) #the new row appends the element on the current column of each row(essentially the row becoming the column)
		resultMatrix.append(resultRow)

	

##--Usage--
	matrixA = Matrix([[1,2],[3,4]])
	TransposeA = T(matrixA)