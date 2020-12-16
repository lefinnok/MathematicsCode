#det
This function calculates and returns the determinant of the input `Matrix` object

##--Arguments--

###matrix | `Matrix`
The original matrix.

##--Logic--

This is a recursive function.

escape condition: single element matrix (for the determinant of a single element matrix is that single element)

	len(matrix.matrix[0]) == 1:

otherwise:

	formula = [element * det(submatrix(matrix,1,idx+1)) * (-1)**idx for idx, element in enumerate(matrix.matrix[0])] #formula of the determinant is: for element in the first row

This formula is modeled after the formula for calculating determinants,

for the determinant of a matrix is the summation of: each of the `elements` of the first row,
multiply by the `determinant` of its `submatrix`, multiply by -1 to the power of the `index`,
where the first element will be positive, second will be negative, and the third will be positive,etc.

And to calculate the `determinant` of its `submatrix`, we have to perform a recursion, which will stop until the
escape condition is met.
