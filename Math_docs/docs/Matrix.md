#Matrix

This Class is the main class of the Matrix script, it represents a matrix object, and the 
main purpose of the class is for structuring.

##--Arguments--

###matrix | `list` | `2D array`
This argument is essentially the matrix that are to be calculated, it transform the `2D list/array` into a 
matrix object which can be used in following functions for calculations. It also checks for the 
dimention of the `2D array` for discrepencies.

##--Attributes--

###self.matrix
This attribute contains the `list` object of the `Matrix` object

###self.r
This attribute contains the number of rows the `Matrix` object has

###self.c
This attribute contains the number of columns the `Matrix` object has

##--Functions--

###form | size: `int`
This function returns a string value of a formatted matrix for display purposes, this function is used 
when the formatting size(or rather spacing) is not ideal when soley using the `print()`, which is defined 
in the `__str__()` function

###element | i: `int`, j:`int`
This function returns the element according to the coordinates (`i`,`j` | `row`, `column`) given.

##--Sub Classes--

###ZeroMatrix| r:`int`, c:`int`
This class creates a zero `Matrix` object by the number of rows and columns

###IdentityMatrix| r:`int`
This class creates an identitt `Matrix` object by the number of rows

###DiagonalMatrix| vals: `vector`
This class creates a diagonal `Matrix` object by a passed `vector` `vals` which indicates the value of the 
diagonal line of the `Matrix`.



