# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 10:30:58 2020

@author: lefin
"""

from fractions import Fraction
from sympy import symbols, Eq, solve # Temporary Algebraic Solution
        

class Matrix():
    '''
    A matrix is a rectangular array of numbers
    
    with r rows and c columns
    '''
    def __init__(self, matrix):
        #validity check
        valid = True
        for row, nextRow in zip(matrix[:-1],matrix[1:]):
            '''
            so if one row has a different length(column) to the next, it is
            not rectangluar, which will trigger and define valid to be false.
            '''
            if not len(row) == len(nextRow):
                valid = False
            
        
        if valid: 
            #define matrix/array
            self.matrix = matrix
            self.r = len(self.matrix)
            self.c = len(self.matrix[0])
        else:
            self.matrix = '[Invalid Matrix]'
    
    def form(self, size = 1):
        '''
        this returns a formatted string for the matrix
        
        ⎡ 1 ⎤   
        ⎢ 3 ⎥ 
        ⎣ 8 ⎦
        
        '''
        if len(self.matrix) == 1:
            res = str(self.matrix[0]).replace(',',''.join([' ' for _ in range(size-1)]))
        else:
            resls = [['⎢'] + [str(element) for element in row] + ['⎥'] for row in self.matrix]
            resls[0][0] = '⎡'
            resls[0][-1] = '⎤'
            resls[-1][0] = '⎣'
            resls[-1][-1] = '⎦'
            maxlen = len(max([max(row,key=len) for row in resls], key=len)) #find the lengthiest string
            res = ''
            for row in resls:
                for prev,aft in zip(row[:-1], row[1:]):
                    res += prev + ''.join([' ' for _ in range((int(maxlen/2)+1 - int(len(aft)/2) - int(len(prev)/2) + (len(prev)+1)%2)+size)])
                res += row[-1] + '\n'
            
        
        return res
    
    def element(self,i,j):
        '''
        elements in a matrix can be accessed through coordinates i, j
        where i will be the y coordinate(index of row)
        and   j will be the x coordinate(index of column)
        
        i, j = row, column
        '''
        if i > self.r or j > self.c:
            return 'Invalid Coordinates'
        #convertion from base 1 to base 0
        i -= 1
        j -= 1
        return self.matrix[i][j]

class ZeroMatrix(Matrix):
    '''
    A Zero Matrix is a matrix with all its elements as 0
    '''
    def __init__(self, r, c):
        matrix = [[0 for _ in range(c)] for _ in range(r)]
        super().__init__(matrix)

class IdentityMatrix(Matrix):
    '''
    An Identity Matrix is a square matrix with a diagonal line of 1 and others as 0
    which is also a diagonal matrix of all 1
    
    It acts as a multiplicative identity:
        Where I*A = A
        and   A*I = A
    
    '''
    def __init__(self,r):
        matrix = [[1 if cidx == ridx else 0 for cidx in range(r)] for ridx in range(r)]
        super().__init__(matrix)

class DiagonalMatrix(Matrix):
    '''
    A Diagonal Matrix is a square matrix where only the diagonal line of the matrix has values,
    while other parts will be all 0, something like an Identity Matrix, but the values are not all 1s
    
    which is both upper and lower triangular
    
    which also means a diagonal matrix could be yielded from putting a square matrix through both uptri() and lowtri()
    
    '''
    def __init__(self,vals):
        matrix = [[vals[ridx] if cidx == ridx else 0 for cidx in range(len(vals))] for ridx in range(len(vals))]
        super().__init__(matrix)

def T(matrix):
    '''
    Matricies can be transposed, 
    as in rows will be columns, 
    and columns will be rows
    
    which basically boils down to,
    getting all values in the first column and put that in the first row
    getting all values in the second column and put that in the second row
                .
                .
                .
    getting all values in the nth column and put that in the nth row
    '''
    return Matrix([[row[cidx] for row in matrix.matrix] for cidx in range(matrix.c)])

def detM(matrix):#[tag: singular, cofactor]
    '''
    A determinant of a square matrix is a scalar value which could be calculated 
    by using elements within said square matrix.
    
    determinants are useful in:
        - determining the multiplicative inverse of one matrix
        - systems of linear equations
        - calculus
            .
            .
            .
    
    If such determinant is 0, this indicates such matrix is singular(non-invertible)
    
    A determinant of a matrix is yielded by:
        - looping through each element in the first row of the matrix
        - multiply the element by the determinant of a matrix which is composed of 
          all elements that is not on the same row or column (think soduku, cross)
        - until the composed matrix have only one element, where the determinant 
          of such matrix (single element matrix), is that sole element.
    
    '''
    if len(matrix[0]) == 1: #if there is only one element in the first row, return the element itself
        return matrix[0][0]
    else:
        '''
        Here, formula is assigned a list of values, which is where each element
        of the first row of the matrix is multiplied by the determinant of the
        submatrix, and -1 to the power of idx, to make the negativity alternative
        
        it can also be interpreted as each element multiplying its cofactor
        
        the submatrix is determined by:
            not in the first row (which is skipped)
            not in the same column (vidx != idx)
        
        Which, in a recursive function, it will perform recursion until the 
        determinants are all single elements
        '''
        formula = [element * detM([[val for vidx, val in enumerate(row) if vidx != idx] for row in matrix[1:]]) * (-1)**idx for idx, element in enumerate(matrix[0])] #formula of the determinant is: for element in the first row
        return sum(formula)

def submatrix(matrix, r, c):
    '''
    A submatrix of an element (with the coordinate r and c) is a matrix composed of elements
    that is not on the same row or column as said element
    '''
    return Matrix([[val for vidx, val in enumerate(row,1) if vidx != c and ridx != r] for ridx, row in enumerate(matrix.matrix,1) if [val for vidx, val in enumerate(row,1) if vidx != c and ridx != r] != []])


def det(matrix):#this is the class version of the function
    if len(matrix.matrix[0]) == 1: 
        return matrix.matrix[0][0]
    else:
        formula = [element * det(submatrix(matrix,1,idx+1)) * (-1)**idx for idx, element in enumerate(matrix.matrix[0])] #formula of the determinant is: for element in the first row
        return sum(formula)


def cof(matrix):
    '''
    A cofactor matrix is a matrix containing the cofactor of each element of the original matrix
    
    A cofactor of an element is denoted by:
        (-1)**(row+col)*det(submatrix)
        
    '''
    return Matrix([[((-1)**(r+c))*det(submatrix(matrix,r,c)) for c,element in enumerate(row,1)] for r,row in enumerate(matrix.matrix,1)])

def adj(matrix):#[tag: cofactor, transpose]
    '''
    An adjoint matrix is the transpose of the cofactor matrix of a matrix
    '''
    return T(cof(matrix))

def square(matrix, mode = 'truncate'):
    '''
    Square matricies are matricies where r == c, which means that they will be very flexible
    when performing vector multiplications with other matricies
    
    this function will turn a matrix into a square matrix by either truncating 
    the matrix or extending the matrix
    
    therefore, modes include:
        'truncate'
        'extend'
    '''
    #compare the length of row and column
    rowLong = matrix.r > matrix.c
    
    res = [[]]
    if mode == 'truncate':
        if rowLong:
            res = matrix.matrix[:matrix.c]
        else:
            res = [row[:matrix.r] for row in matrix.matrix]
    if mode == 'extend':
        if rowLong:
            res = [row + [0 for _ in range(matrix.r - matrix.c)] for row in matrix.matrix]
        else:
            res = matrix.matrix + [[0 for _ in range(matrix.c)] for _ in range(matrix.c - matrix.r)]
    
    return Matrix(res)

def uptri(matrix):
    '''
    Upper triangular matricies are matricies where all values below the diagonal line is 0
    
    in other words, element = 0 if row idx > col idx
    
    this function will return an uppertriangle matrix from a square matrix
    '''
    return Matrix([[0 if ridx > cidx else val for cidx, val in enumerate(row)] for ridx, row in enumerate(matrix.matrix)])

def lowtri(matrix):
    '''
    Lower triangular matricies are matricies where all values above the diagonal line is 0
    
    in other words, element = 0 if row idx < col idx
    
    this function will return a lower triangle matrix from a square matrix
    '''
    return Matrix([[0 if ridx < cidx else val for cidx, val in enumerate(row)] for ridx, row in enumerate(matrix.matrix)])

def summ(a, b):
    '''
    summation operations could be done between two Matricies with same dimentions.
    '''
    if a.r != b.r or a.c != b.c:
        return 'Matricies with different dimentions cannot be summed'
    
    # The calculation
    '''
    The summation operation is performed by adding each value of the same coordinate together
    as shown below.
    '''
    return Matrix([[aVal + bVal for aVal, bVal in zip(aRow, bRow)] for aRow, bRow in zip(a.matrix, b.matrix)])



def sub(a,b):
    '''
    subtraction operations could be done between two Matricies with same dimentions.
    '''
    if a.r != b.r or a.c != b.c:
        return 'Matricies with different dimentions cannot be subtracted'
    
    # The calculation
    '''
    The subtraction operation is performed by adding each value of the same coordinate together
    as shown below.
    '''
    return Matrix([[aVal - bVal for aVal, bVal in zip(aRow, bRow)] for aRow, bRow in zip(a.matrix, b.matrix)])



def scalarMulti(scalar, matrix):
    '''
    all matricies can be multiplied by a scalar, by multiplying each value in the matrix by that value
    '''
    return Matrix([[val*scalar for val in row] for row in matrix.matrix])



def additiveInverse(a): #[tag: ZeroMatrix, scalarMulti]
    '''
    all matricies have an additive inverse, which when added with each other will yield a zero matrix / null matrix
    '''
    return scalarMulti(-1, a)



def vectorMulti(a, b):
    '''
    Multiplications(Vector) can be performed between two matricies under the
    circumstances where 
    each row vector of the first matrix must have the same number of elements
    as each column vector of the second matrix.
    which means c(a) must be == to r(b)
    '''
    if a.c != b.r:
        return 'Matricies invalid for vector multiplication.'
    
    '''
    When a valid pair of matricies multiply, they does so by multiply
    each row vector to each column vector
    
    which is done by multiplying each corrisponding value and perform summation
    of all results
    
    For example:
                ⎡ 1 ⎤   
     [2 5 9] * | 3 | = [ 2*1 + 5*3 + 9*8 ] 
               ⎣ 8 ⎦   
    
    ''Therefore, the number of rows in matrix a defines the number of rows in 
    the result 
    and consecetively, the number of columns in matrix b defines the number 
    of columns in the result''
    
    in other words, each row in a multiplpies by each row in b(Transposed)
    
    simply:
        matrix = []
        for each rowA:
            row = []
            for each columnB:
                row.append(sum([valA*valB for valA, valB in rowA,colB]))
            matrix.append(row)
    '''
    return Matrix([[sum([valA * valB for valA,valB in zip(rowA,colB)]) for colB in T(b).matrix] for rowA in a.matrix])

def vectorMultiStr(a, b):#output the string
    return Matrix([[' {' + ' + '.join([str(valA) + ' * ' + str(valB) for valA,valB in zip(rowA,colB)]) + '} ' for colB in T(b).matrix] for rowA in a.matrix])
    

def multiInverse(matrix):#[tag: determinant, vector multiplication, identity matrix, scalar multiplication, adjoint matrix]
    '''
    A multiplicative inverse of a matrix is a matrix that will yield an identity matrix
    when vector multiplied with the original matrix
    
    A * A^-1 = I
    
    which can also be yielded by such:
        A**-1 = (1/det(A))*adj(A)
    '''
    if det(matrix) == 0:
        return None
    else:
        return scalarMulti((Fraction(1,det(matrix))),adj(matrix))

def mPower(matrix,power):
    if power == 1:
        return matrix
    else:
        return vectorMulti(matrix, mPower(matrix,power-1))

def systemOfEquations(coeffM, varV, constV):
    '''
    Matricies could be used to solve system of equations by assigning 3 matricies,
    where 2 of them are vectors.
    
    one coefficient matrix for the coefficeints,
    one variable vector to store the unknown variables,
    and one constant vector to store constants.
    
    vecMulti(coeffM, varV) = constV
    
    e.g.
        [2,-3,5]   [x1]   [3]
        [6, 2,7] * [x2] = [9]
        [-4,9,3]   [x3]   [1]
        
        x1 = x
        x2 = y
        x3 = z
        
        2x-3y+5z = 3
        6x+2y+7z = 9
        -4x+9y+3z= 1
    
    Operations:
        scalar multiply the entire row
        sum/subtract between rows
    
    Objective:
        make as much 0s as possible
    '''

def eigenValue(matrix):
    '''
    The eigenvalue of a matrix could be found with the characteristic equation:
        det(A-lambdaI) = 0
    
    Where:
        A => matrix
        lambda => eigenvalue
        I => Identity Matrix
    
    For example:
        Matrix: [2 5]
                [6 4]
        
        (2-lambda)*(4-lambda) - 5*6 = 0 
    
    The dimention of a matrix will determine the number of eigenvalues it has
    e.g. a 3x3 matrix will have 3 eigenvalues
    '''
    l = symbols('l')
    return solve(det(sub(matrix,scalarMulti(l, IdentityMatrix(matrix.r)))),l)

def eigenVector(matrix,eVIndex):
    '''
    By definition of eigen vectors and eigen values, it is said that it has to adhere to the equation:
        AX = lX
    
    where:
        A is the initial matrix
        X is the eigen matrix
        l is the eigen value
    
    so therefore, after we found an eigen value for the initial matrix
    for the corrisponding eigen value, we could find a list of eigen vectors
    '''
    eV = eigenValue(matrix)
    if eVIndex >= len(eV):
        print('Eigenvalue Index of Range')
    else:
        pass
        

l = symbols('l')

n = Matrix([[1-l,3,-1],
            [1,-1-l,-1],
            [2,-1,-2-l]])

m = Matrix([[1,3,-1],
            [1,-1,-1],
            [2,-1,-2]])
print(solve(det(n),l))

print(eigenValue(m))
    


a = Matrix([[1,4,3],
            [6,2,5]])

b = Matrix([[1,2],
            [6,2],
            [3,7]])

c = Matrix([[3,7,2],
            [8,7,3]])

d = Matrix([[1,2,3,4],
            [5,6,7,8],
            [2,3,7,2],
            [6,5,4,9]])

e = Matrix([[1,2,3,4],
            [5,6,100000,8],
            [2,3,7,2],
            [6,5,4,9]])

f = Matrix([[1,2,3,4]])

I3 = IdentityMatrix(3)

diag = DiagonalMatrix([1,2,6,3,5,1])

#print(f.form())

#print(additiveInverse(b).form(),'\n', b.form(), '\n' ,summ(additiveInverse(b),b).form())
#print(vectorMulti(a,I3).form())
#print(ZeroMatrix(10,2).form())
#print(IdentityMatrix(10).form())
#print(diag.form())      
#print(uptri(lowtri(square(b, 'extend'))).form())

'''
A transpose function performed on a matrix is spreaded through each element within a bracket if done so

T(a + b) = T(a) + T(b)

T(a * b) = T(b) * T(a)  (The position of the matrix has to be changed due to the change in number of rows and number of columns)

T(scalarMultiplication(k, a)) = scalarMultiplication(k, T(a)) 
'''

#print(T(summ(a,c)).form(), '\n', summ(T(a),T(c)).form())
#print(T(vectorMulti(a, b)).form(), '\n', vectorMulti(T(b), T(a)).form())

#print(detM([[1,2,3,4],[5,6,7,8],[2,3,7,2],[6,5,4,9]]))
#print(det(d))
#print(submatrix(square(a),1,1).form())


'''
Multiplicative Inverse Demonstartions

print(d.form(2))
print(det(d))
print(adj(d).form(2))
print(multiInverse(d).form(2))
print(scalarMulti(2,multiInverse(d)).form(2))
print(multiInverse(scalarMulti(2,d)).form(2))
print(vectorMulti(d,multiInverse(d)).form(2))
'''
'''
a = Matrix([[1,2,4],[-2,5,3],[1,0,2]])

b = Matrix([[-19,-23,13],[20,25,-2],[46,35,-10]])

print(a.form(2))
print(b.form(2))

print(vectorMulti(a, a).form())
print(T(b).form())

c = sub(vectorMulti(a, a),T(b))
print(c.form())
print(vectorMultiStr(a, c).form())
print(vectorMulti(a, c).form())
print(vectorMulti(a, scalarMulti(Fraction(1,8),c)).form())
print(scalarMulti(Fraction(1,8),c).form(),multiInverse(a).form())
'''

'''
A = Matrix([[1,3,-1],[1,-1,-1],[2,-1,-2]])
X = Matrix([[-1,3,5]])
P = Matrix([[-1,1,5],[3,0,1],[5,1,3]])
D = vectorMulti(vectorMulti(multiInverse(P),A),P)
D7 = mPower(D,7)

print(det(P))
print(P.form())
print(cof(P).form())
print(adj(P).form())
print(multiInverse(P).form())
print(vectorMulti(multiInverse(P),A).form())
print(vectorMulti(vectorMulti(multiInverse(P),A),P).form())
print(mPower(A,7).form())
print(D7.form())
print(vectorMulti(P,D7).form(2))
print(vectorMulti(vectorMulti(P,D7),multiInverse(P)).form(2))
'''