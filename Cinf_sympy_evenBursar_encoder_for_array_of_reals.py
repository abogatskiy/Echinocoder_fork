# Encode list of real numbers treated as multiset.
# E.g. this method can encode things like:
#
#        [[3,2],[4,1],[-2,1]]
#
# Although this implementation claims to only operate on arrays of reals, 
# it might also be able to encode complex arrays, albeit to complex 
# outputs. This could be used by a complexly compressed method later?
# 
# The number of outputs shoud be 
#
#          ORDER(m,n) == Binom(m+n, n) - 1 
#
# where n is the number of vectors in the set, and m is the dimension of each of those vectors. 
# It is called "evenBursar" to contrast it from "bursar" since the former treats all elements 
# of the R^m vectors even-handedly, whereas the latter treats the top and bottom compoents of
# R^m vectors specially.
#
# Here is an $m=3$, $n=2$ example:
# Encoding  $\{(3,1,4),(2,2,5)\}$ should generate
# 
#       [9, 20, 3, 13, 2, 5, 23, 8, 6]
#
# since the polynomial
#
#       $(y + (3*x0 + 1*x1 + 4*x2))(y + (2*x0 + 2*x1 + 5*x2))$
#
# is equal to 
#
#       $ + 1 y^2 + 9 x0 y + 20 x2^2 + 3 x2 y + 13 x1 x2 + 2 x1^2 + 5 x1 y + 23 x0 x2 + 8 x0 x1 + 6 x0^2 $
#
# .

name="Cinf_sympy_evenBursar_encoder_for_array_of_reals"

from math import prod
#from itertools import combinations
#import numpy as np
#import toolsA
import sympy

# get_tuples gets list of tuples of fixed length and fixed total over the non-negativ3 integers.
# E.G.:
# >>> list(get_tuples(3, 2))
# [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]
def get_tuples(length, total):
    if length == 1:
        yield (total,)
        return

    for i in range(total + 1):
        for t in get_tuples(length - 1, total - i):
            yield (i,) + t


def encode(data):
    
    #print("Data is")
    #print(data)

    n = len(data)
    if n==0:
        return []

    m = len(data[0])
    if m==0:
        return []

    #part = sum( [ sympy.Poly(sympy.symbols("x"+str(i)))*val for i, val in enumerate(data[0]) ] )
    #print("******** FIRST PART IS ",part)
    polys = [ 
      sum( [ sympy.Poly(sympy.symbols("x"+str(i)))*val for i, val in enumerate(vector) ] )
      + sympy.Poly(sympy.abc.y) for vector in data ]

    #print("****** polys",polys)

    product = prod(polys)
    print ("**** Product",product)

    # # extract coeffs from product slowly:
    # for c in range(1,n+1):
    #      yPower = n-c
    #      for xPower in range(c*(m-1)+1):
    #          coeff = product.coeff_monomial((abc.x)**xPower * (abc.y)**yPower) # This could be made faster by using coeff_monomial((xPower,yPower)) or coeff_monomial((yPower,xPower)) however I have not figured out how to guarantee which order will work reliably.
    #          print(coeff)


    monomials = get_tuples(m+1, n)
    itermonomials = iter(monomials)
    firstMonomial = next(itermonomials)
    # Confirm that coeff of first monomial is one .. the y^n term:
    firstCoeff = product.coeff_monomial(firstMonomial)
    if firstCoeff!=1:
        print("Expecfted 1 for coeffiecient of y^n term but got ",firstCoeff)
        raise Exception("Bug in implementation of Bursar's method!")
    # Now that we skipped that term:    
    coeffs = [ product.coeff_monomial( monomial ) for monomial in itermonomials ]

    # # extract coeffs from product fast:
    # coeffs = [ [
    #          product.coeff_monomial((sympy.abc.x)**xPower * (sympy.abc.y)**(n-c) ) # n-c==yPower.  This line could be made faster by using coeff_monomial((xPower,yPower)) or coeff_monomial((yPower,xPower)) however I have not figured out how to guarantee which order will work reliably.
    #            for xPower in range(c*(m-1)+1) ]
    #            for c in range(1,n+1) ]

    print("**** Coeffs",coeffs)
    
    return coeffs

    #xPolys = [ np.polynomial.Polynomial(vector) for vector in data ] 
    #print("x polynomials are: ")
    #for xPoly in xPolys:
    #    print (xPoly)
    # 
    #n=len(data)

    #return [ sum( [ prod(c)  for c in combinations(data, r+1) ] ) for r in range(len(data)) ]

    # return [ sum( [ prod(c)  for c in combinations(data, r+1) ] ) for r in range(len(data)) ]
    
    # Alternative
    # 
    # ans = np.array([ sum( [ prod(c)  for c in combinations(data, r+1) ] ) for r in range(len(data)) ])
    #
    # All encoders have to output lists of real numbers (at least for now) so:
    #if np.iscomplexobj(ans):
    #  ans=tools.expand_complex_to_real_pairs(ans)
    #
    # return ans



