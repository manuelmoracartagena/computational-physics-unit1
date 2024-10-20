# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 19:13:08 2023

@author: Manuel Mora Cartagena
"""

import numpy as np
import scipy
from scipy import sparse
import matplotlib.pyplot as plt

#Resolución sistema de ecuaciones en forma matricial de EDP en 2D
#Ecuacion de Laplace y Poisson en 2D
#Apartado 1: Método diferencias finitas

L = 100 #Esto es la longitud en la que se cálcula el valor del potencial
h = 1 #Defino el valor de h. Esto es la separación entre puntos

#Defino la matriz tridiagonal del sistema de EDP
M = int(L/h) #Dimensión de los bloques de la matriz
diag = -4*np.ones(M) #Defino la diagonal de la matriz tridiagonal
off = np.ones(M-1) #Defino las diagonales superior e inferior a la diagonal principal. En esto caso son iguales
A = sparse.diags([diag, off, off], [0,-1,1],shape=(M,M)).toarray() #Genero la matriz tridiagonal
#Si se omite shape se genera una matriz cuadrada lo suficientemente grande para que pueda contener la diagonal

Identity = sparse.eye(M).toarray() #Calculo la matriz indentidad de dimensión M
#sparse.eye calcula la matriz dispersa con unos en la diagonal, es decir, la matriz indentidad

A2 = sparse.kron(Identity, A).toarray() #Calculo el producto tensorial entre la matriz indentidad y la matriz A
#sparse.kron calcula el producto de kronecker de dos matrices dispersas A y B

#Ahora voy a meter los bloques de matrices indentidad de los lados
B = sparse.diags([off, off], [-1, 1], shape=(M,M)).toarray()
B2 = sparse.kron(B,Identity).toarray()


#Sumo las dos matrices a bloques que he generado y obtengo la matriz final
C = A2+B2

#Impongo las condiciones de contorno en la matriz C

#Para que la primera fila sea 100
for i in range(0, M):
    C[i,:] = 0
    C[i,i] = 1
        

#Para que la última fila sea 0
for i in range(M*M-M, M*M):
    C[i,:] = 0
    C[i,i] = 1
    
    
#Para que la primera columna sea 0
for i in range(M, M*M-M, M):
    C[i,:] = 0
    C[i,i] = 1
 
    
#Para que la última columna sea 0
for i in range(2*M-1, M*M-M, M):
    C[i,:] = 0
    C[i,i] = 1







#Defino el vector de términos independientes
b = np.zeros(M*M)


#Impongo ahora las condiciones de contorno al vector de términos independientes b
for i in range(M):
    b[i] = 100*h**2



#Resolvemos Cx = b
#Con el siguiente código resolvemos el sistema de ecuaciones lineales en forma matricial de una matriz dispersa
potencial = scipy.sparse.linalg.spsolve(C, b)
potencial2 = potencial.reshape(M, M) #Lo paso de vector columna N^2x1 a matriz NxN




#Dibujamos el resultado con countour

x = np.linspace(0, L, M)
y = np.linspace(0, L, M)
X,Y = np.meshgrid(x, y)

plt.figure()
plt.contourf(X, Y, potencial2, 10, alpha=.75, cmap=plt.cm.hot)
plt.colorbar(label = 'Potencial (V)')
L = plt.contour(X, Y, potencial2, 10, colors='black', linewidths=1) 
plt.clabel(L, inline=1, fontsize=10)
plt.title('Método de diferencias finitas')
plt.xticks(())
plt.yticks(())



#Apartado 2: Método Jacobi

error = 10**(-6) #Defino el error
#Genero la matriz con las condiciones de contorno  
jacobi = np.zeros((M, M))
jacobi[0, :] = 100
for i in range(1, M-1):
    for j in range(1, M-1):
        jacobi[i, j] = np.random.rand()

jacobi2 = np.copy(jacobi) #Genero una copia


continua = True
while continua:
    continua = False
    for i in range(1, M-1):
        for j in range(1, M-1):
            jacobi2[i, j] = 1/4*(jacobi[i+1, j] + jacobi[i-1, j]+jacobi[i, j+1]+jacobi[i, j-1])
    for i in range(1, M-1):
        for j in range(1, M-1):
            if jacobi2[i, j] - jacobi[i, j] > error:
                jacobi = np.copy(jacobi2)
                continua = True
    
            
#Dibujamos con contour
    
plt.figure()
plt.contourf(X, Y, jacobi2, 10, alpha=.75, cmap=plt.cm.hot)
plt.colorbar(label = 'Potencial (V)')
L = plt.contour(X, Y, jacobi2, 10, colors='black', linewidths=1) 
plt.clabel(L, inline=1, fontsize=10)
plt.title('Método de Jacobi')
plt.xticks(())
plt.yticks(())
              
          
                
                

#Apartado 3: Condiciones de contorno de Neumann

D = A2+B2 #Genero la matriz D para imponer las nuevas condiciones de contorno

#Impongo las condiciones de contorno en la matriz D

#Para que la primera fila sea 100
for i in range(0, M):
    D[i,:] = 0
    D[i,i] = 1
        

#Para que la última fila sea 0
for i in range(M*M-M, M*M):
    D[i,:] = 0
    D[i,i] = 1


#Ahora impongo las condiciones de contorno en las paredes verticales
'''
#La derivada parcial del potencial respecto a la coordenada vertical es 0
#Como estoy aproximando con diferencias finitas, esto es equivalente a decir que el elemento ij
es igual al elemento adyacente y así sucesivamente
'''

#Para la primera columna 
for i in range(M, M*M-M, M):
    D[i,:] = 0
    D[i,i] = -1
    D[i, i+1] = 1
 
    
#Para la última columna
for i in range(2*M-1, M*M-M, M):
    D[i,:] = 0
    D[i,i] = -1
    D[i, i-1] = 1




#Resolvemos Dx = b
potencial_neumann = scipy.sparse.linalg.spsolve(D, b)
potencial2_neumann = potencial_neumann.reshape(M, M)


#Dibujamos con contour
    
plt.figure()
plt.contourf(X, Y, potencial2_neumann, 10, alpha=.75, cmap=plt.cm.hot)
plt.colorbar(label = 'Potencial (V)')
L = plt.contour(X, Y, potencial2_neumann, 10, colors='black', linewidths=1) 
#plt.clabel(L, inline=1, fontsize=10)
plt.title('Condiciones de Contorno de Neumann')
plt.xticks(())
plt.yticks(())

plt.show()