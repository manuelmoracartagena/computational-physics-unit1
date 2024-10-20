# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:46:55 2023

@author: Manuel Mora Cartagena
"""

import numpy as np
import matplotlib.pyplot as plt
import time



#BLOQUE I
#Ejercicio 1
#Se trata de resolver la ecuación matricial Ax = v



#Apartado a
def TridiagonalSolver(d, o, u, r): #Funcion a la que le entran 4 arrays
    n = len(d) #Calculo la dimension 3 n
    h = np.ones(n-1) #Genero un array de dimension n-1
    p = np.ones(n) #Genero un array de dimension n
    #Calculo los valores de h y p iniciales
    h[0] = h[0]*o[0]/d[0]
    p[0] = p[0]*r[0]/d[0]
    #Calculo ahora los valores de h y p recursivamente
    for i in range(1, n-1):
        h[i] = h[i]*(o[i]/(d[i] - u[i-1]*h[i-1]))
        p[i] = p[i]*((r[i] - u[i-1]*p[i-1])/(d[i] - u[i-1]*h[i-1]))
    p[n-1] = p[n-1]*((r[n-1] - u[n-2]*p[n-2])/(d[n-1] - u[n-2]*h[n-2]))

    x = np.ones(n) #Genero el array de soluciones x de dimensión n
    x[n-1] = p[n-1] #Calculo ahora el último elemento de x
    for i in range(n-2, -1, -1): #Calculo los valores de x recursivamente y hacia atrás
        x[i] = p[i] - h[i]*x[i+1]
    return x  

dimensiones = []
tiempos1 = []
tiempos2 = []
tiempos3 = []
for dimension in range(100, 10001, 100):
    #Definimos la matriz tridiagonal aleatoria
    d = np.random.random(dimension)
    o = np.random.random(dimension-1)
    u = np.random.random(dimension-1)
    r = np.random.random(dimension)
    
    matriz = np.random.random((dimension, dimension))#Definimos la matriz aleatoria
    
    #Resolvemos la matriz tridiagonal
    inicio = time.time()
    x1 = TridiagonalSolver(d, o, u, r)
    fin = time.time()
    tiempos1.append(fin-inicio)

    #Resolvemos calculando la matriz inversa
    inicio = time.time()
    invmatriz = np.linalg.inv(matriz)
    x2 = np.dot(invmatriz, r)
    fin = time.time()
    tiempos2.append(fin-inicio)
    
    #Resolvemos con la rutina solve
    inicio = time.time()
    x3 = np.linalg.solve(matriz, r)
    fin = time.time()
    tiempos3.append(fin-inicio)

    dimensiones.append(dimension)



#Apartado b
#Esto fue lo primero que hice: calcularlo con las librerias de numpy para una sola matriz

A = np.array([[1, 4, 0, 0], [3, 4, 1, 0], [0, 2, 3, 4], [0, 0, 1, 3]])#Defino una matriz
v = np.array([6, 2, 4, 7])#Defino los términos independientes

invA = np.linalg.inv(A)#Calculo la inversa de la matriz
x2 = np.dot(invA, v) #Calculo el producto de las matrices invA y v


x3 = np.linalg.solve(A, v) #Usamos el segundo método para calcular la solución de la ec.matricial


plt.figure()
plt.plot(dimensiones, tiempos1, 'b', label = 'Algoritmo de Thomas')
plt.plot(dimensiones, tiempos2, 'r', label = 'Calculando la matriz inversa')
plt.plot(dimensiones, tiempos3, 'g', label = 'Mediante la rutina solve')
plt.xlabel('Dimensión de la matriz')
plt.ylabel('Tiempo de cálculo (s)')
plt.title('Tiempo de cálculo en función de la dimensión de la matriz')
plt.legend(loc = 'best')
plt.show()

