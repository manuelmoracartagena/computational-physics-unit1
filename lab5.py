# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 21:51:45 2023

@author: Manuel Mora Cartagena
"""
#Resolución de la ecuación de Schrödinger con el método de Crank-Nicholson

import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import animation

#Define  una función para generar la condicion inicial
def condicion_inicial(): 
    return np.exp((-1/2)*(x/sigma)**2)*np.exp(1j*k*x)

#Funciones para la animación
def init():
    line.set_data([], [])
    return line,


def animate(i):
    line.set_data(x, u[i])
    return line,


#Parámetros del problema
L = 12 #Define el limite en el dominio espacial
t = 0.5 #Define el limite en el dominio  temporal
dx = 0.02 #Define la variación espacial
dt = 0.5*(dx**2) #Define la variación temporal
n = int(L/dx) #Define el número de puntos en los que se discretiza el espacio
m = int(t/dt) #Define el número de puntos en los que se discretiza el tiempo
x = np.arange(-L/2, L/2+dx, dx) #Define el intervalo espacial
k = 15*np.pi
sigma = 0.5

#Generación de matrices para aplicar el método de Crank-Nicholson
r = (1j*dt)/(2*dx**2) #Define una constante para simplficar
#Defino la matriz que multiplica a u en k+1 (la de la izquierda)
diag1 = (1 + 2*r)*np.ones(n+1) #Define la diagonal principal de la matriz
off1 = -r*np.ones(n) #Define la diagonal superior e inferior de la matriz, en este caso son iguales
A1 = scipy.sparse.diags([diag1, off1, off1], [0,-1,1],shape=(n+1,n+1), dtype = complex).toarray() #Define la matriz tridiagonal
#Defino la matriz que multiplica a u en k (la de la derecha)
diag2 = (1 - 2*r)*np.ones(n+1) #Define la diagonal principal de la matriz
off2 = r*np.ones(n) #Define la diagonal superior e inferior de la matriz, en este caso son iguales
A2 = scipy.sparse.diags([diag2, off2, off2], [0,-1,1],shape=(n+1,n+1), dtype = complex).toarray() #Define la matriz tridiagonal
B = np.dot(np.linalg.inv(A1), A2) #Genera la matriz final del sistema de ecuaciones
#Modifico B para aplicar las condiciones de contorno
B[:,0] = 0
B[:,-1] = 0
#Genero un array para la solucón y aplico las condiciones iniciales
u = np.zeros([m+1, n+1], dtype = complex) #Define un array para la solución
u[0] = condicion_inicial() #Define la condicion 
#Aplico el método de Crank-Nicholson
for i in range(m): #Calcula la solución en cada instante de tiempo i
    u[i+1] = np.dot(B, u[i])

#Animación
fig = plt.figure()
ax = plt.axes(xlim=(-L/2, L/2), ylim=(-1.25, 1.25))
plt.title('Ecuación de Schrödinger')
plt.xlabel('x')
plt.ylabel('u')
line, = ax.plot([], [], lw=2, color = 'blue')
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=m, interval=10, blit=True)
plt.show()

'''
#Dibujo en diferentes instantes de tiempo para el informe
for i in range(30, 211, 60):
    tiempo = round(i*dt, 2)
    plt.figure()
    ax = plt.axes(xlim=(-L/2, L/2), ylim=(-1.25, 1.25))
    plt.title('Ecuación de Schrödinger (Tiempo = '+ str(tiempo) +'s)')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.plot(x, u[i], color = 'blue')
    plt.show()
'''