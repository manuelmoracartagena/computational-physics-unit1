# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 18:32:30 2023

@author: Manuel Mora Cartagena 
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import animation



def condicion_incial(x): #Define la función que da la condición inicial
    return np.exp(-10*(x-1)**2)


def centradas(): #Define la función que calcula la solución con diferencias centradas
    diag = np.ones(n+1) #Define la diagonal
    on = -(dt/(2*dx))*np.ones(n) #Define la diagonal superior
    off = (dt/(2*dx))*np.ones(n) #Define la diagonal inferior
    A = scipy.sparse.diags([diag, on, off], [0,1,-1],shape=(n+1,n+1)).toarray() #Define la matriz
    u0 = condicion_incial(x) #Define la condicion inicial
    u = [] #Define una lista para las soluciones
    u.append(u0) #Añade a la lista la condición inicial
    for i in range(m+1): #Define el bucle para calcular la solucion en cada instante de tiempo
        u_temp = np.dot(A, u0) #Calcula la solucion en el instante i
        u.append(u_temp) #Añade la solucion calculada a la lista de soluciones
        u0 = u_temp #Actualiza para la siguiente iteración
    return u


def upwind(): #Define la función que calcula la solución con upwind
    diag = (1 - (dt/dx))*np.ones(n+1) #Define la diagonal
    off = (dt/dx)*np.ones(n) #Define la diagonal inferior
    A = scipy.sparse.diags([diag,off], [0,-1],shape=(n+1,n+1)).toarray() #Define la matriz
    u0_0 = condicion_incial(x) #Define la condicion inicial
    u = [] #Define una lista para las soluciones
    u.append(u0_0) #Añade a la lista la condición inicial
    for i in range(m+1): #Define el bucle para calcular la solucion en cada instante de tiempo
        u_temp = np.dot(A, u0_0) #Calcula la solucion en el instante i
        u.append(u_temp) #Añade la solucion calculada a la lista de soluciones
        u0_0 = u_temp #Actualiza para la siguiente iteración
    return u


def downwind(): #Define la función que calcula la solución con downwind
    diag = (1 + (dt/dx))*np.ones(n+1) #Define la diagonal
    on = -(dt/dx)*np.ones(n) #Define la diagonal superior
    A = scipy.sparse.diags([diag,on], [0,1],shape=(n+1,n+1)).toarray() #Define la matriz
    u0_0 = condicion_incial(x) #Define la condicion inicial
    u = [] #Define una lista para las soluciones
    u.append(u0_0) #Añade a la lista la condición inicial
    for i in range(m+1): #Define el bucle para calcular la solucion en cada instante de tiempo
        u_temp = np.dot(A, u0_0) #Calcula la solucion en el instante i
        u.append(u_temp) #Añade la solucion calculada a la lista de soluciones
        u0_0 = u_temp #Actualiza para la siguiente iteración
    return u


def lax_wendroff():
    diag = (1 - (dt/dx)**2)*np.ones(n+1) #Define la diagonal
    on = (((dt/dx)**2/2) - (dt/(2*dx)))*np.ones(n) #Define la diagonal superior
    off = (((dt/dx)**2/2) + (dt/(2*dx)))*np.ones(n) #Define la diagonal inferior+
    A = scipy.sparse.diags([diag,on,off], [0,1,-1],shape=(n+1,n+1)).toarray() #Define la matriz
    u0_0 = condicion_incial(x) #Define la condicion inicial
    u = [] #Define una lista para las soluciones
    u.append(u0_0) #Añade a la lista la condición inicial
    for i in range(m+1): #Define el bucle para calcular la solucion en cada instante de tiempo
        u_temp = np.dot(A, u0_0) #Calcula la solucion en el instante i
        u.append(u_temp) #Añade la solucion calculada a la lista de soluciones
        u0_0 = u_temp #Actualiza para la siguiente iteración
    return u

    
    


#Funciones para la animación
def init():
    line.set_data([], [])
    return line,


def animate_centradas(i):
    line.set_data(x, centradas()[i])
    return line,


def animate_upwind(i):
    line.set_data(x, upwind()[i])
    return line,


def animate_downwind(i):
    line.set_data(x, downwind()[i])
    return line,


def animate_lax_wendroff(i):
    line.set_data(x, lax_wendroff()[i])
    return line,
    
    
    
    
    
      
#Datos del problema 
L = 10 #Define el limite en el dominio espacial
t = 12 #Define el limite en el dominio  temporal
dx = 0.1 #Define la variación espacial
dt = 0.05 #Define la variación temporal
n = int(L/dx) #Define el número de puntos en los que se discretiza el espacio
m = int(t/dt) #Define el número de puntos en los que se discretiza el tiempo
x = np.arange(0, L+dx, dx) #Define el intervalo espacial


#Animación del método diferencias centradas
fig1 = plt.figure()
ax1 = plt.axes(xlim=(-0.1, 10.1), ylim=(-1.5, 1.5))
plt.title('Ecuación de advección con diferencias centradas')
plt.xlabel('x')
plt.ylabel('u')
line, = ax1.plot([], [], lw=2, color = 'blue')
anim = animation.FuncAnimation(fig1, animate_centradas, init_func=init, frames=m, interval=1, blit=True)
plt.show()

#Animación del método upwind
fig2 = plt.figure()
ax2 = plt.axes(xlim=(-0.1, 10.1), ylim=(-1.5, 1.5))
plt.title('Ecuación de advección con upwind')
plt.xlabel('x')
plt.ylabel('u')
line, = ax2.plot([], [], lw=2, color = 'red')
anim = animation.FuncAnimation(fig2, animate_upwind, init_func=init, frames=m, interval=1, blit=True)
plt.show()

#Animación del método downwind
fig3 = plt.figure()
ax3 = plt.axes(xlim=(-0.1, 10.1), ylim=(-1.5, 1.5))
plt.title('Ecuación de advección con downwind')
plt.xlabel('x')
plt.ylabel('u')
line, = ax3.plot([], [], lw=2, color = 'green')
anim = animation.FuncAnimation(fig3, animate_downwind, init_func=init, frames=m, interval=1, blit=True)
plt.show()

#Animación del método lax wendroff
fig4 = plt.figure()
ax4 = plt.axes(xlim=(-0.1, 10.1), ylim=(-1.5, 1.5))
plt.title('Ecuación de advección con Lax Wendroff')
plt.xlabel('x')
plt.ylabel('u')
line, = ax4.plot([], [], lw=2, color = 'purple')
anim = animation.FuncAnimation(fig4, animate_lax_wendroff, init_func=init, frames=m, interval=1, blit=True)
plt.show()

'''
#Dibujo en diferentes instantes para cada método para el informe
#Método diferencias centradas
for i in range(30, 211, 60):
    plt.figure()
    ax = plt.axes(xlim=(-0.1, 10.1), ylim=(-1.5, 1.5))
    plt.title('Ecuación de advección con diferencias centradas (Tiempo = '+ str(int(i*dt)) +'s)')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.plot(x, centradas()[i], color = 'blue')
    plt.show()

#Método upwind
for i in range(20, 141, 40):
    plt.figure()
    ax = plt.axes(xlim=(-0.1, 10.1), ylim=(-1.5, 1.5))
    plt.title('Ecuación de advección con upwind (Tiempo = '+ str(int(i*dt)) +'s)')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.plot(x, upwind()[i], color = 'red')
    plt.show()

#Método downwind
for i in range(30, 211, 60):
    plt.figure()
    ax = plt.axes(xlim=(-0.1, 10.1), ylim=(-1.5, 1.5))
    plt.title('Ecuación de advección con downwind (Tiempo = '+ str(int(i*dt)) +'s)')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.plot(x, downwind()[i], color = 'green')
    plt.show()

#Método lax wendroff
for i in range(20, 141, 40):
    plt.figure()
    ax = plt.axes(xlim=(-0.1, 10.1), ylim=(-1.5, 1.5))
    plt.title('Ecuación de advección con Lax Wendroff (Tiempo = '+ str(int(i*dt)) +'s)')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.plot(x, lax_wendroff()[i], color = 'purple')
    plt.show()
'''
 