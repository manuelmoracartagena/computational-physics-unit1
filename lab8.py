# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 12:40:53 2023

@author: Manuel Mora Cartagena
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def condicion_incial():
    return 3*np.sin((1*np.pi*x)/L)


def centradas():
    u = np.zeros((m+1, n+1)) #Define un array de ceros de dimension m+1 filas y n+1 columnas
    u[0] = condicion_incial() #Define la condición inicial y la añade al array
    for k in range(m): #Bucle que calcula la soluciçón en cada instante de tiempo
        u[k+1, 1:-1] = u[k, 1:-1]*(1 - (dt/(2*dx))*(u[k, 2:] - u[k, :-2]))
    return u


def upwind():
    u = np.zeros((m+1, n+1)) #Define un array de ceros de dimension m+1 filas y n+1 columnas
    u[0] = condicion_incial() #Define la condición inicial y la añade al array
    for k in range(m): #Bucle que calcula la soluciçón en cada instante de tiempo
        u[k+1, 1:-1] = u[k, 1:-1]*(1 - (dt/(2*dx))*(u[k, 1:-1] - u[k, :-2]))
    return u


def downwind():
    u = np.zeros((m+1, n+1)) #Define un array de ceros de dimension m+1 filas y n+1 columnas
    u[0] = condicion_incial() #Define la condición inicial y la añade al array
    for k in range(m): #Bucle que calcula la soluciçón en cada instante de tiempo
        u[k+1, 1:-1] = u[k, 1:-1]*(1 - (dt/(2*dx))*(u[k, 2:] - u[k, 1:-1]))
    return u


def conservativo():
    u = np.zeros((m+1, n+1)) #Define un array de ceros de dimension m+1 filas y n+1 columnas
    u[0] = condicion_incial() #Define la condición inicial y la añade al array
    for k in range(m): #Bucle que calcula la soluciçón en cada instante de tiempo
        u[k+1, 1:-1] = u[k,1:-1]-0.1/4*(u[k,2:]**2-u[k,:-2]**2) + 0.1**2/8*((u[k,2:]+u[k,1:-1])*(u[k,2:]**2-u[k,1:-1]**2)-(u[k,1:-1]+u[k,:-2])*(u[k,1:-1]**2-u[k,:-2]**2))
    return u


def periodicas():
    u = np.zeros((m+1, n+1)) #Define un array de ceros de dimension m+1 filas y n+1 columnas
    u[0] = condicion_incial() #Define la condición inicial y la añade al array
    u[0, -1] = u[0, 0] #Intento de implementar condiciones de contorno para el primer instante de tiempo
    for k in range(m): #Bucle que calcula la soluciçón en cada instante de tiempo
        u[k+1, 1:-1] = u[k,1:-1]-0.1/4*(u[k,2:]**2-u[k,:-2]**2) + 0.1**2/8*((u[k,2:]+u[k,1:-1])*(u[k,2:]**2-u[k,1:-1]**2)-(u[k,1:-1]+u[k,:-2])*(u[k,1:-1]**2-u[k,:-2]**2))
        #u[k+1, -1] = u[k+1, 0] #Intento de implementar condiciones de contorno en cada instante de tiempo
        u[k+1, 0]= u[k, -2]
        u[k, -1]= u[k, 1]
    return u
    

#Funciones para animaciones
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


def animate_conservativo(i):
    line.set_data(x, conservativo()[i])
    return line,


def animate_periodicas(i):
    line.set_data(x, periodicas()[i])
    return line,




#Datos del problema 
L = 10 #Define el limite en el dominio espacial
t = 10 #Define el limite en el dominio  temporal
dx = 0.1 #Define la variación espacial
dt = 0.05 #Define la variación temporal
n = int(L/dx) #Define el número de puntos en los que se discretiza el espacio
m = int(t/dt) #Define el número de puntos en los que se discretiza el tiempo
x = np.arange(0, L+dx, dx) #Define el intervalo espacial


#Animación del método diferencias centradas
fig1 = plt.figure()
ax1 = plt.axes(xlim=(-0.1, L+0.1), ylim=(-4, 4))
plt.title('Ecuación de Burguers con diferencias centradas')
plt.xlabel('x')
plt.ylabel('u')
line, = ax1.plot([], [], lw=2, color = 'blue')
anim = animation.FuncAnimation(fig1, animate_centradas, init_func=init, frames=m, interval=10, blit=True)
plt.show()

#Animación del método upwind
fig2 = plt.figure()
ax2 = plt.axes(xlim=(-0.1, L+0.1), ylim=(-4, 4))
plt.title('Ecuación de Burguers con upwind')
plt.xlabel('x')
plt.ylabel('u')
line, = ax2.plot([], [], lw=2, color = 'red')
anim = animation.FuncAnimation(fig2, animate_upwind, init_func=init, frames=m, interval=10, blit=True)
plt.show()

#Animación del método downwind
fig3 = plt.figure()
ax3 = plt.axes(xlim=(-0.1, L+0.1), ylim=(-4, 4))
plt.title('Ecuación de Burguers con downwind')
plt.xlabel('x')
plt.ylabel('u')
line, = ax3.plot([], [], lw=2, color = 'green')
anim = animation.FuncAnimation(fig3, animate_downwind, init_func=init, frames=m, interval=10, blit=True)
plt.show()

#Animación del método conservativo
fig4 = plt.figure()
ax4 = plt.axes(xlim=(-0.1, L+0.1), ylim=(-4, 4))
plt.title('Ecuación de Burguers con esquema conservativo')
plt.xlabel('x')
plt.ylabel('u')
line, = ax4.plot([], [], lw=2, color = 'purple')
anim = animation.FuncAnimation(fig4, animate_conservativo, init_func=init, frames=m, interval=0.01, blit=True)
plt.show()

#Animación con condiciones periódicas
fig5 = plt.figure()
ax5 = plt.axes(xlim=(-0.1, L+0.1), ylim=(-4, 4))
plt.title('Ecuación de Burguers con condiciones periódicas')
plt.xlabel('x')
plt.ylabel('u')
line, = ax5.plot([], [], lw=2, color = 'orange')
anim = animation.FuncAnimation(fig5, animate_periodicas, init_func=init, frames=m, interval=0.01, blit=True)
plt.show()

'''
#Dibujo en diferentes instantes para cada método para el informe
#Método diferencias centradas
for i in range(0, 31, 10):
    plt.figure()
    ax1 = plt.axes(xlim=(-0.1, L+0.1), ylim=(-4, 4))
    plt.title('Ecuación de Burguers con diferencias centradas (Tiempo = '+ str((i*dt)) +'s)')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.plot(x, centradas()[i], color = 'blue')
    plt.show()
    
#Método upwind
for i in range(0, 31, 10):
    plt.figure()
    ax1 = plt.axes(xlim=(-0.1, L+0.1), ylim=(-4, 4))
    plt.title('Ecuación de Burguers con upwind (Tiempo = '+ str((i*dt)) +'s)')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.plot(x, upwind()[i], color = 'red')
    plt.show()
    
#Método downwind
for i in range(0, 31, 10):
    plt.figure()
    ax1 = plt.axes(xlim=(-0.1, L+0.1), ylim=(-4, 4))
    plt.title('Ecuación de Burguers con downwind (Tiempo = '+ str((i*dt)) +'s)')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.plot(x, downwind()[i], color = 'green')
    plt.show()
    
#Método conservativo
for i in range(0, 101, 20):
    plt.figure()
    ax1 = plt.axes(xlim=(-0.1, L+0.1), ylim=(-4, 4))
    plt.title('Ecuación de Burguers con esquema conservativo (Tiempo = '+ str((i*dt)) +'s)')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.plot(x, conservativo()[i], color = 'purple')
    plt.show()
    
#Condiciones periódicas
for i in range(0, 101, 20):
    plt.figure()
    ax1 = plt.axes(xlim=(-0.1, L+0.1), ylim=(-4, 4))
    plt.title('Ecuación de Burguers con condiciones periódicas (Tiempo = '+ str((i*dt)) +'s)')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.plot(x, conservativo()[i], color = 'orange')
    plt.show()
'''