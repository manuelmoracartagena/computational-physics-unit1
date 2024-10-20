# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 17:20:16 2023

@author: Manuel Mora Cartagena
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import animation


def difusion_ftcs():
    #Define la matriz
    diag = -2*np.ones(n) #Define la diagonal de la matriz tridiagonal
    off = 1*np.ones(n-1) #Define las diagonales superior e inferior a la diagonal principal, en esto caso son iguales
    A = scipy.sparse.diags([diag, off, off], [0,-1,1],shape=(n,n)).toarray() #Define la matriz tridiagonal
    I = scipy.sparse.eye(n).toarray() #Define la matriz indentidad
    B = I + (D*dt/dx**2)*A #Define la matriz del problema
    #Modifico la matriz para aplicar las condiciones de contorno
    B[0,:] = 0
    B[0, 0] = 1
    B[n-1,:] = 0
    B[n-1, n-1] = 1
    
    
    #Define el vector de solución inicial
    T_0 = np.zeros(n)
    for i in range(1, n-1):
        T_0[i] = 100
    
    T = [] #Define una lista para los valores de la solución
    T.append(T_0) #Añade el valor inicial
    for i in range(m):
        T_temp = np.dot(B, T_0) #Calcula la solución en el instante i
        T.append(T_temp) #Añade a la lista la solución en el instante i
        T_0 = T_temp #Cambia la solución inicial a la que se acaba de calcular
    return T


def difusion_cn(caso):
    if caso == 1:
        r = (D*dt)/(2*dx**2) #Define el coeficiente r
        #Define la matriz de la izquierda de la ec.
        diag1 = ((1+2*r)/r)*np.ones(n) #Define la diagonal de la matriz tridiagonal
        off1 = -1*np.ones(n-1) #Define las diagonales superior e inferior a la diagonal principal, en esto caso son iguales
        A1 = scipy.sparse.diags([diag1, off1, off1], [0,-1,1],shape=(n,n)).toarray() #Define la matriz tridiagonal
        #Modifico la matriz para aplicar las condiciones de contorno
        A1[0,:] = 0
        A1[0, 0] = 1
        A1[n-1,:] = 0
        A1[n-1, n-1] = 1
        #Define la matriz de la derecha de la ec.
        diag2 = ((1-2*r)/r)*np.ones(n) #Define la diagonal de la matriz tridiagonal
        off2 = np.ones(n-1) #Define las diagonales superior e inferior a la diagonal principal, en esto caso son iguales
        A2 = scipy.sparse.diags([diag2, off2, off2], [0,-1,1],shape=(n,n)).toarray() #Define la matriz tridiagonal
        #Modifico la matriz para aplicar las condiciones de contorno
        A2[0,:] = 0
        A2[0, 0] = 1
        A2[n-1,:] = 0
        A2[n-1, n-1] = 1
        B = np.dot(np.linalg.inv(A1), A2) #Define la matriz final del sistema
        

        
    if caso == 2:
        r1 = (D*dt)/(2*dx**2) #Define el coeficiente r del caso 1
        r2 = (D2*dt)/(2*dx**2) #Define el coeficiente r del caso 2
        #Define la matriz de la izquierda de la ec.
        diag1_1 = ((1+2*r1)/r1)*np.ones(int(n*0.4)) #Define la diagonal de la matriz tridiagonal
        diag1_2 = ((1+2*r2)/r2)*np.ones(int(n*0.2))
        diag1_3 = np.concatenate((np.concatenate((diag1_1, diag1_2)), diag1_1))
        off1 = -1*np.ones(n-1) #Define las diagonales superior e inferior a la diagonal principal, en esto caso son iguales
        A1 = scipy.sparse.diags([diag1_3, off1, off1], [0,-1,1],shape=(n,n)).toarray() #Define la matriz tridiagonal
        #Modifico la matriz para aplicar las condiciones de contorno
        A1[0,:] = 0
        A1[0, 0] = 1
        A1[n-1,:] = 0
        A1[n-1, n-1] = 1
        #Define la matriz de la derecha de la ec.
        diag2_1 = ((1-2*r1)/r1)*np.ones(int(n*0.4)) #Define la diagonal de la matriz tridiagonal
        diag2_2 = ((1-2*r2)/r2)*np.ones(int(n*0.2))
        diag2_3 = np.concatenate((np.concatenate((diag2_1, diag2_2)), diag2_1))
        off2 = np.ones(n-1) #Define las diagonales superior e inferior a la diagonal principal, en esto caso son iguales
        A2 = scipy.sparse.diags([diag2_3, off2, off2], [0,-1,1],shape=(n,n)).toarray() #Define la matriz tridiagonal
        #Modifico la matriz para aplicar las condiciones de contorno
        A2[0,:] = 0
        A2[0, 0] = 1
        A2[n-1,:] = 0
        A2[n-1, n-1] = 1
        B = np.dot(np.linalg.inv(A1), A2) #Define la matriz final del sistema
        
         
    #Define el vector de solución inicial
    T_0 = np.zeros(n)
    T_0[0] = 50
    for i in range(1, n-1):
        T_0[i] = 100
            
    T = [] #Define una lista para los valores de la solución
    T.append(T_0) #Añade el valor inicial
    for i in range(m):
        T_temp = np.dot(B, T_0) #Calcula la solución en el instante i
        T.append(T_temp) #Añade a la lista la solución en el instante i
        T_0 = T_temp #Cambia la solución inicial a la que se acaba de calcular
    return T

        

#Funciones para la animación
def init():
    line.set_data([], [])
    return line,


def animate1(i):
    line.set_data(x, difusion_ftcs()[i])
    return line,


def animate2(i):
    line.set_data(x, difusion_cn(1)[i])
    return line,
    
def animate3(i):
    line.set_data(x, difusion_cn(2)[i])
    return line,



#Datos del problema
D = 10**(-2) #Define el coeficiente de difusión
D2 = 10**(-3) #Define otro coeficiente de difusión
L = 1 #Define el dominio espacial
dx = 0.05 #Define la variación espacial
n = int(L/dx) #Define el número de puntos en los que se discretiza el espacio
t = 20 #Define el dominio temporal
dt = (0.5*dx**2)/(2*D) #Define la variación temporal con el criterio de estabilidad
m = int(t/dt) #Define el número de puntos en los que se discretiza el tiempo
x = np.arange(0, L, dx) #Define el espacio en el que se dibuja   

#Animación del primer apartado
fig1 = plt.figure()
ax = plt.axes(xlim=(-0.1, 1.1), ylim=(-20, 120)) #Define el límite de los ejes para que no se muevan
plt.title('Ecuación de difusión FTCS')
plt.xlabel('Longitud')
plt.ylabel('Temperatura')
line, = ax.plot([], [], lw=2, color = 'blue')
anim = animation.FuncAnimation(fig1, animate1, init_func=init, frames=m, interval=10, blit=True)
plt.show()

#Animación del segundo apartado
fig2 = plt.figure()
ax = plt.axes(xlim=(-0.1, 1.1), ylim=(-20, 120)) #Define el límite de los ejes para que no se muevan
plt.title('Ecuación de difusión Crank-Nicholson')
plt.xlabel('Longitud')
plt.ylabel('Temperatura')
line, = ax.plot([], [], lw=2, color = 'red')
anim = animation.FuncAnimation(fig2, animate2, init_func=init, frames=m, interval=10, blit=True)
plt.show()

#Animación del tercer apartado
fig3 = plt.figure()
ax = plt.axes(xlim=(-0.1, 1.1), ylim=(-20, 120)) #Define el límite de los ejes para que no se muevan
plt.title('Ecuación de difusión con distitnos coeficientes de difusión')
plt.xlabel('Longitud')
plt.ylabel('Temperatura')
line, = ax.plot([], [], lw=2, color = 'orange')
anim = animation.FuncAnimation(fig3, animate3, init_func=init, frames=m, interval=10, blit=True)
plt.show()

'''
#Dibuja cada apartado en distintos instantes de tiempo para el informe
#Primer apartado
for i in range(0, m, int(m/3)):
    t = round(i*dt, 2)
    plt.figure()
    ax = plt.axes(xlim=(-0.1, 1.1), ylim=(-20, 120))
    plt.title('Ecuación de difusión FTCS')
    plt.xlabel('Longitud')
    plt.ylabel('Temperatura')
    plt.plot(x, difusion_ftcs()[i], lw=2, color = 'blue', label = 'Tiempo  = '+ str(t) + ' s')
    plt.legend(loc = 'best')
    plt.show()

#Segundo apartado
for i in range(0, m, int(m/3)):
    t = round(i*dt, 2)
    plt.figure()
    ax = plt.axes(xlim=(-0.1, 1.1), ylim=(-20, 120))
    plt.title('Ecuación de difusión Crank-Nicholson')
    plt.xlabel('Longitud')
    plt.ylabel('Temperatura')
    plt.plot(x, difusion_cn(1)[i], lw=2, color = 'red', label = 'Tiempo  = '+ str(t) + ' s')
    plt.legend(loc = 'best')
    plt.show()

#Tercer apartado
for i in range(0, m, int(m/3)):
    t = round(i*dt, 2)
    plt.figure()
    ax = plt.axes(xlim=(-0.1, 1.1), ylim=(-20, 120))
    plt.title('Ecuación de difusión con distintos coeficientes de difusión')
    plt.xlabel('Longitud')
    plt.ylabel('Temperatura')
    plt.plot(x, difusion_cn(2)[i], lw=2, color = 'orange', label = 'Tiempo  = '+ str(t) + ' s')
    plt.legend(loc = 'best')
    plt.show()
'''


    