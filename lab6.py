# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 13:09:46 2023

@author: Manuel Mora Cartagena
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import animation


def explicito(a):
    #Define las matrices
    r = 1/(1 + (2*k*dt/p)) #Define una constante para simplificar 
    I = scipy.sparse.eye(n+1).toarray() #Define la matriz indentidad
    diag = r*(2 + (2*k*dt/p) - ((2*c**2*dt**2)/(dx**2)))*np.ones(n+1) #Define la diagonal de la matriz tridiagonal
    off = r*((c**2)*(dt**2)/(dx**2))*np.ones(n) #Define las diagonal superior e inferior a la diagonal principal, en este caso son iguales
    A1 = scipy.sparse.diags([diag, off, off], [0,-1,1],shape=(n+1,n+1)).toarray() #Define la matriz tridiagonal que multiplica a u en k
    A0 = I*r #Define la matriz que multiplica a u en k-1
    #Modifico ahora las matrices para aplicar las condiciones de contorno
    #Modifico la matriz A1
    A1[0, :] = 0
    A1[0, 0] = 1
    A1[-1,:] = 0
    A1[-1, -1] = 1
    #Modifico la matriz A2
    A0[0, :] = 0
    A0[0, 0] = 1
    A0[-1,:] = 0
    A0[-1, -1] = 1
    #Define las condiciones iniciales
    u0_0 = np.zeros(n+1)
    u0_1 = np.sin(np.pi*x*a) 
    #Define las condiciones de contorno en el termino independiente 
    u0_0[0] = 0
    u0_0[-1] = 0
    u0_1[0] = 0
    u0_1[-1] = 0   
    u = [] #Define una lista para las soluciones
    u.append(u0_0) #Añade el primer valor inicial
    u.append(u0_1) #Añade el segundo valor incial
    for i in range(m+1): #Define el bucle para calcular la solucion en cada instante de tiempo
        u_temp = np.dot(A1, u0_1) - np.dot(A0, u0_0) #Calcula la solucion en el instante i
        u.append(u_temp) #Añade la solucion calculada a la lista de soluciones
        #Actualiza los valores de las soluciones para la siguiente iteración
        u0_0 = u0_1
        u0_1 = u_temp
    return u


def implicito(a):
    r = ((c**2)*(dt**2))/(dx**2) #Define una constante para simplificar
    q = (k*dt)/p #Define una constante para simplificar
    I = scipy.sparse.eye(n+1).toarray() #Define la matriz indentidad
    diag = (1 + q + 2*r)*np.ones(n+1) #Define la diagonal de la matriz tridiagonal
    off = -r*np.ones(n) #Define las diagonal superior e inferior a la diagonal principal, en este caso son iguales
    A = scipy.sparse.diags([diag, off, off], [0,-1,1],shape=(n+1,n+1)).toarray() #Define la matriz tridiagonal que multiplica a u en k+1
    B = (q-1)*I
    E = 2*I
    #Modifico ahora las matrices para aplicar las condiciones de contorno
    #Modifico la matriz A
    A[0, :] = 0
    A[0, 0] = 1
    A[-1,:] = 0
    A[-1, -1] = 1
    #Modifico la matriz B
    B[0, :] = 0
    B[0, 0] = 1
    B[-1,:] = 0
    B[-1, -1] = 1
    #Modifico la matriz E
    E[0, :] = 0
    E[0, 0] = 1
    E[-1,:] = 0
    E[-1, -1] = 1
    #Calculo las matrices para obtener la solución en cada iteración
    C = np.dot(np.linalg.inv(A), E)
    D = np.dot(np.linalg.inv(A), B)
    #Define las condiciones iniciales
    u0_0 = np.zeros(n+1)
    u0_1 = np.sin(np.pi*x*a) 
    #Define las condiciones de contorno en el termino independiente 
    u0_0[0] = 0
    u0_0[-1] = 0
    u0_1[0] = 0
    u0_1[-1] = 0   
    u = [] #Define una lista para las soluciones
    u.append(u0_0) #Añade el primer valor inicial
    u.append(u0_1) #Añade el segundo valor incial
    for i in range(m+1): #Define el bucle para calcular la solucion en cada instante de tiempo
        u_temp = np.dot(C, u0_1) + np.dot(D, u0_0) #Calcula la solucion en el instante i
        u.append(u_temp) #Añade la solucion calculada a la lista de soluciones
        #Actualiza los valores de las soluciones para la siguiente iteración
        u0_0 = u0_1
        u0_1 = u_temp
    return u


def telegrafo(a):
    I = scipy.sparse.eye(n+1).toarray() #Define la matriz indentidad
    diag = (dt - 2*dt**2)*np.ones(n+1)
    off = np.ones(n)
    A = scipy.sparse.diags([diag, off, off], [0,-1,1],shape=(n+1,n+1)).toarray()
    B = A/(1 + dt)
    C = I/(1 + dt)
    #Modifico ahora las matrices para aplicar las condiciones de contorno
    #Modifico la matriz B
    B[0, :] = 0
    B[0, 0] = 1
    B[-1,:] = 0
    B[-1, -1] = 1
    #Modifico la matriz C
    C[0, :] = 0
    C[0, 0] = 1
    C[-1,:] = 0
    C[-1, -1] = 1
    #Define las condiciones iniciales
    u0_0 = np.zeros(n+1)
    u0_1 = np.sin(np.pi*x*a) 
    #Define las condiciones de contorno en el termino independiente 
    u0_0[0] = 0
    u0_0[-1] = 0
    u0_1[0] = 0
    u0_1[-1] = 0   
    u = [] #Define una lista para las soluciones
    u.append(u0_0) #Añade el primer valor inicial
    u.append(u0_1) #Añade el segundo valor incial
    for i in range(m+1): #Define el bucle para calcular la solucion en cada instante de tiempo
        u_temp = np.dot(B, u0_1) - np.dot(C, u0_0) #Calcula la solucion en el instante i
        u.append(u_temp) #Añade la solucion calculada a la lista de soluciones
        #Actualiza los valores de las soluciones para la siguiente iteración
        u0_0 = u0_1
        u0_1 = u_temp
    return u
    

  

#Funciones para la animación
def init():
    line.set_data([], [])
    return line,


def animate_explicito(i):
    line.set_data(x, explicito(1)[i])
    return line,


def animate_implicito(i):
    line.set_data(x, implicito(1)[i])
    return line,


def animate_telegrafo(i):
    line1.set_data(x, telegrafo(1)[i])
    line2.set_data(x, telegrafo(2)[i])
    line3.set_data(x, telegrafo(3)[i])
    return line1, line2, line3,
    

    
    
#Datos del problema
c = 1 #Define la velocidad de la luz
k = 0.001 #Define la constante k
p = 0.01 #Define la constante p
L = 1 #Define el limite en el dominio espacial
t = 40 #Define el limite en el dominio  temporal
dx = 0.1 #Define la variación espacial
dt = 0.2*dx #Define la variación temporal
n = int(L/dx) #Define el número de puntos en los que se discretiza el espacio
m = int(t/dt) #Define el número de puntos en los que se discretiza el tiempo
x = np.arange(0, L+dx, dx) #Define el intervalo espacial
j = 2 #Número de modos normales como condiciones iniciales 

#Animación del primer apartado 
fig1 = plt.figure()
ax1 = plt.axes(xlim=(-0.1, 1.1), ylim=(-20, 20)) #Define el límite de los ejes para que no se muevan
plt.title('Ecuación de ondas amortiguada método explícito')
plt.xlabel('x')
plt.ylabel('u')
line, = ax1.plot([], [], lw=2, color = 'blue')
anim = animation.FuncAnimation(fig1, animate_explicito, init_func=init, frames=m, interval=10, blit=True)
plt.show()

#Animación del segundo apartado
fig2 = plt.figure()
ax2 = plt.axes(xlim=(-0.1, 1.1), ylim=(-20, 20)) #Define el límite de los ejes para que no se muevan
plt.title('Ecuación de ondas amortiguada método implícito')
plt.xlabel('x')
plt.ylabel('u')
line, = ax2.plot([], [], lw=2, color = 'red')
anim = animation.FuncAnimation(fig2, animate_implicito, init_func=init, frames=m, interval=10, blit=True)
plt.show()

#Animación del tercer apartado
fig3 = plt.figure()
ax3 = plt.axes(xlim=(-0.1, 1.1), ylim=(-7.5, 7.5)) #Define el límite de los ejes para que no se muevan
plt.title('Ecuación del telégrafo')
plt.xlabel('x')
plt.ylabel('u')
line1, = ax3.plot([], [], lw=2, color = 'red', label = 'n = 1')
line2, = ax3.plot([], [], lw=2, color = 'green', label = 'n = 2')
line3, = ax3.plot([], [], lw=2, color = 'blue', label = 'n = 3')
plt.legend(loc = 'best')
anim = animation.FuncAnimation(fig3, animate_telegrafo, init_func=init, frames=int(m/4), interval=10, blit=True)
plt.show()

'''
#Dibujo para cada apartado
#Primer aparatado
for i in range(25, m+1, 250): #Define un bucle para dibujar en cada instante de tiempo
    tiempo = round(i*dt, 2)
    plt.figure()
    ax = plt.axes(xlim=(-0.1, 1.1), ylim=(-20, 20)) #Define el límite de los ejes para que no se muevan
    plt.plot(x, explicito(1)[i], label = 'Tiempo = ' + str(tiempo), color = 'blue')
    plt.title('Ecuación de ondas amortiguada método explícito')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend(loc = 'best')
    plt.show()
#Segundo aparatado
for i in range(25, m+1, 250): #Define un bucle para dibujar en cada instante de tiempo
    tiempo = round(i*dt, 2)
    plt.figure()
    ax = plt.axes(xlim=(-0.1, 1.1), ylim=(-20, 20)) #Define el límite de los ejes para que no se muevan
    plt.plot(x, implicito(1)[i], label = 'Tiempo = ' + str(tiempo), color = 'red')
    plt.title('Ecuación de ondas amortiguada método implícito')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend(loc = 'best')
    plt.show()
#Tercer aparatado
#Dibuja en muchos instantes de tiempo para elegir los frames deseados (son muchas figuras)
for i in range(0, 501, 1): #Define un bucle para dibujar en cada instante de tiempo
    tiempo = round(i*dt, 2)
    plt.figure()
    ax = plt.axes(xlim=(-0.1, 1.1), ylim=(-8, 8)) #Define el límite de los ejes para que no se muevan
    plt.plot(x, telegrafo(1)[i], label = 'n = 1', color = 'red')
    plt.plot(x, telegrafo(2)[i], label = 'n = 2', color = 'green')
    plt.plot(x, telegrafo(3)[i], label = 'n = 3', color = 'blue')
    plt.title('Ecuación del telégrafo (Tiempo = ' + str(tiempo)+' s)')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend(loc = 'best')
    plt.show()
'''

