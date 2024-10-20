# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 15:14:19 2023

@author: Manuel Mora Cartagena
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

#Resolución de la ecuación diferencial para el oscilador armónico amortiguado

def oscilador(w, a, x0, v0, t): #Define la solución exacta del oscialdor amortiguado
    l = a/2
    if l > w: #Movimiento sobreamortiguado
        r1 = -l + np.sqrt(l**2 - w**2)
        r2 = -l - np.sqrt(l**2 - w**2)
        c2 = (v0 - x0*r1)/(r2 + r1)
        c1 = x0 - c2
        return c1*np.exp(r1*t) + c2*np.exp(r2*t)
    if l == w: #Movimiento críticamente amortiguado
        r = -l
        c1 = x0
        c2 = v0 - r*x0
        return (c1 + c2*t)*np.exp(r*t)
    if l < w: #Movimiento subamortiguado
        c1 = x0
        c2 = (v0 + l*x0)/(np.sqrt(w**2 - l**2))
        return np.exp(-l*t)*((c1*np.cos(np.sqrt(w**2 - l**2)*t) + c2*np.sin(np.sqrt(w**2 - l**2)*t)))
    

def forward(h):
    n = int(t/h) #Define el número de puntos
    B = np.array(([0, 1], [-w**2, -a])) #Define la matriz B 
    I = scipy.sparse.eye(2).toarray() #Define la matriz indentidad 2x2
    A_forward = I + h*B #Define la matriz A para Euler forward
    solucion_inicial_forward = np.array([x0, v0]) #Define el vector columna con las condiciones iniciales para Euler foward
    x_forward = [] #Define el vector columna de las posiciones para Euler forward
    v_forward = [] #Define el vector columna de las velocidades para Euler forward
    for i in range(n):
        solucion_forward = np.dot(A_forward, solucion_inicial_forward) #Calcula la solución con Euler forward en el insatnte h*i
        x_forward.append(solucion_forward[0]) #Añade la posicion calculada en el instante h*i
        v_forward.append(solucion_forward[1]) #Añade la velocidad calculada en el instante h*i
        solucion_inicial_forward = solucion_forward
    return(x_forward, v_forward)


def backward(h):
    n = int(t/h) #Define el número de puntos
    B = np.array(([0, 1], [-w**2, -a])) #Define la matriz B 
    I = scipy.sparse.eye(2).toarray() #Define la matriz indentidad 2x2
    A_backward = np.linalg.inv(I - h*B) #Define la matriz A para Euler backward 
    solucion_inicial_backward = np.array([x0, v0]) #Define el vector columna con las condiciones iniciales para Euler backward
    x_backward = [] #Define el vector columna de las posiciones para Euler backward
    v_backward = [] #Define el vector columna de las velocidades para Euler backward
    for i in range(n):
        solucion_backward = np.dot(A_backward, solucion_inicial_backward) #Calcula la solución con Euler forward en el insatnte h*i
        x_backward.append(solucion_backward[0]) #Añade la posicion calculada en el instante h*i
        v_backward.append(solucion_backward[1]) #Añade la velocidad calculada en el instante h*i
        solucion_inicial_backward = solucion_backward
    return(x_backward, v_backward)


def cn(h):
    n = int(t/h) #Define el número de puntos
    B = np.array(([0, 1], [-w**2, -a])) #Define la matriz B 
    I = scipy.sparse.eye(2).toarray() #Define la matriz indentidad 2x2
    A_cn = np.dot(np.linalg.inv(I - 0.5*h*B), I + 0.5*h*B) #Define la matriz A para Crank Nicholson
    solucion_inicial_cn = np.array([x0, v0]) #Define el vector columna con las condiciones iniciales para Crank Nicholson
    x_cn = [] #Define el vector columna de las posiciones para Crank Nicholson
    v_cn = [] #Define el vector columna de las velocidades para Crank Nicholson
    for i in range(n):
        solucion_cn = np.dot(A_cn, solucion_inicial_cn) #Calcula la solución con Crank Nicholson en el insatnte h*i
        x_cn.append(solucion_cn[0]) #Añade la posicion calculada en el instante h*i
        v_cn.append(solucion_cn[1]) #Añade la velocidad calculada en el instante h*i
        solucion_inicial_cn = solucion_cn
    return(x_cn, v_cn)


def desviacion_t(aproximacion, real, n): #Define la función de la desviación cuadrática media en función del tiempo
    suma = 0
    desviacion = []
    for i in range(n):
        suma += (aproximacion[i] - real[i])**2
        desviacion.append(np.sqrt(suma/i))
    return desviacion


def desviacion_vt(k): #Define la función de la desviación cuadrática media en función de la variación temporal
    error_forward = []
    error_backward = []
    error_cn = []
    for i in k:
        n = int(t/i)
        tiempo = np.linspace(0, t-i, n)
        x_real = oscilador(w, a, x0, v0, tiempo)
        x_forward = forward(i)[0]
        x_backward = backward(i)[0]
        x_cn = cn(i)[0]
        error_temp_forward = 0
        error_temp_backward = 0
        error_temp_cn = 0
        for j in range(n):
            error_temp_forward += (x_forward[j] - x_real[j])**2
            error_temp_backward += (x_backward[j] - x_real[j])**2
            error_temp_cn += (x_cn[j] - x_real[j])**2
        error_forward.append(np.sqrt(error_temp_forward/n))
        error_backward.append(np.sqrt(error_temp_backward/n))
        error_cn.append(np.sqrt(error_temp_cn/n))
    return(error_forward, error_backward, error_cn)




#Parámetros del problema
m = 1 #Define la masa unida al muelle
a = 0.5 #Define el coeficiente de amortiguación 
w = 2 #Define la frecuencia natural del muelle (w^2 = k/m)
h = 0.01 #Define la variación temporal
t = 10 #Define el tiempo de simulación
x0 = 1 #Define la posición inicial
v0 = 0 #Define la velocidad inicial
n = int(t/h) #Define el número de puntos
tiempo = np.linspace(0, t-h, n) #Define el intervalo temporal en el que se va a dibujar
k = np.linspace(0.001, 0.1, 200) #Define el intervalo de variaciones temporales en el que se va a dibujar


#Dibujamos para cada método
#Euler forward
plt.figure()
plt.plot(tiempo, oscilador(w, a, x0, v0, tiempo), 'g', label = 'Solución exacta')
plt.plot(tiempo, forward(h)[0], 'b', label = 'Euler forward')
plt.xlabel('Tiempo (s)')
plt.ylabel('Posición')
plt.title('Oscilador armónico amortiguado resuelto con Euler forward')
plt.legend(loc = 'best')
 
#Euler backward
plt.figure()
plt.plot(tiempo, oscilador(w, a, x0, v0, tiempo), 'g', label = 'Solución exacta')
plt.plot(tiempo, backward(h)[0], 'r', label = 'Euler backward')
plt.xlabel('Tiempo (s)')
plt.ylabel('Posición')
plt.title('Oscilador armónico amortiguado resuelto con Euler backrward')
plt.legend(loc = 'best')

#Crank Nicholson
plt.figure()
plt.plot(tiempo, oscilador(w, a, x0, v0, tiempo), 'g', label = 'Solución exacta')
plt.plot(tiempo, cn(h)[0], 'y', label = 'Crank Nicholson')
plt.xlabel('Tiempo (s)')
plt.ylabel('Posición')
plt.title('Oscilador armónico amortiguado resuelto con Crank-Nicholson')
plt.legend(loc = 'best')

#Dibujamos las posiciones para todos los métodos
plt.figure()
plt.plot(tiempo, oscilador(w, a, x0, v0, tiempo), 'g', label = 'Solución exacta')
plt.plot(tiempo, forward(h)[0], 'b', label = 'Euler forward')
plt.plot(tiempo, backward(h)[0], 'r', label = 'Euler backward')
plt.plot(tiempo, cn(h)[0], 'y', label = 'Crank Nicholson')
plt.xlabel('Tiempo (s)')
plt.ylabel('Posición')
plt.title('Oscilador armónico amortiguado')
plt.legend(loc = 'best')

#Dibujamos las desviaciones en función del tiempo para todos los métodos
plt.figure()
plt.plot(tiempo, desviacion_t(forward(h)[0], oscilador(w, a, x0, v0, tiempo), n), 'b', label = 'Desviación Euler forward')
plt.plot(tiempo, desviacion_t(backward(h)[0], oscilador(w, a, x0, v0, tiempo), n), 'r', label = 'Desviación Euler backward')
plt.plot(tiempo, desviacion_t(cn(h)[0], oscilador(w, a, x0, v0, tiempo), n), 'y', label = 'Desviación Crank Nicholson')
plt.xlabel('Tiempo (s)')
plt.ylabel('Desviación cuadrática media')
plt.title('Desviación cuadrática media en función del tiempo')
plt.legend(loc = 'best')

#Dibujamos las desviaciones en función de la variación temporal para todos los métodos
plt.figure()
plt.plot(k, desviacion_vt(k)[0], 'b', label = 'Desviación Euler forward')
plt.plot(k, desviacion_vt(k)[1], 'r', label = 'Desviación Euler backward')
plt.plot(k, desviacion_vt(k)[2], 'y', label = 'Desviación Crank Nicholson')
plt.xlabel('Variación temporal $\Delta t$ (s)')
plt.ylabel('Desviación cuadrática media')
plt.title('Desviación cuadrática media en función de la variación temporal')
plt.legend(loc = 'best')

plt.show()