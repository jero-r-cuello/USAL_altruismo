# -*- coding: utf-8 -*-
"""
Created on Fri May 31 09:53:14 2024

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.stats import sem

# Graficos sacando inconsistentes, partiendo por la mediana del Altruism

base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)
social_discounting = pd.read_excel("df_discounting.xlsx")
altruismo = pd.read_excel("datos_altruism.xlsx")
consistencia = pd.read_excel("mapa_consistencias.xlsx").drop([1,5,10,20,50,100], axis=1)
consistencia_inter_D = pd.read_excel("consistencia_inter_D.xlsx")
social_distances = [1,5,10,20,50,100]

df = pd.concat([social_discounting,altruismo,consistencia,consistencia_inter_D], axis=1).drop("Unnamed: 0", axis=1)
#%% Filtrado y variables importantes

# Filtramos a los inconsistentes
consistentes_primer_paso = df[df["Consistencia"] == True]
consistentes = consistentes_primer_paso[consistentes_primer_paso[0] == True]


rtas_total_mean = consistentes.iloc[:,0:6].describe().loc["mean"]
rtas_total_median = consistentes.iloc[:,0:6].describe().loc["50%"]
rtas_total_q4 = consistentes.iloc[:,0:6].describe().loc["25%"]
rtas_total_q1 = consistentes.iloc[:,0:6].describe().loc["75%"]
st_error = sem(consistentes.iloc[:,0:6])

# Obtenemos la mediana de escala de altruismo
altruism_median = consistentes["total_altruismo"].describe()["50%"]



# Separamos en dos grupos
no_altruistas = consistentes[consistentes["total_altruismo"] <= altruism_median]
altruistas = consistentes[consistentes["total_altruismo"] > altruism_median]

# Guardamos las respuestas medias de altruistas y no altruistas
rtas_no_altruistas = no_altruistas.iloc[:,0:6].describe().loc["mean"]
rtas_altruistas = altruistas.iloc[:,0:6].describe().loc["mean"]
#%% Respuestas medias y sus curvas ajustadas según grupo de altruismo

# Las funciones están más abajo, tengo problemas para importarlas desde el otro archivo.
# RESOLVER ESO

datos_y = [rtas_no_altruistas,rtas_altruistas]
lista_funciones, lista_s, lista_r = fiting_hyperbolic(datos_y, social_distances)

D_fit = np.linspace(min(np.array(social_distances)), max(np.array(social_distances)), 100)
# Plot the data and the fitted curve
# Faltaría poner los errores
plt.figure(num=2)
plt.plot(social_distances, rtas_no_altruistas, 'go', label='No altruistas')
plt.plot(social_distances, rtas_altruistas, 'bo', label='Altruistas')
plt.plot(D_fit, lista_funciones[0], 'g-')
plt.plot(D_fit, lista_funciones[1], 'b-')
plt.xlabel('D')
plt.ylabel('V')
plt.title('Ajuste a las medias según altruismo')
plt.legend()

#%% Hacer el gráfico de mediana, q1 y q4 de todos los sujetos consistentes

datos_y = [rtas_total_q1,rtas_total_q4,rtas_total_median]
lista_funciones, lista_s, lista_r = fiting_hyperbolic(datos_y, social_distances)
D_fit = np.linspace(min(np.array(social_distances)), max(np.array(social_distances)), 100)

plt.figure(num=3)
plt.plot(social_distances, rtas_total_q1, 'go', label='Quartil 1')
plt.plot(social_distances, rtas_total_q4, 'bo', label='Quartil 4')
plt.plot(social_distances, rtas_total_median, 'ro', label='Median')
plt.plot(D_fit, lista_funciones[0], 'g-')
plt.plot(D_fit, lista_funciones[1], 'b-')
plt.plot(D_fit, lista_funciones[2], 'r-')
plt.xlabel('D')
plt.ylabel('V')
plt.title('Ajuste a la mediana, Q1 y Q4')
plt.legend()

#%% Hacer el gráfico de la media y error bars de todos los consistentes

lista_funciones, lista_s, lista_r = fiting_hyperbolic([rtas_total_mean], social_distances)
D_fit = np.linspace(min(np.array(social_distances)), max(np.array(social_distances)), 100)

plt.figure(num=4)
plt.errorbar(social_distances, rtas_total_mean, st_error, fmt="go", ecolor="r")
plt.plot(D_fit, lista_funciones[0], 'g-')
plt.xlabel('D')
plt.ylabel('V')
plt.title('Ajuste a la media')
plt.show()
#%%

# Define the hyperbolic function
def hyperbolic(D, A, s):
    """
    Formula hiperbólica de descuento social (Jones and Rachlin, 2006)

    """
    return A / (1 + s * D)

def fiting_hyperbolic(datos_y, social_distances):
    """
    Ajuste de la formula hiperbólica de descuento social a nuestros datos.

    Parameters
    ----------
    datos_y :
        Los valores obtenidos a partir de los turning points.
    social_distances : 
        Lista con valores de las distancias sociales.

    Returns
    -------
    lista_funciones :
        Los valores que toma la curva fiteada.
    lista_s : TYPE
        El valor de s, sensibilidad a la distancia social.
    lista_r : TYPE
        R cuadrado, índice de ajuste.

    """
    lista_funciones = []
    lista_s = []
    lista_r = []
    
    for values in datos_y:
        V = pd.Series(values)
        A = 75
        D = np.array(social_distances)
        
        popt, pcov = curve_fit(hyperbolic, D, V, p0=(A, 1))
        s_optimized = popt[1]
        
        
        print("Optimized value of s:", s_optimized)
        
        y_fit = hyperbolic(D_fit, *popt)
        r_sq = r_squared(V,y_fit)
        
        lista_funciones.append(y_fit)
        lista_s.append(s_optimized)
        lista_r.append(r_sq)
    
    return lista_funciones, lista_s, lista_r

def r_squared(y_data,y_fit):
    """
    Devuelve un índice de que tan bueno es el ajuste de nuestra curva a nuestros datos

    Parameters
    ----------
    y_data :
        Nuestros datos.
    y_fit : TYPE
        Datos de la curva ajustada.

    Returns
    -------
    R_squared :
        Índice del ajuste.

    """
    # Calculate residuals
    residuals = y_data - [y_fit[int(i)-1] for i in social_distances]
    
    # Calculate Total Sum of Squares (TSS)
    TSS = np.sum((y_data - np.mean(y_data))**2)
    
    # Calculate Residual Sum of Squares (ESS)
    ESS = np.sum(residuals**2)
    
    # Calculate R-squared
    R_squared = 1 - (ESS / TSS)
    print("R-squared:", R_squared)
    
    return R_squared

#%%

def inter_D_per_subject(rtas_sujeto):
    rta_anterior = 1000
    consistencia = True
    
    for rta in rtas_sujeto:
        if rta <= rta_anterior:
            rta_anterior = rta
        else:
            consistencia = False
            break
    
    return consistencia

def inter_D_consistency(df_completo):
    consistencia_inter_D = []
    
    for n in range(len(df_completo)):
        consistencia_inter_D.append(inter_D_per_subject(df.iloc[n,0:6]))
    
    return consistencia_inter_D
#%%

