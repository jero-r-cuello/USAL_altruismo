import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.stats import sem
# Análisis del ascendente

base_dir = os.path.dirname(os.path.abspath(__file__))
archivo = os.path.join(base_dir, "Formulario Imp+DE+DS (A).xlsx")
 
#%% Seleccionar los datos y pasarlos a binario (A=0,B=1)(A=egoísta,B=altruista)

df_raw = pd.read_excel(archivo, header=None, dtype=str,skiprows=5)
n_1 = df_raw.iloc[:, 91:101].apply(lambda x: x.str.startswith('B').astype(int))
n_5 = df_raw.iloc[:, 111:121].apply(lambda x: x.str.startswith('B').astype(int))
n_10 = df_raw.iloc[:, 121:131].apply(lambda x: x.str.startswith('B').astype(int))
n_20 = df_raw.iloc[:, 151:161].apply(lambda x: x.str.startswith('B').astype(int))
n_50 = df_raw.iloc[:, 171:181].apply(lambda x: x.str.startswith('B').astype(int))
n_100 = df_raw.iloc[:, 191:201].apply(lambda x: x.str.startswith('B').astype(int))

# Todos los binarios a excel
# headers = ["0","5","15","25","35","45","55","65","75","85"]

# n_1.to_excel(f'{base_dir}/rtas_n_1.xlsx',header=headers)
# n_5.to_excel(f'{base_dir}/rtas_n_5.xlsx',header=headers)
# n_10.to_excel(f'{base_dir}/rtas_n_10.xlsx',header=headers)
# n_20.to_excel(f'{base_dir}/rtas_n_20.xlsx',header=headers)
# n_50.to_excel(f'{base_dir}/rtas_n_50.xlsx',header=headers)
# n_100.to_excel(f'{base_dir}/rtas_n_100.xlsx',header=headers)

rtas = [n_1,n_5,n_10,n_20,n_50,n_100]
#%% Definimos funciones. 

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
        consistencia_inter_D.append(inter_D_per_subject(df_completo.iloc[n,0:6]))
    
    return consistencia_inter_D

# Chequear si funciona efectivamente, o solo aparentemente jaja
def consistency_per_D(rta_distancia_social_n):
    """
    

    Parameters
    ----------
    rta_distancia_social_n :
        Vector con las respuestas de un (1) sujeto a una (1) distancia social.

    Returns
    -------
    consistente :
        Boolean, True si es consistente en la distancia social, False si no lo es.

    """
    lista_rtas = list(rta_distancia_social_n)
    if all(valor == lista_rtas[0] for valor in lista_rtas):
        consistente = True
        return consistente
    
    cambios = 0
    ultimo_cambio = 0
    # Iteramos para ver si cambia más de una vez (ya sabemos que una vez si cambia)
    for i, rta in enumerate(lista_rtas):
        # Si pasa a la opción egoísta
        if rta != lista_rtas[ultimo_cambio]:
            cambios += 1
            ultimo_cambio = i
    
    if cambios > 1:
        consistente = False
        
    else:
        consistente = True
    
    return consistente

# Chequear si funciona efectivamente, o solo aparentemente jaja
def subjects_consistency(vectores):
    """
    

    Parameters
    ----------
    vectores :
        Las respuestas de un (1) sujeto a todas las distancias sociales.

    Returns
    -------
    DataFrame con la consistencia en cada distancia social y la consistencia
    total.

    """
    consistencias_sujetos = pd.DataFrame()
    for n in range(len(vectores[0])):
        rtas_sujeto = [n_1.iloc[n],n_5.iloc[n],n_10.iloc[n],n_20.iloc[n],n_50.iloc[n],n_100.iloc[n]]
        
        consistencia_individual = []
        
        for rta_distancia_social_n in rtas_sujeto:
            consistencia_individual.append(consistency_per_D(rta_distancia_social_n))
        
        consistencia_sujeto = all(consistencia_individual)
        consistencia_individual.append(consistencia_sujeto)
        
        consistencias_sujetos = pd.concat([consistencias_sujetos,pd.DataFrame([consistencia_individual])])
        
    return consistencias_sujetos.reset_index(drop=True)

def find_turning_point(vector, alone = False):
    """
    Encuentra el turning point en la decision del individuo

    Parameters
    ----------
    vector : list / Series
        Un vector de respuestas a una distancia social específica de la tarea
        de descuento social.
        
    alone : bool, optional
        Para su ejecución aislada evaluarlo en True. The default is False.

    Returns
    -------
    El indice en el que hay turning point. Si todos los valores son iguales,
    se devuelve un valor arbitrario (-1 o 0)

    """
    
    # Primero verificamos si hay un turning point, o todas las rtas son iguales
    lista = list(vector)
    if all(valor == lista[0] for valor in lista):
        if alone:
            print("Son todos iguales")
        if lista[0] == 1:
            return -1  # Si el vector es full 1
        else:
            return 0  # Si el vector es full 0
    
    else:
        for i, valor in enumerate(lista):
            # Si pasa a la opción egoísta
            if valor != lista[0]:
                if alone:
                    return i, valor
                else:
                    return i
                
def make_turning_points_df(vectors):
    """
    Genera un DataFrame con los turning points de cada sujeto para cada
    distancia social

    Parameters
    ----------
    vectors : Lista de DataFrames
        Toma los DataFrames de respuestas de cada distancia social.

    Returns
    -------
    final_df : DataFrame

    """
    lista_turning_points = []
    dict_asc = {0: 0, 1: 2.5, 2: 10, 3: 20, 4: 30, 5: 40, 6: 50, 7: 60, 8: 70, 9: 80, -1: 90}
    final_df = pd.DataFrame()
    
    for n in range(len(vectors[0])):
        vectores = [n_1.iloc[n],n_5.iloc[n],n_10.iloc[n],n_20.iloc[n],n_50.iloc[n],n_100.iloc[n]]

        for i in range(len(vectores)):
            a = find_turning_point(vectores[i])
            lista_turning_points.append(a)
    
    last_value_index = 0
    for _ in range(len(n_1)):
        lista_b = []
        rango = lista_turning_points[last_value_index:(last_value_index+6)]
        
        for value in rango:
            lista_b.append(dict_asc[value])
        
        last_value_index = last_value_index+6
        final_df = pd.concat([final_df,pd.DataFrame(lista_b).T], ignore_index=True)

    return final_df

#%% Creamos el df de turning points y lo guardamos como excel

turning_points_df = make_turning_points_df(rtas)
headers_distancias = ["1","5","10","20","50","100"]
turning_points_df.to_excel(f'{base_dir}/turning_points_df.xlsx', header=headers_distancias)
mapa_consistencias = subjects_consistency(rtas)
mapa_consistencias.to_excel(f'{base_dir}/mapa_consistencias.xlsx', header=[1,5,10,20,50,100,"Consistencia"])

#%% Descriptivas
d_stats = turning_points_df.describe(percentiles=([.2,.5,.8]))

median = d_stats.loc["50%"]
mean = d_stats.loc["mean"]
q_1 = d_stats.loc["20%"]
q_4 = d_stats.loc["80%"]
social_distances = [1,5,10,20,50,100]

plt.figure(1)
plt.plot(social_distances, median,'-o', color='black')
plt.plot(social_distances, q_1, '-o', color='blue')
plt.plot(social_distances, q_4,'-o', color='red')
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


#%% Genera un plot (replica de un paper) de la mediana, q1 y q2 y sus curvas fiteadas

datos_y = [median,q_1,q_4]
lista_funciones, lista_s, lista_r = fiting_hyperbolic(datos_y, social_distances)
D_fit = np.linspace(min(np.array(social_distances)), max(np.array(social_distances)), 100)

# Plot the data and the fitted curve
plt.figure(num=2)
plt.plot(social_distances, median, 'go', label='Median')
plt.plot(social_distances, q_1, 'bo', label='Quartil 1')
plt.plot(social_distances, q_4, 'ro', label='Quartil 4')
plt.plot(D_fit, lista_funciones[0], 'g-')
plt.plot(D_fit, lista_funciones[1], 'b-')
plt.plot(D_fit, lista_funciones[2], 'r-')
plt.xlabel('D')
plt.ylabel('V')
plt.title('Ajuste a la mediana, Q1 y Q4')
plt.legend()
#%% Genera un df con los valores de s para cada sujeto

datos_y = [list(row)[1] for row in turning_points_df.iterrows()]
lista_funciones, lista_s, lista_r = fiting_hyperbolic(datos_y, social_distances)

df_discounting = turning_points_df.copy()
df_discounting["Valor de s"] = lista_s
df_discounting["R_squared"] = lista_r

df_discounting.to_excel(f'{base_dir}/df_discounting.xlsx', header=["1","5","10","20","50","100","Valor de s","R_squared"])

#%% Gráfico de la media, hiperbole ajustada y error bars. Todos juntos

std = d_stats.loc["std"]
st_error = sem(turning_points_df)
lista_funciones, lista_s = fiting_hyperbolic([mean], social_distances)
D_fit = np.linspace(min(np.array(social_distances)), max(np.array(social_distances)), 100)

plt.figure(num=3)
plt.errorbar(social_distances, mean, std, fmt="go", ecolor="r")
plt.plot(D_fit, lista_funciones[0], 'g-')
plt.xlabel('D')
plt.ylabel('V')
plt.title('Ajuste a la media')
plt.show()

#%% Gráfico de la media, hiperbole ajustada y error bars. Separado por mediana (High altruism/low altruism)
# Primero hay que limpiar los datos. Sacar a los inconsistentes, ver los que tienen k raros, etc
# Lo mismo con todos los gráficos :) 

df_discounting["Valor de s"]

std = d_stats.loc["std"]
st_error = sem(turning_points_df)
lista_funciones, lista_s = fiting_hyperbolic([mean], social_distances)

plt.figure(num=3)
plt.errorbar(social_distances, mean, st_error, fmt="go", ecolor="r")
plt.plot(D_fit, lista_funciones[0], 'g-')
plt.show()

#%%

lista_funciones, lista_s = fiting_hyperbolic([turning_points_df.loc[44]], social_distances)
D_fit = np.linspace(min(np.array(social_distances)), max(np.array(social_distances)), 100)

plt.figure(num=4)
plt.plot(social_distances, turning_points_df.loc[44])
plt.plot(D_fit, lista_funciones[0], 'g-')
plt.show()

print(r_squared(turning_points_df.loc[44],lista_funciones))