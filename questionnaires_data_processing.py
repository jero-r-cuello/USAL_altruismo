# -*- coding: utf-8 -*-
"""
Elaborated for the Neuroscience and Experimental Psychology lab, USAL Argentina

Language: Spanish
"""

import pandas as pd
import numpy
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
archivo = os.path.join(base_dir, "Formulario Imp+DE+DS (A).xlsx")
 
#%% Separamos y renombramos los ítems según el cuestionario

df_raw = pd.read_excel(archivo, header=None, dtype=str,skiprows=5)
DERS = df_raw.iloc[:, 15:43].rename(columns={15:1,16:2,17:3,18:4,19:5,20:6,21:7,
                                             22:8,23:9,24:10,25:11,26:12,27:13,
                                             28:14,29:15,30:16,31:17,32:18,
                                             33:19,34:20,35:21,36:22,37:23,
                                             38:24,39:25,40:26,41:27,42:28})

CUBI = df_raw.iloc[:, 43:61].rename(columns={43:1,44:2,45:3,46:4,47:5,48:6,
                                             49:7,50:8,51:9,52:10,53:11,54:12,
                                             55:13,56:14,57:15,58:16,59:17,
                                             60:18})
Altruism = df_raw.iloc[:, 61:81]

#%%
def evaluacion_DERS(DERS_data):
    """
    Evalua las respuestas de un conjunto de sujetos a la prueba DERS

    Parameters
    ----------
    DERS_data : DataFrame
        Un dataframe con las respuestas del 1 al 5 a los 28 ítems del DERS.

    Returns
    -------
    rtas_sujetos : list
        Lista de listas. Cada una contiene el puntaje de cada categoría del DERS.

    """
    pregs_invertidas = [1,2,6,7,9]
    valores_invertidos = {1:5,2:4,3:3,4:2,5:1}
    
    desc_emocional = [3,13,14,15,17,22,25,26,28]
    rechazo_emocional = [10,11,18,19,20,23,24]
    interf_cotidiana = [12,16,21,27]
    desat_emocional = [2,6,7,9]
    conf_emocional = [1,4,5,8]
    
    rtas_sujetos = []
    # Iteramos por cada sujeto, reseteando la lista
    for sujeto in range(len(DERS_data)):
        rtas_sujeto = []
        
        # Iteramos por cada ítem y lo agregamos a la lista
        for c in DERS_data.columns:
            rtas_sujeto.append(DERS_data.loc[sujeto,c][0])
            
        # Invertimos los valores de items si es necesario
        for i in range(len(rtas_sujeto)):
            if i+1 in pregs_invertidas:
                    rtas_sujeto[i] = valores_invertidos[int(rtas_sujeto[i])]
                    
        # Obtenemos los valores para guardar en el df general
        desc_emocional_sujeto = sum([int(rtas_sujeto[i-1]) for i in desc_emocional])
        rechazo_emocional_sujeto = sum([int(rtas_sujeto[i-1]) for i in rechazo_emocional])
        interf_cotidiana_sujeto = sum([int(rtas_sujeto[i-1]) for i in interf_cotidiana])
        desat_emocional_sujeto = sum([int(rtas_sujeto[i-1]) for i in desat_emocional])
        conf_emocional_sujeto = sum([int(rtas_sujeto[i-1]) for i in conf_emocional])
        total = sum([int(rta) for rta in rtas_sujeto])
        
        vector_sujeto = [desc_emocional_sujeto,rechazo_emocional_sujeto,interf_cotidiana_sujeto,
                         desat_emocional_sujeto,conf_emocional_sujeto,total]

        rtas_sujetos.append(vector_sujeto)
        
    return rtas_sujetos

def evaluacion_CUBI(CUBI_data):
    """
    Evalua las respuestas de un conjunto de sujetos a la prueba CUBI-18

    Parameters
    ----------
    CUBI_data : DataFrame
        Un dataframe con las respuestas del 1 al 5 a los 18 ítems del CUBI-18.

    Returns
    -------
    rtas_sujetos : list
        Lista de listas. Cada una contiene el puntaje de cada categoría del CUBI.

    """
    urgencia_compulsiva = [1,4,7,10,13,16]
    imp_por_imprevision = [2,5,8,11,14,17]
    busqueda_de_sensaciones = [3,6,9,12,15,18]
    
    valores_CUBI = {"TD":1, "D":2, "N":3, "A":4, "TA":5}
    valores_invertidos = {1:5, 2:4, 3:3, 4:2, 5:1}
    
    rtas_sujetos = []
    # Iteramos por cada sujeto, reseteando la lista
    for sujeto in range(len(CUBI_data)):
        rtas_sujeto = []
        
        # Iteramos por cada ítem y lo agregamos a la lista
        for c in CUBI_data.columns:
            rtas_sujeto.append(valores_CUBI[CUBI_data.loc[sujeto,c][:2]])
        
        # Guardamos las respuestas en cada categoría
        urgencia_compulsiva_sujeto = sum([int(rtas_sujeto[i-1]) for i in urgencia_compulsiva])
        imp_por_imprevision_sujeto = sum([int(valores_invertidos[rtas_sujeto[i-1]]) for i in imp_por_imprevision])
        busqueda_de_sensaciones_sujeto = sum([int(rtas_sujeto[i-1]) for i in busqueda_de_sensaciones])
        total = sum([int(rta_sujeto) for rta_sujeto in rtas_sujeto])
        
        vector_sujeto = [urgencia_compulsiva_sujeto,imp_por_imprevision_sujeto,
                         busqueda_de_sensaciones_sujeto,total]
        
        rtas_sujetos.append(vector_sujeto)
        
    return rtas_sujetos

def evaluacion_altruism(altruism_data):
    """
    Evalua las respuestas de un conjunto de sujetos a la prueba de altruismo (especificar cual se utilizó)

    Parameters
    ----------
    altruism_data : DataFrame
        Un dataframe con las respuestas del 1 al 5 a los 20 ítems del TEST (ESPECIFICAR).

    Returns
    -------
    rtas_sujetos : list
        Lista con el puntaje total del TEST (ESPECIFICAR).

    """
    valores_altruism = {"Nunca":1, "Una vez":2, "Más de una vez":3, 
                        "Frecuentemente":4, "Muy frecuentemente":5}
    
    rtas_sujetos = []
    # Iteramos por cada sujeto, reseteando la lista
    for sujeto in range(len(altruism_data)):
        rtas_sujeto = []
        
        # Iteramos por cada ítem y lo agregamos a la lista
        for c in altruism_data.columns:
            rtas_sujeto.append(valores_altruism[altruism_data.loc[sujeto,c][:20]])
    
        total = sum([int(rta_sujeto) for rta_sujeto in rtas_sujeto])
        
        rtas_sujetos.append(total)
    
    return rtas_sujetos

#%% Generamos el excel de datos DERS

df_DERS = pd.DataFrame(evaluacion_DERS(DERS),columns=["desc_emocional",
                                                      "rechazo_emocional",
                                                      "interf_cotidiana", 
                                                      "desat_emocional",
                                                      "conf_emocional", 
                                                      "total_DERS"])

df_DERS.to_excel(f'{base_dir}/datos_DERS.xlsx')

#%% Generamos el excel de datos CUBI

df_CUBI = pd.DataFrame(evaluacion_CUBI(CUBI),columns=["urgencia_compulsiva",
                                                      "imp_por_imprevision",
                                                      "busqueda_de_sensaciones",
                                                      "total_CUBI"])

df_CUBI.to_excel(f'{base_dir}/datos_CUBI.xlsx')

#%% Generamos el excel de datos Altruismo

df_altruism = pd.DataFrame(evaluacion_altruism(Altruism),columns=["total_altruismo"])

df_altruism.to_excel(f'{base_dir}/datos_altruism.xlsx')

print("Procesado finalizado")