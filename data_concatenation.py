# -*- coding: utf-8 -*-
"""
Elaborated for the Neuroscience and Experimental Psychology lab, USAL Argentina

Language: Spanish
"""

import os
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

CUBI = pd.read_excel("datos_CUBI.xlsx")
DERS = pd.read_excel("datos_DERS.xlsx")
altruism = pd.read_excel("datos_altruism.xlsx")
social_discounting = pd.read_excel("df_discounting.xlsx")
consistencia_total = pd.read_excel("mapa_consistencias.xlsx").iloc[:,7]
consistencia_inter_D = pd.read_excel("consistencia_inter_D.xlsx")

df_completo = pd.concat([CUBI,DERS,altruism,social_discounting,consistencia_total,consistencia_inter_D], axis=1).drop('Unnamed: 0', axis=1)

df_completo.to_excel("df_completo.xlsx")