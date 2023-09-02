# -*- coding: utf-8 -*-
"""Correlation-transp.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14OnLjtYphp4ddWuAuaH8OCHxm3cSlUAw
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

bd1 = pd.read_excel("bd1.xlsx")
bd2 = pd.read_excel("bd2.xlsx")
bd3 = pd.read_excel("bd3_elec.xlsx")

bd2

"""**Label encoder**"""

prova = bd2.copy()

prova = prova[["pessimiste_05",	"pessimiste_06",	"pessimiste_07",	"optimiste_05",	"optimiste_06",	"optimiste_07",	"nutriscoregrade",	"nova_group"]]

lista = ["pessimiste_05",	"pessimiste_06",	"pessimiste_07",	"optimiste_05",	"optimiste_06",	"optimiste_07",	"nutriscoregrade",	"nova_group"]

le = LabelEncoder()
for el in lista :
  prova[el]= le.fit_transform(prova[el])

prova



f = plt.subplots(figsize=(5, 5), dpi=200)
hm = sns.heatmap(prova[["pessimiste_05",	"pessimiste_06",	"pessimiste_07",	"optimiste_05",	"optimiste_06",	"optimiste_07",	"nutriscoregrade",	"nova_group"]].corr(), annot=True, linewidths=0.5, fmt='.2f')



"""**BD3**"""

bd3

corr3 = bd3.copy()

corr3 = corr3[["pessimiste_05",	"pessimiste_06",	"pessimiste_07",	"optimiste_05",	"optimiste_06",	"optimiste_07",	"nutriscoregrade",	"nova_group", "yuka_score"]]



list_2 = ["pessimiste_05",	"pessimiste_06",	"pessimiste_07",	"optimiste_05",	"optimiste_06",	"optimiste_07",	"nutriscoregrade",	"nova_group", "yuka_score"]

le = LabelEncoder()
for el in list_2 :
  corr3[el]= le.fit_transform(corr3[el])

f = plt.subplots(figsize=(5, 5), dpi=200)
hm = sns.heatmap(corr3[["pessimiste_05",	"pessimiste_06",	"pessimiste_07",	"optimiste_05",	"optimiste_06",	"optimiste_07",	"nutriscoregrade",	"nova_group", "yuka_score"]].corr(), annot=True, linewidths=0.5, fmt='.2f')

