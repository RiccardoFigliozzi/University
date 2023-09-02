#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[170]:


data = pd.read_excel("OpenFood_Petales.xlsx")


# In[171]:


data.tail()


# In[175]:


H = data[data["productname"]=="Noir pétales de rose"]
type(H.iloc[0][5])


# In[ ]:


plt.hist(data['energy100g'], bins = 5)


# In[ ]:


#new_data= data[['nutriscoregrade', 'energy100g', 'saturatedfat100g', 'sugars100g','fiber100g','proteins100g','sodium100g']]


# In[174]:


new_data= data[["productname",'energy100g', 'saturatedfat100g', 'sugars100g','fiber100g','proteins100g','sodium100g']]


# In[13]:


new_data.head()


# In[ ]:


new_data.isnull().sum()


# **Normalization**

# In[ ]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[ ]:


scaler = StandardScaler()
scaled_data = scaler.fit_transform(new_data.values)


# In[ ]:


scaled_data[0][0]


# In[ ]:


scaled_data[0]


# In[ ]:


norm_data= data[['energy100g', 'saturatedfat100g', 'sugars100g','fiber100g','proteins100g','sodium100g']]


# In[ ]:


for i in range(len(norm_data)) :
  for j in range(len(new_data.columns)):
    norm_data.iloc[i,j] = scaled_data[i][j]
  


# In[ ]:


norm_data


# In[ ]:


f = plt.subplots(figsize=(3, 3), dpi=200)
hm = sns.heatmap(norm_data[['energy100g', 'saturatedfat100g', 'sugars100g','fiber100g','proteins100g','sodium100g']].corr(), annot=True, linewidths=0.5, fmt='.2f')


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


attributes = [col for col in new_data.columns if col != 'nutriscoregrade']
X = norm_data[attributes].values
y = data['nutriscoregrade']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=100, 
                                                    stratify=y)


# In[ ]:


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# In[ ]:


param_list = {'max_depth': [None] + list(np.arange(2, 20)),
              'min_samples_split': [2, 5, 10, 20, 30, 50, 100],
              'min_samples_leaf': [1, 5, 10, 20, 30, 50, 100],
             }

clf = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1)

random_search = RandomizedSearchCV(clf, param_distributions=param_list, n_iter=100)
random_search.fit(X, y)
report(random_search.cv_results_, n_top=3)


# In[ ]:


clf = DecisionTreeClassifier(criterion='gini', max_depth=19, min_samples_split=5, min_samples_leaf=1)
clf = clf.fit(X_train, y_train)


# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score


# In[ ]:


print('Test Accuracy %s' % accuracy_score(y_test, y_pred))
print('Test F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))
confusion_matrix(y_test, y_pred)


# In[ ]:


feature_imp = pd.Series(clf.feature_importances_,index=new_data.columns).sort_values(ascending=False)


# In[ ]:


select_feat = feature_imp[feature_imp>=0.03]
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Creating a bar plot
sns.barplot(x=select_feat, y=select_feat.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# In[ ]:


lista = []
for col, imp in zip(attributes, clf.feature_importances_):
    lista.append((imp,col))
    lista.sort(reverse=True)
lista


# In[ ]:


import pydotplus
from sklearn import tree
from IPython.display import Image


# In[ ]:


dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=attributes,
                                
                                filled=True, rounded=True,  
                                special_characters=True)
                                #,max_depth=3)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())


# In[ ]:


sns.displot(data, x="energy100g", hue="nutriscorescore")


# ### Histogram

# In[58]:


for i in new_data.columns: 
    if i != "productname":
        sns.set_theme(style="darkgrid")
        sns.displot(new_data, x=i, bins=4)


# **Profils & poids**

# In[201]:


poids = pd.read_csv("poids.csv")
profil = pd.read_csv("profil.csv")


# In[202]:


poids


# In[203]:


profil


# **ELECTRE TRI** 

# In[2]:


def concordance_partiel_1(H, b, j, critere):
  H=new_data[new_data['productname']==H]
  b=profil[profil['profil']==b]
  c=0
  if critere == 'max':
    if H.iloc[0][j] >= b.iloc[0][j]:
      c=1
      return c
    else :
      return c
  if critere == 'min':
      if b.iloc[0][j] >= H.iloc[0][j]:
        c =1 
        return c
      else :
        return c
  return 


# In[15]:


concordance_partiel_1('Pétales de sarrasin', 'b6', 3,'max')


# In[16]:


concordance_partiel_1('Pétales de sarrasin', 'b6', 6,'min')


# In[3]:


def concordance_partiel_2(b, H, j, critere):
  H=new_data[new_data['productname']==H]
  b=profil[profil['profil']==b]
  c=0
  if critere == 'min':
    if H.iloc[0][j] >= b.iloc[0][j]:
      c=1
      return c
    else :
      return c
  if critere == 'max':
      if b.iloc[0][j] >= H.iloc[0][j]:
        c =1 
        return c
      else :
        return c
  return 


# In[18]:


concordance_partiel_2('b5', 'Pétales de sarrasin',5 ,"max")


# In[19]:


concordance_partiel_2('b5', 'Pétales de sarrasin',5 ,"min")


# **L’indice de concordance global**

# In[4]:


def concondance_global_1(H_in, b_in, critere):
  H=new_data[new_data['productname']==H_in]
  b=profil[profil['profil']==b_in]
  num=0
  den=0
  for i in range(1,len(H.columns)):
    num += poids.iloc[0][i-1]*concordance_partiel_1(H_in, b_in, i, critere[i-1])
    den += poids.iloc[0][i-1]
  return num/den


# In[21]:


concondance_global_1('Pétales de sarrasin', 'b5',['max','min','max','max','min','min'])


# In[5]:


def concondance_global_2(b_in, H_in, critere):
  H=new_data[new_data['productname']==H_in]
  b=profil[profil['profil']==b_in]
  num=0
  den=0
  for i in range(1,len(H.columns)):
    num += poids.iloc[0][i-1]*concordance_partiel_2(b_in, H_in, i, critere[i-1])
    den += poids.iloc[0][i-1]
  return num/den


# In[23]:


concondance_global_2('b2','Pétales de sarrasin', ['max','min','max','max','min','min'])


# **Détermination de la relation de surclassement S**

# In[67]:


def surclassament_1(H_in, b_in, lamb, critere):
    if concondance_global_1(H_in, b_in, critere) >= lamb:
      print(str(H_in)+" S "+str(b_in)) 
      return 1
    else:
      print(str(H_in)+" ne surclasse pas "+str(b_in))
      return
    return


# In[25]:


surclassament_1('Pétales de sarrasin', 'b5', 0.5,['max','min','max','max','min','min'])


# In[34]:


def surclassament_2(b_in, H_in, lamb, critere):
    if concondance_global_2(b_in, H_in, critere) >= lamb:
      print(str(b_in)+" S "+str(H_in)) 
      return 1
    else:
      print(str(b_in)+" ne surclasse pas "+str(H_in)) 
      return 0
    return


# In[27]:


surclassament_2("b2",'Pétales de sarrasin', 0.5,['max','min','max','max','min','min'])


# In[28]:


surclassament_2("b5",'Pétales de sarrasin', 0.5,['max','min','max','max','min','min'])


# In[204]:


nutri_cat=['a','b','c','d','e']


# **Procédures d’affectation**

# In[71]:


def pessimiste(H_in, lamb, critere):
    for i in range(len(profil)):
      b_in= profil.iloc[i][0]
      if surclassament_1(H_in, b_in, lamb, critere) == 1:
        print("cat:", nutri_cat[i-1])
        break
    return nutri_cat[i-1]


# In[186]:


pessimiste('Pétales de sarrasin', 0.5, ['min','min','min','min','max','max'])


# In[188]:


pessimiste('Pétales de carotte au thym', 0.5, ['min','min','min','min','max','max'])


# In[90]:


def optimiste(H_in, lamb, critere):
    for i in range(len(profil)-1,-0,-1):
        b_in= profil.iloc[i][0]
        #if surclassament_2(b_in, H_in, lamb, critere) == 1 and i==len(profil)-1:
         #   return nutri_cat[len(profil)-2]
        if surclassament_2(b_in, H_in, lamb, critere) == 1:
            print("cat:", nutri_cat[i])
            break
    return nutri_cat[i]


# In[246]:


optimiste('Spécialité de Rose', 0.5, ['min','min','min','min','max','max'])


# In[247]:


optimiste('Pétales de sarrasin', 0.5, ['min','min','min','min','max','max'])


# ### ELECTRI BD Petals

# In[16]:


bd1=new_data.copy()
bd1["pessimiste_05"]=1


# In[19]:


for i in range(len(bd1)):
    bd1["pessimiste_05"][i] = pessimiste(new_data.iloc[i]["productname"], 0.5, ['min','min','min','min','max','max'])


# In[20]:


bd1["pessimiste_06"]=1


# In[21]:


for i in range(len(bd1)):
    bd1["pessimiste_06"][i] = pessimiste(new_data.iloc[i]["productname"], 0.6, ['min','min','min','min','max','max'])


# In[22]:


bd1["pessimiste_07"]=1


# In[23]:


for i in range(len(bd1)):
    bd1["pessimiste_07"][i] = pessimiste(new_data.iloc[i]["productname"], 0.7, ['min','min','min','min','max','max'])


# In[121]:


bd1.tail()


# In[102]:


bd1["optimiste_05"]=1


# In[103]:


for i in range(0,212):
    #print(i)
    bd1["optimiste_05"][i] = optimiste(new_data.iloc[i]["productname"], 0.5, ['min','min','min','min','max','max'])


# In[ ]:





# In[104]:


for i in range(214,len(bd1)):
    #print(i)
    bd1["optimiste_05"][i] = optimiste(new_data.iloc[i]["productname"], 0.5, ['min','min','min','min','max','max'])


# In[117]:


bd1.iloc[212]["optimiste_05"]


# In[132]:


bd1.iloc[212]["productname"]


# In[116]:


bd1["optimiste_05"][213] = optimiste('Petales de rose', 0.5, ['min','min','min','min','max','max'])


# In[ ]:


for i in range(len(bd1)):
    bd1["new_optimiste_05"][i] = optimiste(new_data.iloc[i]["productname"], 0.5, ['min','min','min','min','max','max'])


# In[65]:


bd1["optimiste_06"]=1


# In[66]:


for i in range(len(bd1)):
    bd1["optimiste_06"][i] = optimiste(new_data.iloc[i]["productname"], 0.6, ['min','min','min','min','max','max'])


# In[67]:


bd1["optimiste_07"]=1


# In[68]:


for i in range(len(bd1)):
    bd1["optimiste_07"][i] = optimiste(new_data.iloc[i]["productname"], 0.7, ['min','min','min','min','max','max'])


# In[122]:


bd1


# In[123]:


bd_prov=bd1.copy()


# In[127]:


bd_prov["nutriscoregrade"]=data["nutriscoregrade"]


# In[128]:


bd_prov


# In[131]:


from openpyxl.workbook import Workbook
bd_prov.to_excel("bd1.xlsx",
             sheet_name='lambda') 

