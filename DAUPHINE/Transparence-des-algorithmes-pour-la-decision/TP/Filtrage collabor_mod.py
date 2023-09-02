#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import math as ma


# In[2]:


data=pd.read_csv("film2.csv")


# In[3]:


data


# In[4]:


data.fillna("No rating", inplace = True)


# In[5]:


data[data['critique']=='Toby']


# ### Manhattan

# In[6]:


def sim_distanceManhattan(pers1, pers2):
    pers1 = data[data['critique']==pers1]
    pers1= pers1.reset_index(drop=True)
    pers2 = data[data['critique']==pers2]
    pers2= pers2.reset_index(drop=True)
    diff=0
    for i in range(1, pers1.shape[1]):
        if (pers1.iloc[0,i]=="No rating" or pers2.iloc[0,i] =="No rating"):
            continue
        else:
            diff+=abs(pers1.iloc[0,i]-pers2.iloc[0,i])
    return diff


# In[7]:


sim_distanceManhattan('Lisa Rose', 'Toby')


# In[8]:


sim_distanceManhattan('Lisa Rose', 'Gene Seymour')


# ### Euclidean

# In[9]:


def sim_distanceEuclidienne(pers1, pers2):
    pers1 = data[data['critique']==pers1]
    pers1= pers1.reset_index(drop=True)
    pers2 = data[data['critique']==pers2]
    pers2= pers2.reset_index(drop=True)
    summ=0
    for i in range(1, pers1.shape[1]):
        if (pers1.iloc[0,i]=="No rating" or pers2.iloc[0,i] =="No rating"):
            continue
        else:
            summ+= ((pers1.iloc[0,i]-pers2.iloc[0,i])**2)
    return ma.sqrt(summ)


# In[10]:


sim_distanceEuclidienne('Lisa Rose', 'Gene Seymour')


# In[11]:


sim_distanceEuclidienne('Lisa Rose', 'Toby')


# ### Nearest Neighbor

# In[12]:


Critiques = [ ]
for i in range(len(data)):
    Critiques.append(data.iloc[i,0])


# In[13]:


Critiques


# In[14]:


def computeNearestNeighbor(nouveauCritique, Critiques): 
    distances=[ ]
    for i in Critiques:
        if nouveauCritique != i:
            distances.append((sim_distanceManhattan(i, nouveauCritique),i))
            distances.sort()
        else :
            continue
    return distances


# In[15]:


computeNearestNeighbor('Lisa Rose', Critiques)


# In[16]:


def computeNearestNeighbor_Euclidean(nouveauCritique, Critiques): 
    distances=[ ]
    for i in Critiques:
        if nouveauCritique != i:
            distances.append((sim_distanceEuclidienne(i, nouveauCritique),i))
            distances.sort()
        else :
            continue
    return distances


# In[17]:


computeNearestNeighbor_Euclidean('Lisa Rose', Critiques)


# ### Raccomandation

# In[18]:


def recommend(nouveauCritique):
    nouveauCritique = data[data['critique']==nouveauCritique ]
    nouveauCritique= nouveauCritique.reset_index(drop=True)
    nn= [ ]
    nn= computeNearestNeighbor(nouveauCritique.iloc[0,0], Critiques)
    best=nn[0][1]
    critique = data[data['critique']==nn[0][1]]
    critique= critique.reset_index(drop=True)
    if all(nouveauCritique.iloc[0] !="No rating" )==True:
        print('Il/elle a vu tous les films')
        return
    else:
        for i in range(1, nouveauCritique.shape[1]):
            if (nouveauCritique.iloc[0,i]=="No rating" and critique.iloc[0,i] !="No rating"):
                print('Film recommandé: ' +str(data.columns[i])+', Vote: '+str(critique.iloc[0,i])) 
        return


# In[19]:


def recommend_mod(nouveauCritique):
    nouveauCritique = data[data['critique']==nouveauCritique ]
    nouveauCritique= nouveauCritique.reset_index(drop=True)
    nn= [ ]
    nn= computeNearestNeighbor(nouveauCritique.iloc[0,0], Critiques)
    recc=[]
    critique = data[data['critique']==nn[0][1]]
    critique= critique.reset_index(drop=True)
    if all(nouveauCritique.iloc[0] !="No rating" )==True:
        print('Il/elle a vu tous les films')
        return
    else:
        for i in range(1, nouveauCritique.shape[1]):
            if (nouveauCritique.iloc[0,i]=="No rating" and critique.iloc[0,i] !="No rating"):
                recc.append((critique.iloc[0,i], data.columns[i]))
                #recc.sort()
                #print('Film recommandé: ' +str(data.columns[i])+', Vote: '+str(critique.iloc[0,i])) 
        return max(recc)


# In[20]:


recommend_mod('Toby')


# In[21]:


recommend('Lisa Rose')


# In[22]:


recommend('Anne')


# In[23]:


recommend('Toby')


# ### Raccomandation b ii

# In[24]:


def recommend_complet(nouveauCritique, film):
    nouveauCritique = data[data['critique']==nouveauCritique ]
    nouveauCritique= nouveauCritique.reset_index(drop=True)
    total=0
    sim=0
    if (nouveauCritique.iloc[0][film]=="No rating" ):
        for j in Critiques:
            critique= data[data['critique']==j]
            critique= critique.reset_index(drop=True)
            if (j!=nouveauCritique.iloc[0,0] and critique.iloc[0][film] !="No rating"):
                    total+=(1/(1+sim_distanceManhattan(nouveauCritique.iloc[0,0], j)))*critique.iloc[0][film]
                    sim+=(1/(1+sim_distanceManhattan(nouveauCritique.iloc[0,0], j)))
        print('Total:', total)
        print('S(a):', sim)
        print('Total/s(a):', total/sim)
    else :
        print('Film déjà voté')
    return  


# In[25]:


recommend_complet('Anne', 'Night')


# In[26]:


def recommend_complet_mod(nouveauCritique, film):
    nouveauCritique = data[data['critique']==nouveauCritique ]
    nouveauCritique= nouveauCritique.reset_index(drop=True)
    total=0
    sim=0
    if (nouveauCritique.iloc[0][film]=="No rating" ):
        for j in Critiques:
            critique= data[data['critique']==j]
            critique= critique.reset_index(drop=True)
            if (j!=nouveauCritique.iloc[0][film] and critique.iloc[0][film] !="No rating"):
                    total+=(1/(1+sim_distanceManhattan(nouveauCritique.iloc[0,0], j)))*critique.iloc[0][film]
                    sim+=(1/(1+sim_distanceManhattan(nouveauCritique.iloc[0,0], j)))
    return  (total/sim)


# In[27]:


def Bestrecommend(nouveauCritique):
    nouveauCritique = data[data['critique']==nouveauCritique ]
    nouveauCritique= nouveauCritique.reset_index(drop=True)
    res=[ ]
    for i in range(1, nouveauCritique.shape[1]):
        if (nouveauCritique.iloc[0,i]=="No rating" ):
            res.append((recommend_complet_mod(nouveauCritique.iloc[0,0], nouveauCritique.columns[i]),nouveauCritique.columns[i]))
    return max(res)


# In[28]:


Bestrecommend('Anne')


# ### Pearson

# In[29]:


def pearson(pers1, pers2):
    pers1 = data[data['critique']==pers1]
    pers1= pers1.reset_index(drop=True)
    pers2 = data[data['critique']==pers2]
    pers2= pers2.reset_index(drop=True)
    sum_xy=0
    sum_x=0 
    sum_y=0
    sum_x2=0
    sum_y2=0
    n=0
    for i in range(1, pers1.shape[1]):
        if (pers1.iloc[0,i] !="No rating" and pers2.iloc[0,i] !="No rating"):
            n += 1 
            x=pers1.iloc[0,i] 
            y=pers2.iloc[0,i] 
            sum_xy +=x*y 
            sum_x += x 
            sum_y += y
            sum_x2 += x**2 
            sum_y2 += y**2
    denominator = ma.sqrt(sum_x2 - (sum_x**2) / n) * ma.sqrt(sum_y2 - (sum_y**2) / n)
    if denominator == 0: 
        return 0
    else:
        return (sum_xy - (sum_x * sum_y) /n ) / denominator


# In[30]:


pearson('Lisa Rose', 'Gene Seymour')


# In[31]:


def PearsonRecommend(nouveauCritique):
    nouveauCritique = data[data['critique']==nouveauCritique ]
    nouveauCritique= nouveauCritique.reset_index(drop=True)
    res=[ ]
    best=0
    best_film=[ ]
    for j in Critiques:
        if(j!=nouveauCritique.iloc[0,0]):
            critique= data[data['critique']==j]
            critique= critique.reset_index(drop=True)
            res.append((pearson(nouveauCritique.iloc[0,0], j),j))
    best= (max(res)[1])
    prox=data[data['critique']==best]
    prox=prox.reset_index(drop=True)
    for i in range(1, nouveauCritique.shape[1]):
        if (nouveauCritique.iloc[0,i]=="No rating" and prox.iloc[0,i] !='No rating'):
            best_film.append((prox.iloc[0,i], nouveauCritique.columns[i]))
    return (max(best_film))


# In[32]:


PearsonRecommend('Anne')


# ### Cosinus

# In[33]:


def cosinus(pers1, pers2):
    pers1 = data[data['critique']==pers1]
    pers1= pers1.reset_index(drop=True)
    pers2 = data[data['critique']==pers2]
    pers2= pers2.reset_index(drop=True)
    sum_xy=0
    sum_x=0 
    sum_y=0
    cos=0
    for i in range(1, pers1.shape[1]):
        if (pers1.iloc[0,i] !="No rating" and pers2.iloc[0,i] !="No rating"):
            x=pers1.iloc[0,i] 
            y=pers2.iloc[0,i] 
            sum_xy +=x*y 
            sum_x += x 
            sum_y += y
            cos= sum_xy/ (ma.sqrt(sum_x**2)* ma.sqrt(sum_y**2))
    return cos


# In[34]:


cosinus('Anne', 'Lisa Rose')


# In[35]:


def CosinusRecommend(nouveauCritique):
    nouveauCritique = data[data['critique']==nouveauCritique ]
    nouveauCritique= nouveauCritique.reset_index(drop=True)
    res=[ ]
    best=0
    best_film=[ ]
    for j in Critiques:
        if(j!=nouveauCritique.iloc[0,0]):
            critique= data[data['critique']==j]
            critique= critique.reset_index(drop=True)
            res.append((cosinus(nouveauCritique.iloc[0,0], j),j))
    best= (max(res)[1])
    prox=data[data['critique']==best]
    prox=prox.reset_index(drop=True)
    for i in range(1, nouveauCritique.shape[1]):
        if (nouveauCritique.iloc[0,i]=="No rating" and prox.iloc[0,i] !='No rating'):
            best_film.append((prox.iloc[0,i], nouveauCritique.columns[i]))
    return (max(best_film))


# In[36]:


CosinusRecommend('Anne')


# ### Raccomandation 5

# In[37]:


def SuperRecommend(nouveauCritique):
    r1=0
    r2=0
    r3=0
    nouveauCritique = data[data['critique']==nouveauCritique ]
    nouveauCritique= nouveauCritique.reset_index(drop=True)
    for i in range(1, nouveauCritique.shape[1]):
        if (nouveauCritique.iloc[0,i]=="No rating"):
            r1= PearsonRecommend(nouveauCritique.iloc[0,0])
            r2= CosinusRecommend(nouveauCritique.iloc[0,0])
            r3=recommend_mod(nouveauCritique.iloc[0,0])
    if(r1==0 and r2==0 and r3==0):
        print('Il/elle a vu tous les films')
        return
    print('Pearson: ',r1)
    print('Cosinus: ',r2)
    print('Manhattan: ',r3)
    return  


# In[38]:


SuperRecommend('Anne')


# In[39]:


SuperRecommend('Lisa Rose')


# ### New dataset

# In[47]:


data=pd.read_csv("new_film_mod.csv")


# In[48]:


data.fillna("No rating", inplace = True)


# In[49]:


data


# In[50]:


Critiques = [ ]
for i in range(len(data)):
    Critiques.append(data.iloc[i,0])


# In[51]:


SuperRecommend('Maccio')


# In[52]:


SuperRecommend('Genevieve')


# ### Conclusion

# #### Pearson

# D'après la mesure de Pearson, la critique Genevieve a donné des notes similaires à celles de Gennaro. Pearson est la mesure la plus efficace mais la plus complexe en terme de calcul  

# #### Cosinus

# Pour cosinus, Flavienne est la critique la plus proche de Genevieve cependant il n'y a pas de notes similaires avec Genevieve. Dans cet exemple, la mesure cosinus n'est pas optimale

# #### Manhattan

# Nous avons des données manquantes donc la distance de Manhattan n'est pas la plus efficace 
