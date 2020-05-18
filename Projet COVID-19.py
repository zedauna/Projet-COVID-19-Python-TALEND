#!/usr/bin/env python
# coding: utf-8

# ### IUT DE CARCASSONNE DE L’UNIVERSITÉ DE PERPIGNAN

# #### Semestre 4,                                                                                                                                                                 VIGAN JÉROS

# """
# Created on Mon May  4 12:29:20 2020  sur Spyder (Ipyhton)
# 
# @author: Jéros
# """

# 
# # =================================================================
# # THEME :Quelles sont les capacités de résilience des pays face à cette pandémie ? 
# # =================================================================

# ## Problématique  
# ### Évaluant les données de ruralité comme le taux de la population rurale et le taux de la terre arable
# ### Évaluant les données économiques comme le revenu national brut (RNB) et la population du monde en 2018
# ### Évaluant la dispersion de la pandémie dans les pays du monde
# ### Évaluant les liens entre les données de ruralité, données économiques et la pandémie 

# # =================================================================
# # Les modules de travail
# # =================================================================

# ### Gestion des Workpace, importation et modification des données

# In[4]:


import os 
import pandas as pd
import pandas.plotting
from pandas.plotting import scatter_matrix
import numpy as np


# ### Gestion desgraphiques et des modélisations , Dataviz  et animation en 3D

# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt
import squarify
from matplotlib import animation
from matplotlib.animation import FuncAnimation,FFMpegFileWriter
from mpl_toolkits.mplot3d import Axes3D


# ### Gestion des modéles d'analyses : ACP, Regression et Machine  Lernaning

# In[6]:


import scipy.stats
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression


# ### Gestion des productions des cartes

# In[7]:


import geopandas as gpd
import mapclassify as mc
#import libpysal as lps
# import geoplot as gplt
#import pysal as ps
#from pysal.contrib.viz import mapping as maps


# ### Gestion des cartes interactives et animations

# In[8]:


import folium
from folium.plugins import HeatMap
import PIL
import io
from ipywidgets import interact
import ipywidgets as widgets
from IPython.display import display
from IPython.display import Image
from IPython.display import HTML


# ### Gestion de la partie inférentielle 

# In[9]:


import matplotlib.mlab as mlab
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
import scipy
import statsmodels
from scipy.stats import chi2_contingency
from scipy.stats import ks_2samp
import scipy.stats as stats
import researchpy as rp


# # =================================================================
# #  Déclaration du dossier de travail
# # =================================================================

# In[10]:


base= r'D:\Navigation\Téléchargements\Cours Distance\Projet COVID\fichier'
base=base.replace('\\','/')
os.chdir(base)


# ### Chargement  des fichiers csv en localhost

# In[11]:


RNB = pd.read_csv('RNB.csv',sep=',')
Continent =pd.read_csv('continent1.csv',sep = ",")
TabMond =pd.read_csv('tableau-donnes-monde.csv',sep = ";")


# ### Chargement des données depuis le site pour avoir la mis à jour

# In[12]:


Deces = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
Infections = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
Guerisions= pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
pandemie = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv')
EvolPand =pd.read_csv('https://www.data.gouv.fr/fr/datasets/r/15a5a5b8-8330-48a0-a385-e01b326d2213',sep = ";" ,skiprows= 3)
EvolMond =pd.read_csv('https://www.data.gouv.fr/fr/datasets/r/f4935ed4-7a88-44e4-8f8a-33910a151d42',sep = ";" ,skiprows= 3)


# # =================================================================
# #  Vérification des données
# # =================================================================

# ### Information sur les donnés

# In[13]:


print(Continent.info())
print(TabMond.info())
#print(RNB.info())


# ## Summary

# In[14]:


EvolPand.describe()


# In[15]:


Continent.describe() 


# In[16]:


TabMond.describe()


# In[17]:


EvolMond.describe()


# In[18]:


RNB.describe()


# In[19]:


Deces.describe()


# In[20]:


pandemie.iloc[:,4:len(pandemie)].describe()


# ### Les dimensions de chaque tableau (dataFrame)

# In[21]:


print(Continent.shape);
print(EvolMond.shape);
print(TabMond.shape);
print(RNB.shape);
print(pandemie.shape);
print(EvolPand.shape);


# ### Affichage des 5 premières lignes des données

# In[22]:


TabMond.head(5)


# In[23]:


RNB.head(2)


# In[24]:


Continent.head(5)


# In[25]:


EvolMond.head(5)


# In[26]:


pandemie['Ratio']=pandemie['Recovered']/pandemie['Confirmed']
ratio=pandemie['Ratio']
pandemie.head(5)


# # =================================================================
# #  Traitement des données
# # =================================================================

# ### Convertion de la colonne date

# In[27]:


pandemie.Last_Update=pd.to_datetime(pandemie.Last_Update,format='%Y-%m-%d %H:%M:%S')
EvolPand.Date=pd.to_datetime(EvolPand.Date,format='%Y-%m-%d')
EvolMond.Date=pd.to_datetime(EvolMond.Date,format='%Y-%m-%d')
Continent.dateRep=pd.to_datetime(Continent.dateRep,format='%d/%m/%Y')

EvolPand['Date1']=EvolPand['Date'].apply(lambda x:x.strftime('%Y-%m'))


# ### Permuter l'ordre des colonnes

# In[28]:


cols = EvolPand.columns.tolist()
cols = cols[-1:] + cols[:-1]
cols
EvolPand = EvolPand[cols]

EvolPand= EvolPand.sort_values('Infections',ascending=False)


# ### Extration des données et fusion des données

# In[29]:


pandemie.rename(columns={'Country_Region':'Pays','Long_':'Long','Confirmed':'Infections','Deaths':'Decedes','Recovered':'Guerisions','ISO3':'Code'},inplace=True)
pandemie.columns


# In[30]:


RNB2014=RNB[['Country Code','2014']]
RNB2014.rename(columns={'Country Code':'Code','2014':'RNB'},inplace=True)
RNB2014.columns


# In[31]:


donnee=pd.merge(pandemie[['Pays','Lat','Long','Infections','Decedes','Guerisions','Code']],RNB2014, how='left', on='Code')
donnee.columns


# In[32]:


Continent.rename(columns={'countryterritoryCode':'Code','popData2018':'Pop2018','continentExp':'Continent'},inplace=True)
Continent.columns


# In[33]:


donnee=pd.merge(donnee,Continent[['Code','Pop2018', 'Continent']], how='left', on='Code')
donnee.columns


# In[34]:


TabMond.rename(columns={'AG.AGR.TRAC.NO':'Code','SP.RUR.TOTL.ZS':'TauxPopRural','AG.LND.ARBL.ZS':'TauxSurfRural'},inplace=True)
TabMond.columns


# In[35]:


donnee=pd.merge(donnee,TabMond[['Code','TauxPopRural', 'TauxSurfRural']], how='left', on='Code')
donnee.columns


# ### Permutation des colones

# In[36]:


donnee = donnee[['Pays', 'Code','Continent','Lat', 'Long', 'Infections','Decedes','Guerisions','RNB','Pop2018','TauxPopRural','TauxSurfRural']]


# In[37]:


list(donnee.columns.values)


# ### Estimations des taux des variables

# In[38]:


donnee['TauxInfections']=donnee['Infections'].apply(lambda x:(x/donnee['Pop2018'].sum(axis=0))*100000)
donnee['TauxDecedes']=donnee['Decedes'].apply(lambda x:(x/donnee['Pop2018'].sum(axis=0))*100000)
donnee['TauxGuerisions']=donnee['Guerisions'].apply(lambda x:(x/donnee['Pop2018'].sum(axis=0))*100000)
donnee['TauxRNB']=donnee['RNB'].apply(lambda x:(x/donnee['Pop2018'].sum(axis=0))*100000)
donnee['TauxPop2018']=donnee['Pop2018'].apply(lambda x:(x/donnee['Pop2018'].sum(axis=0))*100000)

donnee.columns


# ### Summary

# In[39]:


donnee.iloc[:,5:len(donnee)].describe()


# ### Completer la table Continent pour faire  les cartes

# In[40]:


Continent=pd.merge(Continent,donnee[['Code','TauxPopRural', 'TauxSurfRural','TauxInfections', 'TauxDecedes', 'TauxGuerisions', 'TauxRNB','TauxPop2018']], how='left', on='Code')
Continent=Continent[['dateRep', 'day', 'month', 'year', 'cases', 'deaths','countriesAndTerritories', 'geoId', 'Code','Continent','cases', 'deaths','TauxPopRural', 'TauxSurfRural', 'TauxInfections', 'TauxDecedes','TauxGuerisions', 'TauxRNB', 'TauxPop2018']]
Continent.columns


# # =============================================================================
# # Analyse univariée
# # =============================================================================

# In[41]:


EvolPand.iloc[:,[2,3,4]].describe()
print("-"*20)
print("moyenne:\n",EvolPand['Infections'].mean())
print("mediane:\n",EvolPand['Infections'].median())
print("variance:\n",EvolPand['Infections'].var(ddof=0))
print("std:\n",EvolPand['Infections'].std(ddof=0))
print("skweness:\n",EvolPand['Infections'].skew())
print("kurtosis:\n",EvolPand['Infections'].kurtosis())
   


# ### Évolution de la pandemie dans le monde (avec le module pandas)

# In[42]:


sns.set()
EvolPand.plot.scatter(x='Date',y='Infections',label='Evolution des infections dans le monde',c='black')
plt.savefig('EvolutionInfecNu.png')
EvolPand.plot.scatter(x='Date',y='Guerisons',label='Evolution des guérisions dans le monde',c='green')
plt.savefig('EvolutGuerisNU.png')
EvolPand.plot.scatter(x='Date',y='Deces',label='Evolution des décès dans le monde',c='red')
plt.grid(True)
plt.savefig('EvolutDeceNu.png')


# ### Histogramme l'évolution de Coronavirs dans le monde (mise à jour depuis le site des données)

# In[43]:


plt.style.use('seaborn-talk')
EvolPand['Infections'].plot.hist()
EvolPand['Guerisons'].plot.hist()
EvolPand['Deces'].plot.hist()
plt.title('Évolution de CORONAVIRUS dans le monde')
plt.legend()
plt.grid(True)
plt.savefig('EvolutionsCorna.png')


# ### Évolutions des allures de Coronavirus dans le monde

# In[44]:


EvolPand=EvolPand.sort_values('Date',ascending=True)
sns.set()
plt.style.use('seaborn-talk')
plt.plot(EvolPand['Date'],EvolPand['Infections'],label='Cas confirmés',color='black')
plt.plot(EvolPand['Date'],EvolPand['Deces'],label='Cas décédés',color='red')
plt.plot(EvolPand['Date'],EvolPand['Guerisons'],label='Cas guerris',color='green')
plt.xlabel('Date')
plt.ylabel('nombre de cas dans le monde')
plt.title('Évolution de Coronavirus dans le monde en date de'+' '+str(EvolPand ['Date'].max()),fontsize=20 )
plt.legend()
plt.show()
plt.savefig('evolution.png')


# ### Visualisation matricielle 

# In[45]:


EvolPand.info()
pandas.plotting.scatter_matrix(EvolPand.select_dtypes(exclude=['object','datetime64[ns]','float64']))
plt.savefig('MatrixPlotEvoCorno.png')


# ### Normalité des données

# #### On peut tester l’adéquation de l'infection à une loi normale à l’aide de test de normalité d'Agostino:

# In[46]:


stats.normaltest(donnee['Infections'])


# On peut donc rejetter l’hypothèse de normalité au niveau de test 5%.

# #### On peut tester l’adéquation de l'infection à une loi normale à l’aide de test de de Kolmogorov-Smirnov:

# In[47]:


ks_2samp(donnee['Infections'],list(np.random.normal(np.mean(donnee['Infections']), np.std(donnee['Infections']), 1000)))


# On peut donc rejetter l’hypothèse de normalité au niveau de test 5%.

# In[48]:


ks_2samp(donnee['Decedes'],list(np.random.normal(np.mean(donnee['Decedes']), np.std(donnee['Decedes']), 1000)))


# In[49]:


ks_2samp(donnee['Guerisions'],list(np.random.normal(np.mean(donnee['Guerisions']), np.std(donnee['Guerisions']), 1000)))


# In[50]:


ks_2samp(donnee['RNB'],list(np.random.normal(np.mean(donnee['RNB']), np.std(donnee['RNB']), 1000)))


# In[51]:


ks_2samp(donnee['Pop2018'],list(np.random.normal(np.mean(donnee['Pop2018']), np.std(donnee['Pop2018']), 1000)))


# In[52]:


ks_2samp(donnee['TauxPopRural'],list(np.random.normal(np.mean(donnee['TauxPopRural']), np.std(donnee['TauxPopRural']), 1000)))


# In[53]:


ks_2samp(donnee['TauxSurfRural'],list(np.random.normal(np.mean(donnee['TauxSurfRural']), np.std(donnee['TauxSurfRural']), 1000)))


# Dans l''ensemble, On  peut donc rejetter l’hypothèse de normalité au niveau de test 5%.

# # =================================================================
# # Analyse bivariée
# # =================================================================
# 

# ### Boxplot par continent des infections de Coronavirus

# In[54]:


# plt.figure(figsize=(8,6), dpi=80)
sns.set()
donnee.dropna().boxplot(column='Infections',by='Continent')
#plt.legend()
plt.savefig('InfectioContinet.png')

# plt.figure(figsize=(8,6), dpi=80)
sns.set()
donnee.dropna().boxplot(column='Decedes',by='Continent')
#plt.legend()
plt.savefig('DecedesContinet.png')

# plt.figure(figsize=(8,6), dpi=80)
sns.set()
donnee.dropna().boxplot(column='Guerisions',by='Continent')
#plt.legend()
plt.savefig('GuerisContinet.png')


# #### On peut tester l'égalité des variances à l’aide de test de levene:
#  H0 : égalité des variances
# contre  H1 : pas d’égalité des variances

# In[55]:


stats.levene(donnee['Infections'],donnee['Decedes'])


# La pvalue obtenue est inférieure à 5 %, donc  test significatif, rejet de H0, pas d'égalité des variances au niveau 5 %.

# #### On peut tester l'égalité des moyennes à l’aide de t.test:
# H0 : égalité des moyennes
# Contre  H1 : pas d’égalité des moyennes

# In[56]:


stats.wilcoxon(donnee['Infections'],donnee['Decedes'])


# La pvalue obtenue est inférieure à 5 %, donc test significatif, rejet de H0, pas d'égalité des moyennes au niveau 5 %.

# ### les statistiques au niveau monde

# In[57]:


donnee= donnee.sort_values('Infections',ascending=False)

colM1=['Infections','Decedes','Guerisions','RNB','Pop2018']
colM2=['TauxInfections','TauxDecedes','TauxGuerisions','TauxRNB','TauxPop2018','TauxPopRural','TauxSurfRural']


# In[58]:


donnee[colM1].sum()


# ### Matrice de corrélation et Significativité entre les variables

# In[59]:


donnee[colM1].corr()


# #### Autre méthode pour obtenir 3 tableaux

# In[60]:


corr_type, corr_matrix, corr_ps = rp.corr_case(donnee[colM1].dropna())
corr_ps


# On peut donc rejetter l’hypothèse d'indépendance au niveau de test 5% sauf RNB est non corrélé significativement à Pop2018 au niveau 5%.

# In[61]:


rp.corr_pair(donnee[colM1].dropna())


# ### Corrélation et significativité au 5% pour toutes les variables

# In[62]:


donnee[colM2].corr()


# In[63]:


corr_type, corr_matrix, corr_ps = rp.corr_case(donnee[colM2].dropna())
corr_type


# In[64]:


corr_matrix


# In[65]:


corr_ps


# In[66]:


rp.corr_pair(donnee[colM2].dropna())


# On peut donc rejetter l’hypothèse d'indépendance au niveau de test 5% sauf RNB,Pop2018,TauxPopRural et TauxSurfRural sont non corrélé significativement entre elles au niveau 5%.

# ### Visualisation des influences entre variables

# In[67]:


sns.pairplot(donnee[['Continent' ,'Infections','Decedes','Guerisions','RNB','Pop2018']].dropna(), hue="Continent", markers=["o", "s", "D","d","x"])


# In[68]:


sns.pairplot(donnee[['Continent' ,'TauxInfections','TauxDecedes','TauxGuerisions','TauxRNB','TauxPop2018','TauxPopRural','TauxSurfRural']].dropna(), hue="Continent", markers=["o", "s", "D","d","x"])


# ### Dataviz

# In[69]:


sns.pairplot(donnee[['Continent' ,'TauxInfections','TauxDecedes','TauxGuerisions','TauxRNB','TauxPop2018','TauxPopRural','TauxSurfRural']].dropna(), hue = 'Continent', diag_kind = 'kde',plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'},size = 2)
plt.suptitle('Dispersion de CORONAVIRUS entre Continent')


# ### Dispersion de la Pandémie sur l'EUROPE

# In[70]:


grid = sns.PairGrid(data= donnee[donnee['Continent']=='Europe'].dropna(), vars = ['TauxInfections', 'TauxDecedes', 'TauxGuerisions','TauxRNB','TauxPop2018','TauxPopRural','TauxSurfRural'])
grid = grid.map_upper(plt.scatter, color = 'darkred')
grid = grid.map_diag(plt.hist, bins = 10, color = 'darkred',edgecolor = 'k')
grid = grid.map_lower(sns.kdeplot, cmap = 'Reds')
plt.suptitle('Dispersion de CORONAVIRUS en EUROPE', size = 15)


# ###  Dispersion de la Pandémie sur l'EUROPE (avec correlation)

# In[71]:


def corr(x, y, **kwargs):
    
    coef = np.corrcoef(x, y)[0][1]
    label = r'$\rho$ = ' + str(round(coef, 2))
    ax = plt.gca()
    ax.annotate(label, xy = (0.2, 0.95), size = 20, xycoords = ax.transAxes)
    
grid = sns.PairGrid(data= donnee[donnee['Continent']=='Europe'].dropna(), vars = ['TauxInfections', 'TauxDecedes', 'TauxGuerisions','TauxRNB','TauxPop2018','TauxPopRural','TauxSurfRural'])
grid = grid.map_upper(plt.scatter, color = 'darkred')
grid = grid.map_upper(corr)
grid = grid.map_lower(sns.kdeplot, cmap = 'Reds')
grid = grid.map_diag(plt.hist, bins = 10, edgecolor =  'k', color = 'darkred');
plt.suptitle('Dispersion de CORONAVIRUS en EUROPE', size = 15)


# # =================================================================
# # Data visualisation (apprentissage)
# # =================================================================

# ### Visualisation & corrélation

# In[72]:


corr=donnee[colM1].corr()
corr1=donnee[colM2].corr()


# In[73]:


ax =sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)
plt.title('Matrice de Corrélation', fontsize = 20)
#plt.xlabel('Vari', fontsize = 15)
#plt.ylabel('Vari', fontsize = 15)
plt.show()


# In[74]:


ax =sns.heatmap(corr1,xticklabels=corr1.columns,yticklabels=corr1.columns)
plt.title('Matrice de Corrélation', fontsize = 20)
#plt.xlabel('Vari', fontsize = 15)
#plt.ylabel('Vari', fontsize = 15)
plt.show()


# ### Visualisation en 3D

# In[78]:


donnee['Continent']=pd.Categorical(donnee['Continent'])
my_color=donnee['Continent'].cat.codes
sns.set_style("white")
fig = plt.figure()
fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
ax = fig.add_subplot(111, projection='3d')
#ax.set_facecolor((0.5, 0.5, 0.5))
ax.scatter(donnee['Infections'],donnee['Decedes'],donnee['Guerisions'],c=my_color,alpha=0.8 ,cmap="Set2_r", s=60)
xAxisLine = ((min(donnee['Infections']), max(donnee['Infections'])), (0, 0), (0,0))
ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
yAxisLine = ((0, 0), (min(donnee['Decedes']), max(donnee['Decedes'])), (0,0))
ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
zAxisLine = ((0, 0), (0,0), (min(donnee['Guerisions']), max(donnee['Guerisions'])))
ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')
ax.set_xlabel("Infections")
ax.set_ylabel("Decedes")
ax.set_zlabel("Guerisions")
ax.set_title("Evolution de CORONAVIRUS dans le monde")
#plt.axis('off')
# plot.show()
plt.close()
#3D en animation
def update(i, fig, ax):
    ax.view_init(elev=20., azim=i)
    return fig, ax
 
anim = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), repeat=True, fargs=(fig, ax))
anim.save('evolution3D.gif', dpi=80, writer='imagemagick', fps=24)
HTML(anim.to_html5_video())


# ### Treemap  : Comparaison à la moyenne selon les variables en fonction des pays

# In[76]:


color_list = ['#0f7216', '#b2790c', '#ffe9a3','#f9d4d4', '#d35158', '#ea3033']

plt.rc('font', size=10)
squarify.plot(sizes=donnee[donnee['Infections']>donnee['Infections'].mean()]['Infections'], label=donnee['Code'], alpha=.8,color=color_list)
plt.axis('off')
plt.title("Infections Superieures à la Moyenne par Pays",fontsize=12,fontweight="bold")
plt.show()

plt.rc('font', size=10)
squarify.plot(sizes=donnee[donnee['Decedes']>donnee['Decedes'].mean()]['Decedes'], label=donnee['Code'], alpha=.8,color=color_list )
plt.axis('off')
plt.title("Décès Superieurs à la Moyenne par Pays",fontsize=12,fontweight="bold")
plt.show()

plt.rc('font', size=10)
squarify.plot(sizes=donnee[donnee['Guerisions']>donnee['Guerisions'].mean()]['Guerisions'], label=donnee['Code'], alpha=.8,color=color_list )
plt.axis('off')
plt.title("Guérisions Superieures à la Moyenne par Pays",fontsize=12,fontweight="bold")
plt.show()


# ### Customer Heatmap 

# In[79]:


def heatmap1(x, y, size):
    fig, ax = plt.subplots()
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num), 
        y=y.map(y_to_num), 
        s=size * size_scale,
        marker='s' 
    )
    
    
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    
corr2 = pd.melt(corr1.reset_index(), id_vars='index')
corr2.columns = ['x', 'y', 'value']
sns.set()
heatmap1(x=corr2['x'],y=corr2['y'],size=corr2['value'].abs())
ax.grid(False, 'major')
ax.grid(True, 'minor')
ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)


# # =================================================================
# # Carte du monde interactive
# # =================================================================

# ### Visualisation des pays touchées par le CORONAVIRUS 

# In[80]:


mpCororna = folium.Map(location=[10,10],zoom_start=1.5,tiles='Stamen Toner')
HeatMap(donnee[['Lat','Long']].dropna(), radius=16).add_to(mpCororna)
mpCororna.save('corona_mapa.html')
#allez visualizer l'image en cliquant sur le fichier HTML dans le wokspace
display(mpCororna)


# ### Classification des pays les plus inféctés et le nombre de décès  journaliers

# In[81]:


mpInfections=folium.Map(location=[42,5],zoom_start=6, max_zoom=5,min_zoom=2)
for i in range(0,len(Infections)):
  folium.Circle(
      location=[Infections.iloc[i]['Lat'],Infections.iloc[i]['Long']],
      fill=True,
      radius=(int((np.log(Infections.iloc[i,-1]+1.00001)))+0.2)*25000, #reduire la taille des cercles
      color='red',
      fill_color='indigo',
      tooltip = "<div style='margin: 0; background-color: black; color: white;'>"+
                    "<h4 style='text-align:center;font-weight: bold'>"+Infections.iloc[i]['Country/Region'] + "</h4>"
                    "<hr style='margin:10px;color: white;'>"+
                    "<ul style='color: white;;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+
                        "<li>Infections: "+str(Infections.iloc[i,-1])+"</li>"+
                        "<li>Deces d'aujourd'hui:   "+str(Deces.iloc[i,-1])+"</li>"+
                        "<li>Taux de mortalite d'aujourd'hui: "+ str(np.round(Deces.iloc[i,-1]/(Infections.iloc[i,-1]+1.00001)*100,2))+ "</li>"+
                    "</ul></div>",
        ).add_to(mpInfections)
mpInfections.save("infections.html")

#allez visualizer l'image en cliquant sur le fichier HTML dans le wokspace

display(mpInfections)


# ###  Classification des pays les plus retablis et le monde de décès journaliers

# In[82]:


mpGuerision=folium.Map(location=[46,2],zoom_start=6, max_zoom=5,min_zoom=2,tiles='Stamen Toner')
for i in range(0,len(Guerisions)):
  folium.Circle(
      location=[Guerisions.iloc[i]['Lat'],Guerisions.iloc[i]['Long']],
      fill=True,
      radius=(int((np.log(Guerisions.iloc[i,-1]+1.00001)))+0.2)*25000,#reduire la taille des cercles
      color='green',
      fill_color='green',
      legend_name='Guerisons',
      tooltip = "<div style='margin: 0; background-color: black; color: white;'>"+
                    "<h4 style='text-align:center;font-weight: bold'>"+Guerisions.iloc[i]['Country/Region'] + "</h4>"
                    "<hr style='margin:10px;color: white;'>"+
                    "<ul style='color: white;;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+
                        "<li>Guerisons: "+str(Guerisions.iloc[i,-1])+"</li>"+
                        "<li>Deces d'aujourd'hui:   "+str(Deces.iloc[i,-1])+"</li>"+
                        "<li>Taux de mortalite d'aujourd'hui: "+ str(np.round(Deces.iloc[i,-1]/(Infections.iloc[i,-1]+1.00001)*100,2))+ "</li>"+
                    "</ul></div>",
        ).add_to(mpGuerision)
mpGuerision.save("guerisions.html")

#allez visualizer l'image en cliquant sur le fichier HTML dans le wokspace
display(mpGuerision)


# #### Carte avec Ratio en curseur 

# # =============================================================================
# # carte du monde
# # =============================================================================
# 

# ### Importation de la carte

# In[84]:


Monde= gpd.read_file("pays-monde.shp")
type(Monde)
Monde.head()


# ### Fusion des données

# In[85]:


donnee['ISO3']=donnee['Code']
donnee.columns
Monde=pd.merge(Monde,donnee[['Continent','Infections','Decedes','Guerisions','RNB','Pop2018','TauxPopRural', 'TauxSurfRural','TauxInfections', 'TauxDecedes', 'TauxGuerisions', 'TauxRNB','TauxPop2018','ISO3']], how='left', on='ISO3')
Monde1=Monde.dropna()


# In[86]:


sns.set_style("white")
ax = Monde1.plot(column='Infections',cmap='Blues',figsize=(15, 15),legend=True, alpha=0.5, edgecolor='k',linewidth=0.4,scheme='Quantiles', k=5)
ax.set_title('Nombres d''infections par le CORONAVIRUS par pays', fontdict = {'fontsize':20}, pad = 12.5) 
#ax.set_axis_off()   Pour supprimer les axes
ax.get_legend().set_bbox_to_anchor((0.2, 0.6))
ax.get_legend().set_title('Legende')


# In[87]:


sns.set_style("white")
ax = Monde1.plot(column='Guerisions',cmap='Greens',figsize=(15, 15),legend=True, alpha=0.5, edgecolor='k',linewidth=0.4,scheme='Quantiles', k=5)
ax.set_title('Nombre de guéris du CORONAVIRUS par pays', fontdict = {'fontsize':20}, pad = 12.5) 
#ax.set_axis_off()
ax.get_legend().set_bbox_to_anchor((0.2, 0.6))
ax.get_legend().set_title('Legende')


# In[88]:


sns.set_style("white")
ax = Monde1.plot(column='Decedes',cmap='Reds',figsize=(15, 15),legend=True, alpha=0.5, edgecolor='k',linewidth=0.4,scheme='Quantiles', k=5)
ax.set_title('Nombre de décès du CORONAVIRUS par pays', fontdict = {'fontsize':20}, pad = 12.5) 
#ax.set_axis_off()
ax.get_legend().set_bbox_to_anchor((0.2, 0.6))
ax.get_legend().set_title('Legende')


# #### Personnalisation des cartes scheme = 'user_defined', classification_kwds = {'bins':[10, 20, 50, 100, 500, 1000, 5000, 10000, 500000]}

# In[89]:


sns.set_style("white")
ax = Monde1.plot(column='Decedes',cmap='GnBu',figsize=(15, 15),legend=True, alpha=0.5, edgecolor='k',linewidth=0.4,scheme = 'user_defined', classification_kwds = {'bins':[10, 20, 50, 100, 500, 1000, 5000, 10000, 500000]})
ax.set_title('Nombre de décès du CORONAVIRUS par pays', fontdict = {'fontsize':20}, pad = 12.5) 
ax.set_axis_off()
ax.get_legend().set_bbox_to_anchor((0.2, 0.6))
ax.get_legend().set_title('Legende')


# ### Carte en gif 

# In[ ]:


#aggregation par date
InfectionsData = Infections.groupby('Country/Region').sum()
DecesData = Deces.groupby('Country/Region').sum()
GuerisionsData = Guerisions.groupby('Country/Region').sum()

#suppression des colonnes 
InfectionsData =InfectionsData.drop(columns = ['Lat', 'Long'])
DecesData =DecesData.drop(columns = ['Lat', 'Long'])
GuerisionsData =GuerisionsData.drop(columns = ['Lat', 'Long'])

#Rechargement de la carte
Monde= gpd.read_file("pays-monde.shp")

#Changement des noms pour faciliter la jointure
Monde.replace('Viet Nam', 'Vietnam', inplace = True)
Monde.replace('Brunei Darussalam', 'Brunei', inplace = True)
Monde.replace('Cape Verde', 'Cabo Verde', inplace = True)
Monde.replace('Democratic Republic of the Congo', 'Congo (Kinshasa)', inplace = True)
Monde.replace('Congo', 'Congo (Brazzaville)', inplace = True)
Monde.replace('Czech Republic', 'Czechia', inplace = True)
Monde.replace('Swaziland', 'Eswatini', inplace = True)
Monde.replace('Iran (Islamic Republic of)', 'Iran', inplace = True)
Monde.replace('Korea, Republic of', 'Korea, South', inplace = True)
Monde.replace("Lao People's Democratic Republic", 'Laos', inplace = True)
Monde.replace('Libyan Arab Jamahiriya', 'Libya', inplace = True)
Monde.replace('Republic of Moldova', 'Moldova', inplace = True)
Monde.replace('The former Yugoslav Republic of Macedonia', 'North Macedonia', inplace = True)
Monde.replace('Syrian Arab Republic', 'Syria', inplace = True)
Monde.replace('Taiwan', 'Taiwan*', inplace = True)
Monde.replace('United Republic of Tanzania', 'Tanzania', inplace = True)
Monde.replace('United States', 'US', inplace = True)
Monde.replace('Palestine', 'West Bank and Gaza', inplace = True)

#Jointure des tables grace aux noms
mergeInfections = Monde.join(InfectionsData, on = 'NAME', how = 'right')
mergeDeces = Monde.join(DecesData, on = 'NAME', how = 'right')
mergeGuerisions = Monde.join(GuerisionsData, on = 'NAME', how = 'right')


# #### infections.gif

# In[ ]:


image_frames = []

#Il faut aciver
#for dates in mergeInfections.columns.to_list()[12:len(mergeInfections.columns)]:
  
    ax = mergeInfections.plot(column = dates, 
                    cmap = 'Blues', 
                    figsize = (15,15), 
                    legend = True,
                    alpha=0.5,
                    scheme = 'user_defined', 
                    classification_kwds = {'bins':[10, 20, 50, 100, 500, 1000, 5000, 10000, 500000]}, 
                    edgecolor = 'black',
                    linewidth = 0.4)
    
    ax.set_title('Nombres d''infections par le CORONAVIRUS par pays: '+ dates, fontdict = 
                 {'fontsize':20}, pad = 12.5)
    
    ax.set_axis_off()
     
    ax.get_legend().set_bbox_to_anchor((0.18, 0.6))
    ax.get_legend().set_title('Legende')
     
    img = ax.get_figure()
    
    
    #f = io.BytesIO()
    img.savefig(f, format = 'png', bbox_inches = 'tight')
    #f.seek(0)
    image_frames.append(PIL.Image.open(f))


# In[ ]:


image_frames[0].save('InfectionsPays.gif', format = 'GIF',
            append_images = image_frames[1:], 
            save_all = True, duration = 300, 
            loop = 3)

f.close()


# In[90]:


Image('InfectionsPays.gif')


# In[91]:


Image('Deces.gif')


# In[95]:


Image('Guerision.gif')


# ## Outils de travail
# ### Talend pour nettoyer et rechantionnger certaines données
# ### Spyder (Ipython) pour traiter les données
# ### Jupyter notebook pour la redaction et mise ne page

# Veuillez voir les cartes gif en workpace

# # CONCLUSION

# En somme, le projet COVID-19 montre une dépendance de cette pandémie de CORONAVIRUS face aux capacités de résilience des pays 
# 
# étudiées et la propagation de cette pandémie est spécifiquement variable en fonction de chaque pays.
