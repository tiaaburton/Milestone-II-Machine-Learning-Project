"""
This is a module that trains and saves the model.
Then it saves outputs to directory
"""
#!pip install --upgrade s-dbw
#!pip install imbalanced-learn
#!pip install scikit-learn-extra
#!pip install factor_analyzer
#!pip install prince
# !pip install selenium
import pandas as pd
from PIL import Image
from bokeh.io import export_png, export_svgs
from bokeh.models import ColumnDataSource, DataTable, TableColumn
from sklearn.neighbors import KNeighborsClassifier
import prince
import pickle
import pickle5 as pickles
from sklearn_extra.cluster import KMedoids
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
# %matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import os
# !pip install factor_analyzer
import factor_analyzer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.stats import zscore
from sklearn import cluster
from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabasz_score
from sklearn import preprocessing
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

#Ignore warnings
import warnings
import numpy as np
from datetime import datetime
from collections import Counter
from collections import defaultdict
from tqdm import tqdm
from random import choices
from sklearn.cluster import DBSCAN
from sklearn import cluster
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids
from sklearn.linear_model import SGDClassifier
import sklearn
from sklearn import linear_model
from sklearn.metrics import precision_score
# import umap
print("0- importing packages")
    
# 1- IMPORTING DATA - RENAMING COLUMNS

# read file funtion
def read_file():
    # path="datasets/A1_B2_data.csv"
    path="../datasets/model_files/A1_B2_data.csv"
    df=pd.read_csv(path)
    
    if 'fullVisitorId' and "date" in df.columns:
        df.set_index(['date','fullVisitorId'],inplace=True)
    try:
        del df['socialEngagementType']
        del df['buyers']
    except:
        pass
    
    # removes the dates multi index (can be retrieved later)
    df.reset_index(inplace=True)
    try:
        dates=df["date"].values
    except:
        pass
    try:
        del df["date"]
    except:
        pass
    try:
        del df["clusters"]
    except:
        pass
    # there are 2 nas
    df.fillna(0,inplace=True)
    df.set_index('fullVisitorId',inplace=True)
    
    colonnes=['hits',
     'pageviews',
     'timeOnSite',
     'browser_woe',
     'country_woe',
     'source_woe',
     'transactions',
     'hits.eCom',
     'hour_ordinal',
     'newVisits',
     'bounces',
     'c_Direct',
     'c_Display',
     'c_Organic',
     'c_PaidSearch',
     'c_Referral',
     'c_Social',
     'medium_affiliate',
     'medium_cpc',
     'medium_cpm',
     'medium_organic',
     'medium_referral',
     'isDirect',
     'OS_Android',
     'OS_BlackBerry',
     'OS_Chrome OS',
     'OS_Firefox OS',
     'OS_Linux',
     'OS_Macintosh',
     'OS_Nintendo Wii',
     'OS_Samsung',
     'OS_Windows',
     'OS_Windows Phone',
     'OS_Xbox',
     'OS_iOS',
     'device_mobile',
     'device_tablet',
     'recency',
     'monetary',
     'frequency',
     'repurchasers']

    
    dico={'totals.hits':"hits",
     'totals.pageviews':"pageviews",
     'totals.timeOnSite':"timeOnSite",
    'totals.transactions':"transactions",
     'totals.newVisits':"newVisits",
     'totals.bounces':"bounces",
     'hits.eCommerceAction.action_type':'hits.eCom',
     'Monetary':"monetary",
     'Frequency':"frequency",
     'repurchasers':"repurchasers",
     'Recency':"recency",
     'hits.hour_ordinal':"hour_ordinal",
            'channelGrouping_Direct':'c_direct',
     'channelGrouping_Display':'c_display',
           'channelGrouping_Organic Search':'c_organic_search',
     'channelGrouping_Paid Search':'c_paid_search',
           'channelGrouping_Referral':'c_referral',
     'channelGrouping_Social':"c_social",
           'geoNetwork.country_woe':'country_woe',
     'trafficSource.source_woe':"source_woe",
           'trafficSource.medium_affiliate':"med_affiliate",
     'trafficSource.medium_cpc':"med_cpc",
           'trafficSource.medium_cpm':"med_cpm",
     'trafficSource.medium_organic':'med_organic',
           'trafficSource.medium_referral':'med_referral',
     'trafficSource.isTrueDirect_code':'direct_is_true',
           'device.browser_woe':"browser_woe",
     'device.operatingSystem_Android':'OS_android',
           'device.operatingSystem_BlackBerry':'OS_blackberry',
     'device.operatingSystem_Chrome OS':'OS_chrome',
           'device.operatingSystem_Firefox OS':'OS_firefox',
     'device.operatingSystem_Linux':'OS_linux',
           'device.operatingSystem_Macintosh':'OS_mac',
           'device.operatingSystem_Nintendo Wii':'OS_wii',
     'device.operatingSystem_Samsung':'OS_samsung',
           'device.operatingSystem_Windows':'OS_windows',
           'device.operatingSystem_Windows Phone':"OS_windwsphone",
     'device.operatingSystem_Xbox':'OS_xbox',
           'device.operatingSystem_iOS':'OS_ios',
     'device.deviceCategory_mobile':'mobile',
           'device.deviceCategory_tablet':'tablet'}
    
    df.rename(columns = dico, inplace = True)
    
    # if drop_first_column=False in one-hot encoding
    try:
        df.rename(columns = {'device.deviceCategory_desktop':'desktop',
                             'channelGrouping_Affiliates':'c_affiliates',
                             'device.operatingSystem_(not set)':'OS_notset',
                             'trafficSource.medium_(none)':'med_none'}
                             , inplace = True)
        
    except:
        pass
    print("1- downloading file") 
    return df
      
# 2- REBALANCING CLASSES

def class_rebalance(df):
    try:
        del df["CLUSTERS"]
    except:
        pass
    
    df["buyers"]=df["monetary"]>0
    targetclass="buyers"

    cols=list(df.columns)
    cols.remove(targetclass)

    ## set predictor x 
    x=df[cols]
    ## set y target
    y=df[targetclass]
    
    # rebalance
    smote = SMOTE(random_state=655)
    # fit predictor and target variable
    x_smote, y_smote = smote.fit_resample(x, y)
    # print('Original dataset shape', Counter(y))
    # print('Resample dataset shape', Counter(y_smote))
    
    fig = plt.figure()
    fig.set_size_inches(8, 3)

    ax1 = plt.subplot(121)
    ax1=sns.countplot(df[targetclass])
    ax1.set_xlabel("Buyers",fontsize=12)
    ax1.set_xticklabels(['0','1'])
    ax1.set_ylabel("Count",fontsize=12)
    ax1.set_title("Before rebalancing",fontsize=14)
    ax1.set_facecolor('#EAEAF2')
    
    ax2 = plt.subplot(122)
    ax2=sns.countplot(y_smote)
    ax2.set_xlabel("Buyers",fontsize=12)
    ax2.set_xticklabels(['0','1'])
    ax2.set_ylabel(" ",fontsize=12)
    ax2.set_title("After rebalancing",fontsize=14)
    ax2.set_facecolor('#EAEAF2')
    
    
    fig.savefig('../visualizations/rebalancing.png', bbox_inches='tight')
    
    rebalanced_df=pd.concat([x_smote, y_smote], axis=1)
    del rebalanced_df[targetclass]
    del df[targetclass]
    print("2- rebalancing classes") 
    return rebalanced_df

#3- SCALING DATA SETS   
# scaling and unscaling handy functions, returns either a DataFrame or a numpy array

def unscale(df_scaled,scaler,array=False):
    if not array:        
        df_unscaled=pd.DataFrame(scaler.inverse_transform(df_scaled),index=df_scaled.index, columns=list(df_scaled.columns))
    else:
        df_unscaled=scaler.inverse_transform(df_scaled)
    return df_unscaled

# returns scaled df or array depending on array option
def scale(df,num_cols,array=False):
    
    idx=df.index
    cols=list(df.columns)

    
    # identifying numeric columns
    
    
    
    # identifying categorical columns
    cat_cols=list(set(cols)-set(num_cols))
    
    # fitting scaler and then transforming 
    scaler = preprocessing.StandardScaler().fit(df[num_cols])

    num_arr=scaler.transform(df[num_cols])
    
    # concatenating with categorical values
    cat_arr=np.array(df[cat_cols])
    scaled_arr=np.hstack((num_arr,cat_arr))
    
    # returning final columns
    final_cols=num_cols+cat_cols
    
    print("3- scaling df") 
    # if array==True return array 
    if array:
        return scaled_arr, scaler, idx, final_cols
    
    else:     
        df_scaled=pd.DataFrame(data=scaled_arr,index=idx, columns=final_cols)
        return df_scaled, scaler, idx, final_cols 


    
## 4- PERFORMING DIMENSION REDUCTION ANALYSIS
### 4.1.  PCA
# visualize PCA optimal number of components

def spree_PCA(X_scaled,download):        
    try:
        del X_scaled["CLUSTERS"]
    except:
        pass

    pca1 = PCA(random_state=655).fit(X_scaled)
    liste_pca=list(np.cumsum(pca1.explained_variance_ratio_))
    liste_pca.append(0)
    liste_pca=sorted(liste_pca)
    pca_cum=pd.Series(liste_pca, index=range(0,len(liste_pca)),name="pca_cum").round(3)
    pca_cum.to_csv("../results/pca_cum.csv")    
    fig = plt.figure()
    fig.set_size_inches(14, 4)

    # Visualizing PCA
    x_values2=np.arange(0,5)
    ax0 = plt.subplot(121)
    ax0=pca_cum.plot(marker='o',markersize=4)
    ax0.set_xlabel('N components',size=14)
    ax0.set_ylabel('Cum explained variance',size=14)
    ax0.set_title("PCA Spree Chart", fontsize=14, fontweight='bold',pad=10)
    ax0.axvline(x=2,color='black',alpha=0.5,ls="--")   
    fig.savefig('../visualizations/pca_spree_full.png', bbox_inches='tight')
    print("4.1. returning spree chart") 
    return fig,pca_cum

## transforming data with PCA=2 and visualizing components
def heat_map_pca(groupe_z):  
    fig = plt.figure()
    fig.set_size_inches(40, 1)    
    ax = plt.subplot(121)
    ax = sns.heatmap(groupe_z,vmin=-2, vmax=2, cmap=sns.diverging_palette(400,-200, n=10),center=0,annot=False,
                     linewidths=.5, linecolor='white')
    ax.set_title("PCA components interpretation",fontsize=12)     
    print("4.1. returning pca heatmap")
    return fig


def pca_transfo_full(df, download):
    if download:
        try:
            filename="../models/pca_model_full.sav"
            pca_model = pickle.load(open(filename, 'rb'))
            pca_components=pd.read_csv("../results/pca_components_full.csv",index_col=0)
            pca_transformed=pd.read_csv("../results/pca_transformed_full.csv",index_col=0)   
            heat_map = Image.open('../visualizations/pca_heat_map_full.png')
            print("4.1. downloading pca transformed and model full")
            return pca_transformed,pca_components,heat_map,pca_model            
        except:
            pass

    col_pca=[]
    cols=list(df.columns)
    index=df.index
    pcn=2
    X_scaled=df

    pca2=PCA(n_components=pcn,random_state=655).fit(X_scaled)
    pca_transformed=pca2.transform(X_scaled)
    pca_components=pd.DataFrame(pca2.components_.transpose(),index=cols)
    pca_components.columns=['PC1','PC2']

    pca_transformed=pd.DataFrame(data=pca_transformed,columns=['PC1','PC2'], index=index)

    heat_map_figure=heat_map_pca(np.round(pca_components.T,3))
    
     # saves pca models and dependencies mode
    heat_map_figure.savefig('../visualizations/pca_heat_map_imbalanced.png', bbox_inches='tight') 
    filename="../models/pca_model_full.sav"
    pickle.dump(pca2, open(filename, 'wb'))
    pca_transformed.to_csv("../results/pca_transformed_full.csv")
    pca_components.to_csv("../results/pca_components_full.csv") 
    print("4.1. processing and saving pca balanced class")
    return pca_transformed,pca_components,heat_map_figure,pca2

def pca_transfo(df, download):
    if download:
        try:
            filename="../models/pca_model.sav"
            pca_model = pickle.load(open(filename, 'rb'))
            pca_components=pd.read_csv("../results/pca_components.csv",index_col=0)
            pca_transformed=pd.read_csv("../results/pca_transformed.csv",index_col=0)   
            heat_map_figure = Image.open('../visualizations/pca_heat_map.png')
            heat_map.show()
            print("4.1. downloading pca transformed and model (imbalanced)")
            return pca_transformed,pca_components,heat_map_figure,pca_model
            
        except:
            pass
   
    col_pca=[]
    cols=list(df.columns)
    index=df.index
    pcn=2
    X_scaled=df

    pca2=PCA(n_components=pcn,random_state=655).fit(X_scaled)
    pca_transformed=pca2.transform(X_scaled)
    pca_components=pd.DataFrame(pca2.components_.transpose(),index=cols)
    pca_components.columns=['PC1','PC2']

    pca_transformed=pd.DataFrame(data=pca_transformed,columns=['PC1','PC2'], index=index)

    heat_map_figure=heat_map_pca(np.round(pca_components.T,3))

    # saves pca models and dependencies mode
    heat_map_figure.savefig('../visualizations/pca_heat_map_imbalanced.png', bbox_inches='tight') 
    filename="../models/pca_model.sav"
    pickle.dump(pca2, open(filename, 'wb'))
    pca_transformed.to_csv("../results/pca_transformed.csv")
    pca_components.to_csv("../results/pca_components.csv") 
    print("4.1. processing and saving pca imbalanced data set")
    
    return pca_transformed,pca_components,heat_map_figure,pca2




### 4.2. FAMD 

# TRANSFORMING df with FAMD
def  famd_visualizations(df_cat_scaled,famd, col):
    # creates the buyers class for visualization and change label
    df_cat_scaled["buyers"]=df_cat_scaled["monetary"]>0
    df_cat_scaled=df_cat_scaled.replace({False: "Non-buyers", True: "buyers"})
    
    # 
    ax=famd.plot_row_coordinates(df_cat_scaled,
                                ax=None,
                                figsize=(7, 6),
                                x_component=0,
                                y_component=1,
                                # labels=X.index,
                                color_labels=['{}'.format(t) for t in df_cat_scaled[col]],
                                ellipse_outline=False, s=2.5,alpha=0.5,
                                ellipse_fill=True,
                               show_points=True)
    ax.set_title("FAMD By {}".format(col),fontsize=15)
    ax.set_xlabel("PC 1",size=14)
    ax.set_ylabel("PC 2",size=14)
    ax.set_xlim(-3, 8)
    ax.set_ylim(-3, 6)
    ax.grid(True)
    ax.legend(fontsize=12,markerscale=5,loc=2,facecolor="white")
    fig = ax.get_figure()
    del df_cat_scaled["buyers"]
    print("4.2- returning famd visualizations")
    return fig


def famd(download):
    if download:
        try:
            famd_channel_fig = Image.open('../visualizations/famd_by_channel.png')
            famd_buyers_fig = Image.open('../visualizations/famd_by_buyers.png')
            df_famd=pd.read_csv("../results/famd_transformed_df.csv",index_col=0)
            df_cat_scaled=pd.read_csv("../datasets/df_cat_scaled.csv",index_col=0)
            filename="../models/famd.sav"
            famd_model=pickle.load(open(filename, 'rb'))
            print("4.2- downloading famd df, model and visualizations")
            return df_cat_scaled,df_famd,famd_model,famd_channel_fig,famd_buyers_fig
            
        except:
            pass

    num_cols=['pageviews','timeOnSite','transactions',
                 'hits.eCom','hour_ordinal',
                  'recency','monetary']
    cat_columns=['channelGrouping', 'geoNetwork.country',
       'trafficSource.source', 'trafficSource.medium', 'device.browser',
       'device.operatingSystem', 'device.deviceCategory']
    
    # recovers drop features
    filename="../datasets/model_files/A1_B2_data_dropped_features.csv"
    df_cat=pd.read_csv(filename)
    df_cat=df_cat[cat_columns]
    df_cato=read_file()
    df_cat.index=df_cato.index
    
    # concatenate with numeric features to form the mixed data frame
    df_concat=pd.concat([df_cato[num_cols],df_cat],axis=1)
    df_concat.head()
    
    # scale the mixed df 
    df_cat_scaled, scaler_cat, idx_cat, cols_cat=scale(df_concat,num_cols) 
  

    # convert to respective datatypes for famd to work
    df_cat_scaled.fillna(0,inplace=True)
    for col in num_cols:
        df_cat_scaled[col] = df_cat_scaled[col].astype(float)
    for col in cat_columns:
        df_cat_scaled[col] = df_cat_scaled[col].astype("object")
        
    # fits famd model
    famd = prince.FAMD(n_components=2,n_iter=3,copy=True,check_input=True,engine='auto',random_state=655)
    X = famd.fit(df_cat_scaled)
    df_famd=famd.transform(df_cat_scaled)
    famd_model=X
    
    # saves df
    df_famd.to_csv("../results/famd_transformed_df.csv")
    # saves cat_scaled
    df_cat_scaled.to_csv("../datasets/df_cat_scaled.csv")
    
    # saves famd model
    filename="../models/famd.sav"
    pickle.dump(X, open(filename, 'wb'))

    
    #help(famd)
    # famd.eigenvalues_
    #returns correlation
    correlations=np.round(famd.column_correlations(df_cat_scaled).sort_values(by=1,ascending=False).head(10),3)
    # X.explained_inertia_

    # returning and save visualizations
    famd_channel_fig=famd_visualizations(df_cat_scaled,X,col='channelGrouping')
    famd_channel_fig.savefig('../visualizations/famd_by_channel.png', bbox_inches='tight')
    
    famd_buyers_fig=famd_visualizations(df_cat_scaled,X,col='buyers')
    famd_buyers_fig.savefig('../visualizations/famd_by_buyers.png', bbox_inches='tight')
    print("4.2- returning famd df, model and visualizations")
    return df_cat_scaled,df_famd,famd_model,famd_channel_fig,famd_buyers_fig




## 5- PERFORMING K-means clustering
### 5.1 Calculating metrics to find optimal k


def save_df_as_image(df, path):
  try:
    source = ColumnDataSource(df)
    df_columns = [df.index.name]
    df_columns.extend(df.columns.values)
    columns_for_table=[]
    for column in df_columns:
        columns_for_table.append(TableColumn(field=column, title=column))

    data_table = DataTable(source=source, columns=columns_for_table,height_policy="auto",width_policy="auto",index_position=None)
    export_png(data_table, filename = path)
  except Exception as e:
    print("error with saving DF as image")


# takes a file and returns cores for k-means
def loop_cluster(df,download,pca=False,full=True):
    
    # tries to download the cache
    if download and not pca:
        print("5.1- downloading kmeans scores")
        try:
            if full:
                scores=pd.read_csv("../results/scores_kmeans_full.csv")
                scores.rename(columns={"Unnamed: 0":"k clusters"},inplace=True)
                # print('downloading')
                return scores
            else:
                scores=pd.read_csv("../results/scores_kmeans.csv")
                scores.rename(columns={"Unnamed: 0":"k clusters"},inplace=True)
                # print('downloading')
                return scores
                
        except:
            pass
        
    if download and pca:
        print("5.1- downloading kmeans scores")
        try:
            if full:
                scores=pd.read_csv("../results/scores_pca_kmeans_full.csv")
                scores.rename(columns={"Unnamed: 0":"k clusters"},inplace=True)
                # print('downloading')
                return scores
            else:
                scores=pd.read_csv("../results/scores_pca_kmeans.csv")
                scores.rename(columns={"Unnamed: 0":"k clusters"},inplace=True)
                # print('downloading')
                return scores
        except:
            pass
            
    
    if "CLUSTERS" in df.columns:
        del df["CLUSTERS"]
    arr=np.array(df)
    start=2
    end=8
    scores=defaultdict(list)
    for n in tqdm(range(start,end)):
        model=cluster.KMeans(n_clusters=n,random_state=655) #10
        model.fit(arr)      
        labels= model.labels_
        scores["inertia"].append(model.inertia_)
        scores["silhouette"].append(round(silhouette_score(df,labels),2))
        scores["bouldin"].append(round(davies_bouldin_score(df,labels),2))  
        scores["calinski"].append(np.int64(calinski_harabasz_score(df,labels)))
        # sd_score=SD(arr, labels,centers_id=None,  alg_noise='bind',centr='mean', nearest_centr=True, metric='euclidean')
        # scores["S_Dbw"].append(round(sd_score,2))  
  
    scores_df=pd.DataFrame(data=scores, index=range(start,end))
    if pca:
        if full:
            scores_df.to_csv("../results/scores_pca_kmeans_full.csv")  
        else:
            scores_df.to_csv("../results/scores_pca_kmeans.csv") 
            
    else:
        if full:
            scores_df.to_csv("../results/scores_kmeans_full.csv") 
        else:
            scores_df.to_csv("../results/scores_kmeans.csv") 
            
    print("5.1- processing kmeans scores")  
    return scores_df

# returns scores for all 4 datasets (15 minutes to run, so please select the load=True )

def scores_kmeans(df_scaled,df_scaled_full,pca_components,pca_components_full,download=True):   
    scores1=loop_cluster(df_scaled,download,pca=False,full=False)
    scores2=loop_cluster(df_scaled_full,download,pca=False,full=True)
    scores3=loop_cluster(pca_components,download,pca=True,full=False)
    scores4=loop_cluster(pca_components_full,download,pca=True,full=True).round(2)
    save_df_as_image(scores4, path="../visualizations/scores4_table.png")
    
    return scores1,scores2,scores3,scores4

def plot_elbow_complete(cluster_scores,download):
    
    start=2
    end=8
    titres=list(cluster_scores.columns)
    colors = ['red', 'orange', 'blue',"green"]

    labels=[x for x in range(start-1,end-1)]
    labelsX=[x for x in range(start-1,end)]
    
    fig, axs = plt.subplots(1,4)
    fig.set_size_inches(18, 4)
    
    for i in range (0,4):
        y=cluster_scores.iloc[:,i].values
        axs[i].plot(labels,y,color=colors[i])
        axs[i].set_title(titres[i],fontsize=14)
        axs[i].set_xticklabels(labelsX, fontdict=None, minor=False,rotation=0,size=12)
        axs[i].set_xlabel("N clusters",size=14)
        if i==0:
            axs[i].set_ylabel("Inertia",size=14)
    fig.savefig('../visualizations/elbow_kmeans.png', bbox_inches='tight')
    plt.show()
    print("5.1- returning elbow scores")  
    return fig
                           
def join_data_sets(scores1,scores2,scores3,scores4,download):
    if download:
        try:           
            silhouette_fig=Image.open('../visualizations/silhouette_score.png')
            total=pd.read_csv("../results/total_scores_k5.csv",index_col=0)
            elbow=plot_elbow_complete(total,download=True)
            print("5.1- downloading elbow scores, silhouette and score tables")  
            return total.round(2),elbow,silhouette_fig
        except:
            pass
    
    
    total=scores1
    total["All feat"]=total["inertia"].values
    total["All feat bal."]=scores2["inertia"]
    total["PCA imb."]=scores3["inertia"]
    total["PCA balanced"]=scores4["inertia"]
    try:
        total.set_index("k clusters", inplace=True)
    except:
        pass
    
    total=total[["All feat","All feat bal.","PCA imb.","PCA balanced"]]
    total.to_csv("../results/total_scores_k5.csv")
    elbow=plot_elbow_complete(total,download)


    silhouette={}
    silhouette["All feat"]=scores1.silhouette.values
    silhouette["All feat bal."]=scores2.silhouette.values
    silhouette["PCA imb."]=scores3.silhouette.values
    silhouette["PCA balanced"]=scores4.silhouette.values
    df_sil=pd.DataFrame(data=silhouette).iloc[4,:]
    df_sil.values
    ax=sns.barplot(y=df_sil.values,x=df_sil.index, color="green",alpha=0.5)
    ax.set_title("Silhouette Score for K= 5 by data set", size=13)
    ax.grid(True)
    for element in (ax.get_xticklabels() + ax.get_yticklabels()):
        element.set_fontsize(11)
        
    silhouette_fig = ax.get_figure()
    silhouette_fig.set_size_inches(4.5, 4)
    silhouette_fig.savefig('../visualizations/silhouette_score.png', bbox_inches='tight')
    print("5.1- processing elbow scores, silhouette and score tables") 
    return total.round(2),elbow,silhouette_fig
    
## 5-2. Performing k-means clustering on pca full  
## performing k-means clustering on df and returns labels you need to set the number of clusters
## as well as all the needed groupby dataframes
#########
# SET NUMBER OF CLUSTERS
n_clusters=5
#########

def cluster_predict(n,df_scaled, pca_df,df):
    cluster_names=["A","B","C","D","E"]
    cluster1="CLUSTERS"
    idx=df_scaled.index
    cols=list(df.columns)
    
    # removes cluster labels
    list_df=[df_scaled,pca_df,df]
    for element in list_df:
        try:
            del(element["CLUSTERS"])
        except:
            pass
    # fitting model with random_state
    model=cluster.KMeans(n_clusters=n,random_state=655) 
    
    # returning labels and model
    labels=list(model.fit_predict(pca_df))
    model.fit(pca_df)
    pickle.dump(model, open("../models/kmeans_pca_k5.sav", 'wb'))

    # Ranking labels according to count  
    unique_labels=list(Counter(labels))
   
    # returning df_scaled with cluster names     
    df_scaled[cluster1]=labels
    
    # replacing label values with ordered unique_labels
    df_scaled[cluster1]=df_scaled[cluster1].replace(unique_labels,cluster_names[:len(unique_labels)])    
    df[cluster1]=df_scaled[cluster1]
   
    # create an ordered group_by with natural mean for key dimensions and save date
    groupe=df.groupby("CLUSTERS").mean().round(2)
    groupe["count"]=df.groupby("CLUSTERS").size().round(2)
    # move count to the first place
    first_column = groupe.pop('count')
    groupe.insert(0, 'count', first_column)
    groupe.sort_values(by='count', ascending=False, inplace=True)
    
    save_df_as_image(groupe.iloc[:,:10], path="../visualizations/table_groupe_clusters_kmeans.png")
    groupe.to_csv("../results/pca_groupe_clusters.csv")
    
   
    # create group_by with mean for Z values of key dimensions
    groupe_z=df_scaled.groupby("CLUSTERS").mean().round(2)
    groupe_z["count"]=df_scaled.groupby("CLUSTERS").size().round(2)
    # move count to the first place
    first_column = groupe_z.pop('count')
    groupe_z.insert(0, 'count', first_column)
    groupe_z.sort_values(by='count', ascending=False, inplace=True)
    
    print("5.2- pca returning groupes and pca_transformed")

    return pca_df,df,groupe_z, groupe

    
def heat_map_kmeans(groupe_z,n_clusters, download):
    if download:
        try:
            heat_map_img = Image.open('../visualizations/kmeans_heat_map_pca_full.png')
            print("5.2- returning pca/kmeans full heat map")  
            return heat_map_img
        except:
            pass    
    
    try:
        del groupe_z["count"]
    except:
        pass
    
    fig = plt.figure()
    fig.set_size_inches(30, 2)
    
    ax2 = plt.subplot(121)
    ax2 = sns.heatmap(groupe_z,vmin=-.2, vmax=1, cmap=sns.diverging_palette(240, 20, n=9),center=0,annot=False,
                     linewidths=0.1, linecolor='gray')
    ax2.set_title("Cluster interpretation: dimensions heatmap",fontsize=14)
    ax2.set_xticklabels(list(groupe_z.columns), fontdict=None, minor=False,rotation=90)
    fig.savefig('../visualizations/kmeans_heat_map_pca_full.png', bbox_inches='tight')
    print("5.2- processing pca/kmeans full heat map")  
    
    return fig

def biplot_df(pca_components):
    # takes a pca components df and return a shortlist of components for visualization on the biplot
    liste=["transactions", "hits","newVisits","med_referral","direct_is_true","timeOnSite", "recency"]
    pca_components.sort_values("PC2")
    short_pca=pca_components.T
    short_pca=short_pca[liste].T
    short_pca["x"]=[0.04,1.07,0.025,0.01,0.045,0.085,0.04]
    short_pca["y"]=[-0.38,0.04,0.25,0.35,-0.175,-0.085,0.14]
    
    return short_pca
    

def biplot(pca_transformed,coeff,pca_cum, download):
    if download:
        try:
            biplot_img = Image.open('../visualizations/B2_clustering_and_biplot.png')
            print("5.2 downloading biplot clustering visualization")
            return biplot_img
        except:
            pass
      
    # visualizes biplot and clustering algorithm
    HUE='CLUSTERS'
    labels=(coeff.index)

    zoom = 1.2
    x=2
    y=3
    xs = pca_transformed.iloc[:,0]
    ys = pca_transformed.iloc[:,1]    
    width = 2.0 * zoom
    scalex = width/(xs.max()- xs.min())
    scaley = width/(ys.max()- ys.min())
    text_scale_factor = 1.15
        
    fig = plt.gcf()
    fig.set_size_inches(7, 7)
    
    # plt.scatter(xs*scalex, ys*scaley, s=0.05,alpha=0.5)
    ax1=sns.scatterplot(data=pca_transformed, x=xs*scalex, y=ys*scaley, hue=HUE, palette="deep",style=HUE)
    ax1.set_title("KMeans K=5 with PCA, explained variance= {}".format(round(pca_cum*100,1)), size=15)
    ax1.grid(True)
    ax1.legend(facecolor="white")
    # plt.grid(color='w', linestyle='-', linewidth=2)
    # ax1.legend(fontsize=14)
    arr=np.array(coeff)
    n=arr.shape[0]
    
    for i in range(n):
        
        plt.arrow(0, 0, arr[i,0], arr[i,1],color='black',alpha=0.7, head_width = 0.03) 
        if labels is None:
            plt.text(arr[i,x]* 1.5, 
                     arr[i,y] * text_scale_factor, 
                     "Var"+str(i+1), color='b', ha='left', va='center')
        else:
            plt.text(arr[i,x], 
                     arr[i,y], 
                     labels[i], color='black', backgroundcolor="white", fontsize=11,ha='left', va='top') #backgroundcolor="white",
            
    plt.grid(color='w', linestyle='-', linewidth=1)
    plt.xlim(-0.1,zoom)
    plt.ylim(-0.8,0.8)
    plt.xlabel("PC 1",fontsize=14)
    plt.ylabel("PC 2", fontsize=14)

    print("5.2 returning biplot clustering visualization")
    fig.savefig('../visualizations/B2_clustering_and_biplot.png', bbox_inches='tight')
    return fig


## 6- ASSESSING KMEDOIDS
def scatter2(data, titre,HUE="CLUSTERS"):
    
    cols=["PC1","PC2"]
    pcn=2
    
    fig = plt.figure()
    fig.set_size_inches(12, 5)

    ax1 = plt.subplot(121)
    if HUE:
        ax1=sns.scatterplot(data=data, x="PC1", y="PC2", hue=HUE, palette="deep",style=HUE)
    else:
        ax1=sns.scatterplot(data=data, x="PC1", y="PC2", palette="deep")
    ax1.set_xlabel("PC 1",fontsize=12)
    ax1.set_ylabel("PC 2",fontsize=12)
    ax1.set_title(label=titre,fontsize=14)
    ax1.set_facecolor('#EAEAF2')

    
    handles, labels = ax1.get_legend_handles_labels()
    
    if HUE:
        # sort both labels and handles by labels
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax1.legend(handles, labels,facecolor="white")
    
    return fig

def cluster_kmedoids(pca_transformed_full, download=True):
    filename = '../models/kmedoidsfinalized_model.sav'
    
    if download:
        try:
        # load the model from disk
            kmedoids = pickle.load(open(filename, 'rb'))
            # result = loaded_model.score(X_test, Y_test)
            kmedoids_df=pd.read_csv("../results/kmedoids_df.csv")
            kmedoids_df.rename(columns = {'Unnamed: 0':'idx'}, inplace = True)
            kmedoids_df.set_index('idx',inplace=True)
            img = Image.open('../visualizations/kmedoids.png')
            img.show()
            print("6. download kmedoids df, model and img")
            return kmedoids_df,kmedoids, img
        except:
            pass


    else:
    
        try:
            del pca_transformed_full["CLUSTERS"]
        except:
            pass
        # sparse_matrix = scipy.sparse.csr_matrix(dense_matrix)
        kmedoids_df=pca_transformed_full.sample(frac=0.50,random_state=32)
        kmedoids = KMedoids(n_clusters=5, random_state=655).fit(kmedoids_df)
        kmedoids_df["CLUSTERS"]=kmedoids.labels_
        cluster_names=["A","B","C","D","E","F","G"]
        unique_labels= list(Counter(kmedoids.labels_))
        kmedoids_df["CLUSTERS"]=kmedoids_df["CLUSTERS"].replace(unique_labels,cluster_names[:len(unique_labels)])

        kmedoids_df.to_csv("../results/kmedoids_df.csv")
        # save the model to disk

        pickle.dump(kmedoids, open(filename, 'wb'))
     
    fig=scatter2(kmedoids_df,"Customer Clustering with KMedoids K=5",HUE="CLUSTERS")
    fig.savefig('../visualizations/kmedoids.png', bbox_inches='tight')
   
    print("6. processing kmedoids df, model and img")
    return kmedoids_df,kmedoids, fig

def kmedoids_scores(kmedoids_df,kmedoids, download):
    if download:
        with open('../results/score_medoids.pickle','rb') as handle:
          score_medoids = pickles.load(handle)
        print("6. downloading kmedoids scores")
        return score_medoids
    else:
        
        labels= kmedoids.labels_
        score_medoids={}
        df=kmedoids_df.copy()
        del df["CLUSTERS"]
        score_medoids["silhouette"]=(round(silhouette_score(df,labels),2))
        score_medoids["bouldin"]=(round(davies_bouldin_score(df,labels),2))  

    y=list(score_medoids.values())
    x=list(score_medoids.keys())

    ax=sns.barplot(y=y,x=x, color="green",alpha=0.5)
    
    ax.set_title("Kmedoids Score for K= 5", size=15)
    
    ax.set_ylim(0, .75)
    for element in (ax.get_xticklabels() + ax.get_yticklabels()):   
        element.set_fontsize(13)
    
    ax.legend(facecolor="white")
    
    fig2 = ax.get_figure()
    fig2.set_size_inches(3,3)
    fig2.savefig('../visualizations/scores_medoids.png', bbox_inches='tight')
    print("6. processing kmedoids scores")
    
    with open('../results/score_medoids.pickle', 'wb') as handle:
        pickle.dump(score_medoids, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("6. processing kmedoids scores")
        
    return score_medoids
    
## 7- ASSESSING DBSCAN
# estimating eps
def dbscan_eps(df_scaled):

    sample=df_scaled.sample(frac=0.10,random_state=32)
    y=np.int64(sample["monetary"]>0)
    X=np.array(sample)

    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X, y)
    mole=neigh.kneighbors(X=X[1,:].reshape(1,45), n_neighbors=None, return_distance=True)
    cum=[]
    poulet=0
    for i in range(0,len(X)):    
        mole=neigh.kneighbors(X=X[i,:].reshape(1,45), n_neighbors=1, return_distance=True)
        cum.append(mole[0][0][0])
        poulet=poulet+mole[0][0][0]
    EPS1=np.median(cum)
    EPS=poulet/len(X)
    return EPS

# returning dbscan scores
def dbscan_scores(pca,df_scaled,download=True):
    eps_list = list(np.round(np.linspace(0.03, 0.36, 11, endpoint=False),3))
    if download:
        df_dbscan=pd.read_csv("../results/dbscan_scores.csv").iloc[:,1:]
        filename="../models/dbscan.sav"
        dbscan = pickle.load(open(filename, 'rb'))
        df_dbscan["eps"]=eps_list
        df_dbscan.set_index("eps", inplace=True)
        print("7. downloading dbscan scores")
        return df_dbscan
    try:
        del pca["CLUSTERS"]
    except:
        pass
    try:
        del df_scaled["CLUSTERS"]
    except:
        pass
    
    Sample_400=defaultdict(list)
    Sample_1000=defaultdict(list)
    df_dbscan=pd.DataFrame(columns=["Sample_400","silhouette_400","bouldin_400","Sample_1000",
                                   "silhouette_1000","bouldin_1000",], index=np.round(eps_list,3))
    min_samples=[400,1000]

    
    X=np.array(pca)
    
    for sample in min_samples:
        truc=[]
        machin=[]
        trucmuche=[]
        for eps in tqdm(eps_list):
            dbscan = cluster.DBSCAN(eps=eps, min_samples=sample)
            labels = dbscan.fit_predict(X)
            n=len(list(np.unique(labels)))
            truc.append(n)
            if n>1:
                machin.append(round(silhouette_score(df_scaled,labels),2))
                trucmuche.append(round(davies_bouldin_score(df_scaled,labels),2)) 
            else:
                machin.append(None)
                trucmuche.append(None)

        df_dbscan["Sample_"+str(sample)]=truc
        df_dbscan["silhouette_"+str(sample)]=machin
        df_dbscan["bouldin_"+str(sample)]=trucmuche
    df_dbscan.to_csv("../results/dbscan_scores.csv")

    print("7. processing dbscan scores")
    return df_dbscan



## returns dbscan clusters
def cluster_predict_db(pca,download):
    if download:
        try:         
            img = Image.open('../visualizations/DBScan.png')
            img.show()
            print("7. downloading dbscan model and visualization")
            model=pickle.load(model, open("../models/dbscan.sav", 'rb'))
            pca=pd.read_csv("../results/dbscan_df.csv",index_col=0)
            
            return model,img,pca
            
        except:
            pass
        
    cluster_names=["A","B","C","D","E"]
    cluster1="CLUSTERS"
    try:
        del pca["CLUSTERS"]
    except:
        pass   
    eps = list(np.round(np.linspace(0.03, 0.36, 11, endpoint=False),3))[4]       
    X=np.array(pca)
    dbscan = cluster.DBSCAN(eps=eps, min_samples=1000)
    model=dbscan.fit(X)
    labels = dbscan.fit_predict(X)
    counter=Counter(labels)
    unique_labels=list(np.unique(labels))
    n=len(unique_labels)
    pca[cluster1]=labels  
    
    pca["CLUSTERS"]=pca["CLUSTERS"].replace(unique_labels,cluster_names[:len(unique_labels)])
    
    # save csv file to disk
    pca.to_csv("../results/dbscan_df.csv")
        
    # save the model to disk
    pickle.dump(model, open("../models/dbscan.sav", 'wb'))
        
    fig=scatter2(pca,"Customer Clustering with DBScan",HUE="CLUSTERS")
    fig.savefig('../visualizations/DBScan.png', bbox_inches='tight')
    print("7. processing model, dbscan visualization and df")
    return model, fig,pca

## 8- RFM clustering
def loading_rfm():
    try:
        rfm = pd.read_csv("../datasets/model_files/B1_rfm_data.csv")
        print("8- importando rfm csv file")
    except:
        print("failed to import")
    if "class" in list(rfm.columns):
        del rfm["class"]
    if 'fullVisitorId' in rfm.columns:
        rfm.set_index('fullVisitorId',inplace=True)
    rfm_buyers=rfm[rfm["Monetary"]>0]
    
    try:
        del rfm_buyers["buyers"]
    except:
        pass
    
    num_cols=['Recency',"Monetary","Frequency"]
    
    rfm_scaled, rfm_scaler,index,cols=scale(rfm_buyers,num_cols,array=True) 
    
    return rfm_buyers,rfm_scaled

# EDA - visualizing key customers data in a violin plot
def plot_rfm(rfm):
    cols=list(rfm.columns)
    units=["days","$","times"]
    ax=["ax1","ax2","ax3"]
    liste_zip=zip(cols,units,ax)

    # Ploting the figure
    fig = plt.figure()
    fig.set_size_inches(14, 6)

    i=1
    for metrics,units,ax in liste_zip:
        ax = plt.subplot(1,4,i)
        ax = sns.violinplot(data=np.array(rfm[metrics]),palette="Set3")
        # ax = sns.violinplot(x=cols, y=metrics, data=Det1,palette="Set3")
        ax.set_ylabel('{} in {}'.format(metrics,units),fontsize=14)
        ax.set_title("{} By Cluster - {}".format(metrics,units),fontsize=14)
        i+=1
    print("8- plotting RFM clustering")
    plt.tight_layout()
    plt.show()


def cluster_score(df,download):
    if download:
        try:
            scores=pd.read_csv("../results/scores_kmeans_rfm_buyers.csv")
            print("8- downloading RFM groupe")
            return scores
        except:
            pass
        
    mod="kmeans"
    # uses cache, tries to download file
    end=7 # will iterate over 6 clusters
    try:
        scores=defaultdict(list)
        pass #scores=pd.read_csv("scores_{}.csv".format(mod),index_col="index")
        # return scores
    except:           
        scores=defaultdict(list)
        
    start=2
    
    try:
        del(df["CLUSTERS"])
    except:
        pass
    
    df=np.array(df)

    for n in range(start,end):
        if mod=="kmeans":            
            model=cluster.KMeans(n_clusters=n,random_state=655, max_iter=300)
        if mod=="dbscan":
            model=DBSCAN(eps=n, min_samples=2)            
        
        model.fit(df)      
        labels = model.labels_
        
        try:
            scores["inertia"].append(np.int64(model.inertia_))
        except:
            pass

        scores["silhouette"].append(round(silhouette_score(df,labels),2))
        scores["bouldin"].append(round(davies_bouldin_score(df,labels),2))  
        scores["calinski"].append(np.int64(calinski_harabasz_score(df,labels)))
        
    scores=pd.DataFrame(data=scores, index=range(start,end)) 
    
    # UPLOAD
    scores.to_csv("../results/scores_kmeans_rfm_buyers.csv")
    print("8- returning RFM scores")
    return scores

## performing k-means clustering on df and returns labels you need to set the number of clusters
## as well as all the needed groupby dataframes
n_clusters=4
def cluster_predict2(n, df_scaled, df,download):
    if download:
        try:
            groupe=pd.read_csv("../results/group_rfm_buyers.csv")
            return groupe
        except:
            pass
            
    cluster_names=["A","B","C","D","E","F","G"]
    cluster1="CLUSTERS"
    
    index=list(df.index)
    cols=list(df.columns)
    
    try:
        del(df["CLUSTERS"])
    except:
        pass
    try:
        del(df_scaled["CLUSTERS"])
    except:
        pass
    
    model=cluster.KMeans(n_clusters=n,random_state=655)
    labels=list(model.fit_predict(df_scaled))
    
    # returning df_scaled with cluster names
    try:        
        df_scaled[cluster1]=labels
    except:
        df_scaled=pd.DataFrame(data=df_scaled, columns=cols,index=index)
        df_scaled[cluster1]=labels
    unique_labels=list(np.unique(labels))
    
    df_scaled[cluster1]=df_scaled[cluster1].replace(unique_labels,cluster_names[:len(unique_labels)])
    
    df[cluster1]=df_scaled[cluster1]
    
    
    groupe=df.groupby("CLUSTERS").mean().round(2)
    groupe["count"]=df.groupby("CLUSTERS").size().round(2)
    
    groupe_z=df_scaled.groupby("CLUSTERS").mean().round(2)
    
    groupe.to_csv("../results/group_rfm_buyers.csv")
    print("8- returning RFM groupe")
    return groupe

if __name__=='__main__':
  
    download=True
    # 1- read file: takes "datasets/A1_B2_data.csv" and returns a dataframe
    df = read_file()
    # makes a copy of the file as original for future use
    df_original = df.copy()
    
    # 2- Rebalancing classes: takes df and returns a data frame with balanced class
    df_full = class_rebalance(df.copy())
    
    # 3- scaling the data sets. Takes: dataframe, numerical column lists.
    # Returns scaled df, scaler object, index and columns.
    num_cols=['pageviews','timeOnSite','browser_woe','country_woe','source_woe',
              'transactions','hits.eCom','hour_ordinal','recency','monetary'] 
    df_scaled, scaler, index, final_cols = scale(df_original,num_cols)       
    df_scaled_full, scaler_full, index_full, final_cols= scale(df_full,num_cols) 
    
    # 4.1- PCA transformation
    # takes: dataframe. Returns a- fit model object b- a series with cum explained scores c- a spree chart
    spree_chart,pca_cum=spree_PCA(df_scaled_full, download)
    
    # takes: data frame. 
    # Returns a- transformed df (90k rows) b- df with PCs (41 rows) c- heat_map d-pca models
    pca_transformed_full,pca_components_full,heat_map_fig_full,pca_model=pca_transfo_full(df_scaled_full, download)
    pca_transformed,pca_components,heat_map_fig,pca=pca_transfo(df_scaled, download)
    
    # 4.2- FAMD visualization and model
    # Takes only download argument. loads "datasets\A1_B2_data_dropped_features.csv"
    # returns the df with concatenated num and cat columns, tranformed df, models, and two visualizations
    df_cat_scaled,df_famd,famd_model,famd_channel_fig,famd_buyers_fig=famd(download)
    
    # 5.1- Iterates over different dataset and returns scores
    # takes several dfs and returns 4 df tables. Returns 4 respective table cores
    scores1,scores2,scores3,scores4=scores_kmeans(df_scaled,df_scaled_full,pca_components,pca_components_full,download=True)
    # takes score tables:
    # returns a- joint df with scores and b- elbow_charts c-silhouette fig
    joint_data,elbow,silhouette_fig=join_data_sets(scores1,scores2,scores3,scores4, download=True)

    # 5.2 Performing kmeans on pca balanced classes
    #########
    # SET NUMBER OF CLUSTERS
    n_clusters=5
    #########
    # takes 3 dfs, the number of clusters.Returns all dfs with cluster label
    pca_df_full,df_full,groupe_z_full, groupe_full=cluster_predict(n_clusters,df_scaled_full, pca_transformed_full,df_full)
    groupe_full
    
    # takes a df and returns a heatmap (png file) to interpret clusters
    heat_map_kmeans_fig=heat_map_kmeans(groupe_z_full,n_clusters, download)
    heat_map_kmeans_fig
    
    # takes: pc components. Returns: df with shortlist of PCs to plot
    short_pca=biplot_df(pca_components_full)
    
    # takes dfs, a series. Returns: a plot
    biplot_kmeans_fig=biplot(pca_transformed_full,short_pca,pca_cum[2], download=True)
    
    
    # 5.1- Iterates over different dataset and returns scores
    # takes several dfs and returns 4 df tables. Returns 4 respective table cores
    scores1,scores2,scores3,scores4=scores_kmeans(df_scaled,df_scaled_full,pca_components,pca_components_full,download=True)
    
    # takes score tables:
    # returns a- joint df with scores and b- elbow_charts c-silhouette fig
    joint_data,elbow,silhouette_fig=join_data_sets(scores1,scores2,scores3,scores4, download=True)
    silhouette_fig
    
    
    # 6- Performing Kmedoids
    # takes df and returns: a- transformed df b- model  c- clustering figure
    kmedoids_df,model_kmedoids,figmedoids=cluster_kmedoids(pca_transformed_full, download=True)
    # takes df and returns a dictionnary with two k:v
    score_medoids=kmedoids_scores(kmedoids_df,model_kmedoids, download=True)
    print(score_medoids)
    
    
    # 7- Performing dbscan
    # Takes: pca_transformed df and returns scores
    download=True
    # Takes: pca_transformed df and returns scores
    dbscan_scores_df=dbscan_scores(pca_transformed,df_scaled,download)  
    # takes: pca transfored and returns clustering figure
    model_dbscan,fig_DBSCAN,pca_dbscan=cluster_predict_db(pca_transformed,download)
    
    
    # 8- RFM clustering
    # takes: no argument. Ouput: loads "datasets/B1_rfm_data.csv"
    rfm_buyers,rfm_scaled=loading_rfm()
    # takes: dataframe. Output: violin plot
    # plot_rfm(rfm_buyers) 
    
    # takes: dataframe. Returns: data frame cluster scores
    scores_rfm=cluster_score(rfm_scaled, download)
    
    # takes dataframe. Returns: dataframe with grouped clusters
    n_clusters=4
    groupe_rfm=cluster_predict2(n_clusters,rfm_scaled, rfm_buyers,download)
    # groupe_rfm

    # liste of outputs
    liste_outputs=[
      spree_chart,
      pca_cum.head(),
      pca_components_full.head(),
      heat_map_fig_full,
      pca_model,
      famd_model,
      famd_channel_fig,
      famd_buyers_fig,
      joint_data.round(2),
      elbow,
      silhouette_fig,
      groupe_full,
      biplot_kmeans_fig,
      short_pca,
      # fig_pca,
      model_kmedoids,
      figmedoids,
      score_medoids,
      model_dbscan,
      fig_DBSCAN,
      pca_dbscan,
      scores_rfm,
      groupe_rfm
    ]

