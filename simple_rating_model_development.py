import pandas as pd
import sqlite3
conn = sqlite3.connect('company_balancesheet_database.db')
df = pd.read_sql("""
 SELECT *
    FROM balancesheet
""", conn)

df.drop('id',inplace=True,axis=1)


# Renaming the columns for better understaning...
df.columns = ['NIF', 'Company Name', 'CNAE_Sector',
              'TotalAssets_2017', 'TotalAssets_2016',
              'TotalAssets_2015', 'OwnResources_2017', 'OwnResources_2016',
              'OwnResources_2015', 'ShortTermDebt_2017',
              'ShortTermDebt_2016', 'ShortTermDebt_2015',
              'LongTermDebt_2017', 'LongTermDebt_2016',
              'LongTermDebt_2015', 'Income_2017',
              'Income_2016', 'Income_2015',
              'Amortization_2017', 'Amortization_2016',
              'Amortization_2015', 'Profit_2017', 'Profit_2016',
              'Profit_2015', 'Status']	   
    

df['target_status'] = [0 if i in ['Activa', ''] else 1 for i in df['Status']] # 0 si Activa, 1 si algo raro!


# Ebita Margin - Ebitda / Turn over (Income - Ventas)
df['ebitda_income'] = (df["Profit_2016"])/(df["Income_2016"]) 

# Total Debt / Ebita 
df['debt_ebitda'] =(df["ShortTermDebt_2016"] + df["LongTermDebt_2016"]) /(df["Profit_2016"] + abs(df["Amortization_2016"]))

# rraa_rrpp: Financial leveraging / apalancamiento financiero 
df['rraa_rrpp'] = (df["TotalAssets_2016"] - df["OwnResources_2016"] ) /df["OwnResources_2016"]

# Log of Operating Income
import numpy as np
df['log_operating_income'] = np.log(df["Income_2016"])


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=1234)

df_clean = df[['ebitda_income','debt_ebitda','rraa_rrpp','log_operating_income','target_status']].replace([np.inf, -np.inf], np.nan).dropna()
X = df_clean[['ebitda_income','debt_ebitda','rraa_rrpp','log_operating_income']]
y = df_clean['target_status']

fitted_model = model.fit(X, y)
y_pred = fitted_model.predict(X)
y_pred_proba = fitted_model.predict_proba(X)[:,1]

print ("ASSESSING THE MODEL...")
# CALCULATING GINI PERFORMANCE ON DEVELOPMENT SAMPLE
from sklearn.metrics import roc_auc_score
gini_score = 2*roc_auc_score(y, y_pred_proba)-1
print ("GINI DEVELOPMENT=", gini_score)

from sklearn.metrics import accuracy_score
print("Accuracy: {0}".format(accuracy_score(y_pred,y)))

print ("SAVING THE PERSISTENT MODEL...")
from joblib import dump#, load
dump(fitted_model, 'Rating_RandomForestClassifier.joblib') 


#
#i=0
#time_in_datetime = datetime.strptime(df.fecha_cambio_estado.iloc[i], "%Y-%m-%d)
#

    