import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm

tickers = ["BZ=F","EURUSD=X"]
montant= 1200

def fetch_yf():
    df_data = yf.download(tickers, start="2020-01-02", end="2026-08-07", interval='1d')
    df_brent=df_data["Close", "BZ=F"]
    df_eurusd=df_data["Close","EURUSD=X"]
    return df_brent, df_eurusd

def fetch_dgec():
    df_dgec_TTC=pd.read_csv('DGEC.csv', sep=";", decimal="," )
    df_dgec = df_dgec_TTC[['Date','Gazole']]
    df_dgec['Date']=pd.to_datetime(df_dgec['Date'] , dayfirst=True)
    df_dgec=df_dgec.set_index('Date')
    df_dgec['Gazole']=df_dgec['Gazole']/1.2
    return df_dgec
    
df_brent, df_eurusd = fetch_yf()
df_dgec = fetch_dgec()

def convert_brent_to_euros(df_brent, df_eurusd): 
    df_brent_eur=df_brent/df_eurusd
    df_brent_mean=df_brent_eur.resample('SME').mean()
    df_brent_mean=df_brent_mean.rename("Brent")
    return df_brent_mean

df_brent_mean= convert_brent_to_euros(df_brent, df_eurusd)

def crn_conv(df_dgec):
    df_dgec_mean=df_dgec.resample('SME').mean()
    ref=df_dgec_mean.iloc[0]
    df_crn=(df_dgec_mean.shift(1)/ref)*100
    df_crn=df_crn.dropna()
    return df_crn

df_crn=crn_conv(df_dgec)

def merge_ts(df_crn, df_brent_mean):
    df_merge = pd.merge(df_crn, df_brent_mean,on="Date")
    return df_merge

df_merge= merge_ts(df_crn, df_brent_mean)

def calc_returns(df_merge): 
    returns_merge = df_merge.pct_change().dropna()
    return returns_merge

returns_merge=calc_returns(df_merge)

def calculate_ratio(returns_merge):
    x=returns_merge["Brent"]
    y= returns_merge["Gazole"]
    x = sm.add_constant(x)
    results = sm.OLS(y, x).fit()
    print(results.summary())
    beta = results.params["Brent"]
    alpha=results.params["const"]
    print(f"alpha: {alpha:.2f}, beta:{beta:.2f}")


"""resultats cata sans ajustement temporel (shift()) le R² est proche de 0..."""


def calculate_ratio_adjust(returns_merge): 
    best_r=0
    r=0
    best_shift=0
    for i in range(10):
        test=returns_merge.copy()
        test["Brent"]=test["Brent"].shift(i)     
        test=test.replace([-np.inf, np.inf],np.nan).dropna()       
        x=test["Brent"]
        y= test["Gazole"]
        x = sm.add_constant(x)
        results = sm.OLS(y, x).fit()
        r=results.rsquared
        print(f" i : {i}, r²:{r}")            
        beta = results.params["Brent"]
        alpha=results.params["const"]
        print(f"alpha: {alpha:.2f}, beta:{beta:.2f}")
        if r>best_r: 
            best_r=r
            best_shift=i
        print(f"meilleur r² : {best_r} à {best_shift}")

calculate_ratio_adjust(returns_merge)



"""class Contrat : 
    def __init__(self,volume):  
        self.volume  

"""
#simulate_unhedged_cost(df, volume_litres, base_price): 
#simulate_forward_hedge(df, volume_litres, beta):
#calculate_margin_calls(df_forward_pnl):

""" 
compute_kpis(unhedged_series, hedged_series, margin_series)
Mission : Retourner un dictionnaire avec trois métriques clés :
La variance (ou volatilité) du budget mensuel (elle doit être proche de zéro avec le hedge).
Le coût total annuel dans les deux scénarios.
Le besoin en fonds de roulement maximal (le pire appel de marge subi).
plot_dashboard(...)
Mission : Générer un graphique propre (via Matplotlib ou Seaborn) montrant les deux courbes (Avec / Sans Couverture) superposées.
"""
