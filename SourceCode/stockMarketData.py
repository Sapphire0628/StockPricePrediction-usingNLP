import pandas as pd
import yfinance as yf

from datetime import datetime
import yfinance as yf
yf.pdr_override() 
from pandas_datareader import data as pdr

df = pd.read_excel("Startup_Listed_In_NASDAD_2000-2020.xlsx",sheet_name='Sheet1')


stockMarketData = pd.DataFrame()






#%%
for i in range(len(df)):
    Symbols = df.loc[i, "Symbols"]
    stockData = pdr.get_data_yahoo(Symbols, start=datetime(2000, 12, 1), end=datetime(2020, 1, 3))
    stockMarketData[Symbols] =stockData['Close']

    
#%%

stockMarketData.to_excel("stockMarketData.xlsx") 


#%%
stockMarketData.to_json("stockMarketData.json")
#%%
df =['VOR', 'VIR', 'AGTC', 'TCBP', 'IKT', 'KOD']
for i in df:
    Symbols = i
    stockData = pdr.get_data_yahoo(Symbols, start=datetime(2000, 12, 1), end=datetime(2000, 1, 3))
    stockMarketData[Symbols] =stockData['Close']