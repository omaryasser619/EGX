import os
# Force CPU Mode to prevent hardware crashes
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import yfinance as yf
import pandas as pd
import numpy as np
import json
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

OUTPUT_FILE = "ml_predictions.json"

EGX100_TICKERS = [
    "ABUK.CA","ADIB.CA","ATLC.CA","ALCN.CA","SVCE.CA","AMER.CA","ACGC.CA",
    "ARAB.CA","AMIA.CA","RREI.CA","ARCC.CA","ASCM.CA","ASPI.CA","BINV.CA",
    "BTFH.CA","CIRA.CA","COSG.CA","POUL.CA","CSAG.CA","PRCL.CA","CLHO.CA",
    "COMI.CA","CNFN.CA","CIEB.CA","SUGR.CA","DSCW.CA","EFID.CA","HRHO.CA",
    "EFIH.CA","EGAL.CA","EGCH.CA","EGTS.CA","PHAR.CA","MPRC.CA","ETRS.CA",
    "EHDR.CA","ECAP.CA","ELKA.CA","KABO.CA","OBRI.CA","ELSH.CA","ELEC.CA",
    "UEGC.CA","SWDY.CA","EMFD.CA","ENGC.CA","EXPA.CA","FWRY.CA","GBCO.CA",
    "GGCC.CA","HELI.CA","HDBK.CA","ISPH.CA","IEEC.CA","IFAP.CA","ICFC.CA",
    "ISMQ.CA","ISMA.CA","JUFO.CA","LCSW.CA","MCRO.CA","MASR.CA","MPCO.CA",
    "MOIL.CA","MEPA.CA","MPCI.CA","MENA.CA","MCQE.CA","MFPC.CA","ATQA.CA",
    "MTIE.CA","EGAS.CA","OFH.CA","OLFI.CA","ODIN.CA","ORAS.CA","ORHD.CA",
    "ORWE.CA","PHDC.CA","PRDC.CA","CCAP.CA","RAYA.CA","SKPC.CA","SCEM.CA",
    "OCDI.CA","TMGH.CA","ETEL.CA","RMDA.CA","CERA.CA","ADPC.CA","UNIT.CA",
    "ZMID.CA", "CPCI.CA"
]

def safe_atr(df, window=14):
    if len(df) < window: return pd.Series(np.nan, index=df.index)
    atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Adj Close'], window=window)
    return atr.average_true_range()

def download_and_clean(ticker, period="2y"):
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.sort_index().copy()
    df['Adj Close'] = df['Close']
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    return df

def add_features(df):
    df['RSI_14'] = RSIIndicator(close=df['Adj Close'], window=14).rsi()
    macd = MACD(close=df['Adj Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    df['EMA_20'] = EMAIndicator(close=df['Adj Close'], window=20).ema_indicator()
    df['ATR_14'] = safe_atr(df, window=14)
    
    df['return_1d'] = df['Adj Close'].pct_change(1, fill_method=None)
    df['return_3d'] = df['Adj Close'].pct_change(3, fill_method=None)
    df['return_7d'] = df['Adj Close'].pct_change(7, fill_method=None)
    df['vol_change'] = df['Volume'].pct_change()
    df['vol_ma20'] = df['Volume'].rolling(20).mean()
    
    df['next_day_price'] = df['Adj Close'].shift(-1)
    df['target_return_1d'] = (df['next_day_price'] - df['Adj Close']) / df['Adj Close']
    
    df['target_class'] = 1 
    df.loc[df['target_return_1d'] > 0.01, 'target_class'] = 2  
    df.loc[df['target_return_1d'] < -0.01, 'target_class'] = 0 
    
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

print("Downloading EGX data for Deep Learning...")
all_train_data = []
prediction_data = {} 

features = ['RSI_14', 'MACD', 'MACD_signal', 'MACD_diff', 'EMA_20', 'ATR_14', 
            'return_1d', 'return_3d', 'return_7d', 'vol_change', 'vol_ma20']

for ticker in EGX100_TICKERS:
    df = download_and_clean(ticker)
    if df is not None and len(df) > 30:
        df = add_features(df)
        train_df = df.iloc[:-1].dropna(subset=features + ['target_class', 'next_day_price'])
        train_df['Ticker'] = ticker
        all_train_data.append(train_df)
        
        today_row = df.iloc[[-1]][features].copy()
        if not today_row.isna().values.any():
            prediction_data[ticker] = today_row

final_train_df = pd.concat(all_train_data)

X_train_raw = final_train_df[features].values
y_class_train = final_train_df['target_class'].values
y_price_train = final_train_df['next_day_price'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)

print(f"Training Deep Neural Networks on {len(X_train_scaled)} historical days...")

class_nn = Sequential([
    Input(shape=(len(features),)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])
class_nn.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
class_nn.fit(X_train_scaled, y_class_train, epochs=20, batch_size=32, verbose=0)

price_nn = Sequential([
    Input(shape=(len(features),)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
price_nn.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
price_nn.fit(X_train_scaled, y_price_train, epochs=20, batch_size=32, verbose=0)

print("Generating tomorrow's predictions via Neural Network...")
output_json = {}
signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}

for ticker, today_features in prediction_data.items():
    today_scaled = scaler.transform(today_features.values)
    
    pred_class_probs = class_nn.predict(today_scaled, verbose=0)
    pred_class_num = np.argmax(pred_class_probs[0])
    pred_price = price_nn.predict(today_scaled, verbose=0)[0][0]
    
    output_json[ticker] = {
        "ml_signal": signal_map[int(pred_class_num)],
        "ml_predicted_price": round(float(pred_price), 2)
    }

with open(OUTPUT_FILE, "w") as f:
    json.dump(output_json, f, indent=4)

print(f"✅ Deep Learning JSON successfully generated at {OUTPUT_FILE}!")
