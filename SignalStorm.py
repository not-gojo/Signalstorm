import pandas as pd
import numpy as np
import requests
import time
import tweepy
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def get_binance_data(symbol='DOGEUSDT', interval='5m', limit=100):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url).json()
    
    data = []
    for candle in response:
        data.append([
            int(candle[0]), float(candle[1]), float(candle[2]), float(candle[3]),
            float(candle[4]), float(candle[5])
        ])
    
    df = pd.DataFrame(data, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Time'] = pd.to_datetime(df['Time'], unit='ms')
    return df

def calculate_indicators(df):
    # Moving Averages
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'] = df['EMA_9'] - df['EMA_21']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['BB_upper'] = df['SMA_20'] + (df['Close'].rolling(window=20).std() * 2)
    df['BB_lower'] = df['SMA_20'] - (df['Close'].rolling(window=20).std() * 2)
    
    return df

def prepare_ml_data(df):
    df = df.dropna()
    features = ['EMA_9', 'EMA_21', 'RSI', 'MACD', 'Signal_Line', 'BB_upper', 'BB_lower']
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)  # 1 if price goes up, else 0
    
    X = df[features]
    y = df['Target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, df

def train_ml_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def set_dynamic_target_stop_loss(df):
    df['Signal_Strength'] = (df['MACD'] - df['Signal_Line']).abs() + df['RSI'].diff().abs()
    df['Signal_Strength'] = (df['Signal_Strength'] - df['Signal_Strength'].min()) / (df['Signal_Strength'].max() - df['Signal_Strength'].min())
    
    df['Target'] = df['Close'] * (1 + 0.01 + df['Signal_Strength'] * 0.02)  # Dynamic target
    df['Stop_Loss'] = df['Close'] * (1 - 0.005 - df['Signal_Strength'] * 0.01)  # Dynamic stop loss
    
    return df

def generate_trade_signal(df):
    latest = df.iloc[-1]
    signal = "BUY" if latest['Target'] > latest['Close'] else "SELL"
    message = f"Trade Alert: {signal} DOGE at {latest['Close']:.4f} USD\nTarget: {latest['Target']:.4f} USD\nStop Loss: {latest['Stop_Loss']:.4f} USD\nSignal Strength: {latest['Signal_Strength']:.2f}"
    return message

def post_to_twitter(message):
    # Twitter API keys (Replace with actual keys)
    api_key = "YOUR_API_KEY"
    api_secret = "YOUR_API_SECRET"
    access_token = "YOUR_ACCESS_TOKEN"
    access_secret = "YOUR_ACCESS_SECRET"
    
    auth = tweepy.OAuthHandler(api_key, api_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth)
    
    try:
        api.update_status(message)
        print("Tweet posted successfully!")
    except Exception as e:
        print(f"Error posting tweet: {e}")

def main():
    symbol = 'DOGEUSDT'
    df = get_binance_data(symbol)
    df = calculate_indicators(df)
    X, y, df = prepare_ml_data(df)
    model = train_ml_model(X, y)
    df = set_dynamic_target_stop_loss(df)
    message = generate_trade_signal(df)
    print(message)
    post_to_twitter(message)
    
if __name__ == "__main__":
    main()