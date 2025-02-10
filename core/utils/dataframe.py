import pandas as pd

def ReadDf(csv_file):
    df = pd.read_csv(csv_file)    
#    df.rename(columns={
#        'timestamp': 'DATETIME',
#        'open': 'OPEN',
#        'high': 'HIGH',
#        'low': 'LOW',
#        'close': 'CLOSE',
#        'volume': 'VOLUME'
#    }, inplace=True)

    columns_to_keep = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = df[columns_to_keep]
    
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df