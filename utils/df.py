from sklearn.preprocessing import MinMaxScaler

def scale_dataframe(df, columns_to_scale, fill_nan='0'):
    for col in columns_to_scale:
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[col].fillna(value=fill_nan).values)
    return df
