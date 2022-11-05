import json

import pandas as pd


def get_alternatives(name):
    df = pd.read_csv('data/sheets.csv', sep=';')
    df = df.drop(df.columns[0], axis=1)

    df: pd.DataFrame = df.iloc[0:3]
    if len(json.loads(get_info(name))) > 0:
        df['Co2Em_diff'] = df['Co2Em'] - json.loads(get_info(name))[0]['Co2Em']
    return df.to_json(orient='records')


def get_info(name):
    df = pd.read_csv('data/sheets.csv', sep=';')

    df = df[df['Name'] == name]

    return df.to_json(orient='records')


print(get_alternatives('Kuumavalssattu 1D RST levy 12.0x1500x3000mm'))

