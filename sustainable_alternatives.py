import json

import pandas as pd
import numpy as np
import math
from sentence_transformers import SentenceTransformer


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def get_sheet_data():
    return pd.read_csv("data/sheets.csv", sep=";")


def get_material_data():
    material_data = pd.read_csv("data/material.csv", sep=";")[["Material", "Embodied Energy", "Carbon Footprint", "Water Usage", "Pro 1", "Pro 2", "Con 1","Con 2", "kg/pc"]]
    material_data.at[6, "Material"] = "Galvanized steel"

    material_data['kg/pc'] = pd.to_numeric(material_data['kg/pc'].map(lambda x: x.replace(",", ".")))
    material_data['Embodied Energy'] = pd.to_numeric(material_data['Embodied Energy'].map(lambda x: x.replace(",", ".")))
    material_data['Carbon Footprint'] = pd.to_numeric(material_data['Carbon Footprint'].map(lambda x: x.replace(",", ".")))
    material_data['Water Usage'] = pd.to_numeric(material_data['Water Usage'].map(lambda x: x.replace(",", ".")))
        
    material_data["n_home"]    = (material_data['Embodied Energy'] * material_data['kg/pc'] * 0.278) / 1500
    material_data["n_trees"]   = (material_data['Carbon Footprint'] * material_data['kg/pc']) / 25.
    material_data["n_bottles"] = (material_data['Water Usage'] * material_data['kg/pc'])
    
    Embodied_Energy = material_data['Embodied Energy']
    Carbon_Footprint = material_data['Carbon Footprint']
    Water_Usage = material_data['Water Usage']
    Embodied_Energy = (Embodied_Energy - Embodied_Energy.min()) / (Embodied_Energy.max() - Embodied_Energy.min())
    Carbon_Footprint = (Carbon_Footprint - Carbon_Footprint.min()) / (Carbon_Footprint.max() - Embodied_Energy.min())
    Water_Usage = (Water_Usage - Water_Usage.min()) / (Water_Usage.max() - Water_Usage.min())
    material_data['Score'] = 1 - (Embodied_Energy + Carbon_Footprint + Water_Usage) / 3
    return material_data

def get_product_code_2_description(sheets_data):
    important_columns = ["Code", "Class", "Name", "Weight", "WeightUnit", "Length", "Height", "Width", "Thickness", "Diameter", "ShortDescription"]
    sheets_data_filetred = sheets_data[important_columns]
    sheets_data_filetred['ShortDescription'] = sheets_data_filetred['ShortDescription'].apply(lambda x: x.replace("\r\n\r\n", ", "))
    
    product_code_2_description = {}
    for sheet in sheets_data_filetred.iterrows():
        sheet = sheet[1]
        sheet_str = "; ".join([f"{col} - {val}" for col, val in sheet.items()])
        product_code_2_description[sheet.Code] = sheet_str
    return product_code_2_description
    

def get_alternatives(product_code, top_k=3):    
    sheets_data = get_sheet_data()
    product_code_2_description = get_product_code_2_description(sheets_data)
    material_data = get_material_data()
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    src_desc = product_code_2_description[product_code]
    
    src_material = sheets_data[sheets_data["Code"] == product_code].Material.to_list()[0]
    df = pd.DataFrame(columns=["Code", "cos", "eff_score", "material"]) 
    for code in product_code_2_description:
        cur_material = sheets_data[sheets_data["Code"] == code].Material.to_list()[0]
        if cur_material not in [src_material, "Non-rusting steel", ]:
            product_desc = product_code_2_description[code]
            sentences = [src_desc, product_desc]
            embeddings = model.encode(sentences)
            cos_sim = cosine_similarity(*embeddings)
            eff_score = material_data[material_data["Material"] == cur_material]["Score"].to_list()[0]
            df.loc[len(df)]=[code, cos_sim, eff_score, cur_material]
    df = df.sort_values(by=["cos"], ascending=False)[1:]

    df_with_best_materials = pd.DataFrame(columns=["Code", "cos", "eff_score", "material"])
    for material in set(df.material):
        sub_df = df[df.material == material]
        best_sample = sub_df[sub_df.cos == sub_df.cos.max()].values[0]
        df_with_best_materials.loc[len(df_with_best_materials)] = list(best_sample)

    df_with_best_materials = df_with_best_materials.sort_values(by=["cos", "eff_score"], ascending=False)
    df_top_k = df_with_best_materials[:top_k]
    df2 = pd.DataFrame({'Code': product_code, 'cos': 1, 'eff_score': material_data[material_data["Material"] == src_material]["Score"].to_list()[0], "material": src_material}, index=[0])
    df_top_k = df2.append(df_top_k,ignore_index = True)
    
    images = [sheets_data[sheets_data["Code"] == c].SmallImageURL.to_list()[0] for c in df_top_k.Code]
    urls = [f"https://www.cronvall.fi/epages/CronvallShop.sf/sec89130f6a70/?ObjectPath=/Shops/CronvallShop/Products/{c}" for c in df_top_k.Code]
    prices = [sheets_data[sheets_data["Code"] == c].Price.to_list()[0].split(".")[0] for c in df_top_k.Code]
    names = [sheets_data[sheets_data["Code"] == c].Name.to_list()[0] for c in df_top_k.Code]
    
    trees_differs =  [material_data[material_data["Material"] == m]["n_trees"].to_list()[0]   for m in df_top_k.material]
    battle_differs = [material_data[material_data["Material"] == m]["n_bottles"].to_list()[0] for m in df_top_k.material]
    energy_differs = [material_data[material_data["Material"] == m]["n_home"].to_list()[0]    for m in df_top_k.material]
    
    trees_differs =  [i - trees_differs[0]  for i in trees_differs] 
    battle_differs = [i - battle_differs[0] for i in battle_differs]
    energy_differs = [i - energy_differs[0] for i in energy_differs]
    
    co2eff = [material_data[material_data["Material"] == m]["Carbon Footprint"].to_list()[0] for m in df_top_k.material]
    pro_1 = [material_data[material_data["Material"] == m]["Pro 1"].to_list()[0] for m in df_top_k.material]
    pro_2 = [material_data[material_data["Material"] == m]["Pro 2"].to_list()[0] for m in df_top_k.material]
    con_1 = [material_data[material_data["Material"] == m]["Con 1"].to_list()[0] for m in df_top_k.material]
    con_2 = [material_data[material_data["Material"] == m]["Con 2"].to_list()[0] for m in df_top_k.material]
    water = [material_data[material_data["Material"] == m]["Water Usage"].to_list()[0] for m in df_top_k.material]

    df_top_k["Images"] = images
    df_top_k["Urls"] = urls
    df_top_k["Prices"] = prices
    df_top_k["Trees"] = trees_differs
    df_top_k["Bottle"] = battle_differs
    df_top_k["Water"] = water
    df_top_k["Energy"] = energy_differs
    df_top_k["Names"] = names
    df_top_k["CO2eff"] = co2eff
    df_top_k["avg_eff_score"] = [material_data.Score.mean(), ] * len(co2eff)
    df_top_k["pro_1"] = pro_1
    df_top_k["pro_2"] = pro_2
    df_top_k["con_1"] = con_1
    df_top_k["con_2"] = con_2
    df_top_k["Water"] = water
    return df_top_k.to_json(orient='records')


def get_info(id):
    df = pd.read_csv('data/sheets.csv', sep=';')
    df = df[df['Code'] == id]

    return df.to_json(orient='records')


