import pandas as pd
import os


def replace_label(dataframe):
    botnet = []
    botnet_ip = '147.32.84.165'
    for i in dataframe.index:
        if (dataframe['Src IP'][i] == botnet_ip or dataframe['Dst IP'][i] == botnet_ip):
            botnet.append(1)
        else:
            botnet.append(0)
    dataframe['Botnet'] = botnet
   

def prepare_ctu_13():
    path = "../CTU-13-Dataset"
    full_dataframe = pd.DataFrame()
    for (root, dirs, files) in os.walk(path, topdown=True):
        for file in files:
            if file.endswith(".csv"):
                #print(root+"/"+file)       
                file_url = f"{root}/{file}"     
                partial_dataframe = pd.read_csv(file_url)
                replace_label(partial_dataframe)
                del partial_dataframe['Label']
                del partial_dataframe['Flow ID']
                del partial_dataframe['Src IP']
                del partial_dataframe['Src Port']
                del partial_dataframe['Dst IP']
                del partial_dataframe['Dst Port']
                del partial_dataframe['Timestamp']
                full_dataframe = full_dataframe.append(partial_dataframe)
    with open("ctu_13_labeled_output.csv", "w") as text_file:
        text_file.write(full_dataframe.to_csv(index=False))

#Preparing the data
prepare_ctu_13()

