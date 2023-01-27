import pandas as pd
import os


def replace_label(dataframe):
    botnet = []
    botnet_ip = [
        '172.16.2.11',
        '172.16.0.2',
        '172.16.0.11',
        '172.16.0.12',
        '172.16.2.12'
    ]
    for i in dataframe.index:
        if (dataframe['Src IP'][i] in botnet_ip or dataframe['Dst IP'][i] in botnet_ip):
            botnet.append(1)
        else:
            botnet.append(0)
    dataframe['Botnet'] = botnet
   

def prepare_isot_2010():
    path = "../ISOT_Botnet_DataSet_2010"
    full_dataframe = pd.DataFrame()
    for (root, dirs, files) in os.walk(path, topdown=True):
        for file in files:
            if file.endswith("pcap_Flow.csv"):
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
    with open("isot_2010_labeled_output.csv", "w") as text_file:
        text_file.write(full_dataframe.to_csv(index=False))

#Preparing the data
prepare_isot_2010()

