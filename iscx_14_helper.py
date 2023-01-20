import pandas as pd
import os


def replace_label(dataframe):
    botnet = []
    botnet_ip = [
'192.168.2.112',
'131.202.243.84',
'192.168.5.122',
'198.164.30.2',
'192.168.2.110',
'192.168.4.118',
'192.168.2.113',
'192.168.1.103',
'192.168.4.120',
'192.168.2.112',
'192.168.2.109',
'192.168.2.105',
'147.32.84.180',
'147.32.84.170',
'147.32.84.150',
'147.32.84.140',
'147.32.84.130',
'147.32.84.160',
'10.0.2.15',
'192.168.106.141',
'192.168.106.131',
'172.16.253.130',
'172.16.253.131',
'172.16.253.129',
'172.16.253.240',
'74.78.117.238',
'158.65.110.24',
'192.168.3.35',
'192.168.3.25',
'192.168.3.65',
'172.29.0.116',
'172.29.0.109',
'172.16.253.132',
'192.168.248.165',
'10.37.130.4',
]
    for i in dataframe.index:
        if (dataframe['Src IP'][i] in botnet_ip or dataframe['Dst IP'][i] in botnet_ip):
            botnet.append(1)
        else:
            botnet.append(0)
    dataframe['Botnet'] = botnet
   

def prepare_iscx_2014():
    path = "../ISCX-Bot-2014"
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
    with open("iscx_2014_labeled_output.csv", "w") as text_file:
        text_file.write(full_dataframe.to_csv(index=False))

#Preparing the data
prepare_iscx_2014()

