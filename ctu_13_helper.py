import pandas as pd
import os


def replace_label(dataframe):
    botnet = []
    botnet_ip = '147.32.84.165'
    for i in dataframe.index:
        if (dataframe['SrcAddr'][i] == botnet_ip or dataframe['DstAddr'][i] == botnet_ip):
            botnet.append(1)
        else:
            botnet.append(0)
    dataframe['Botnet'] = botnet
    # with open("ctu_13_labeled_output.csv", "w") as text_file:
    #     text_file.write(dataframe.to_csv(index=False))

def prepare_ctu_13():
    path = "../CTU-13-Dataset"
    full_dataframe = pd.DataFrame()
    for (root, dirs, files) in os.walk(path, topdown=True):
        for file in files:
            if file.endswith(".binetflow"):
                #print(root+"/"+file)       
                file_url = f"{root}/{file}"     
                partial_dataframe = pd.read_csv(file_url)
                del partial_dataframe['Label']
                replace_label(partial_dataframe)
                full_dataframe = full_dataframe.append(partial_dataframe)
    with open("ctu_13_labeled_output.csv", "w") as text_file:
        text_file.write(full_dataframe.to_csv(index=False))

#Preparing the data

#prepare_ctu_13()

file_url = "../CTU-13-Dataset/1/capture20110810.binetflow"
dataframe = pd.read_csv(file_url)
del dataframe['Label']

replace_label(dataframe)


print(dataframe.shape)

print(dataframe.head())
