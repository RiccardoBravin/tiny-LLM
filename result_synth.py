#code to synthesize the results of the experiments

import os
import pathlib
import re
from scipy.stats import t
import numpy as np

def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(n)  # Standard error of the mean
    h = sem * t.ppf((1 + confidence) / 2., n-1)
    return mean, h

models = ["mlm_BERT_original_43", "mlm_Mamba_model_noNANO_30", "mlm_NanoBERT_original_29", "mlm_BERT_Efficient_50", "Embedder_only", "Embedder_+_conv", "mlm_Nano_Bert_Efficient_27", "mlm_Nano_Bert_Differential_Skip_49" ]
# models = ["BERT_efficient", "mlm_BERT_Efficient_50","MAMBA_noNANO", "mlm_Mamba_model_noNANO_30"]

if True:
    for model in models:
        print(f"Results for model: {model}")

        # For each file inside the results/model directory
        for file in os.listdir(f"results/{model}"):
            #keep from the file name only what is after cls_ and before _classification...
            #file_name = file.split("_")[0]
            try:
                file_name = re.split(r"^([\w-]+_\d+)", file)[1]
            except:
                continue
            print(f"\t\t{file_name}")
            print(f"{'acc':<15}{'f1':<15}{'mcc':<15}")

            #for each file extract the float values after the keywords "f1",and "accuracy"
            with open(f"results/{model}/{file}", "r") as f:
                lines = f.readlines()
                f1s = []
                accuracies = []
                mccs = []
                sccs = []
                for line in lines:
                    if "F1" in line:
                        f1 = float(line.split(":")[1])
                        # print("F1:",f1)
                        f1s.append(f1)
                    if "Accuracy" in line:
                        accuracy = float(line.split(":")[1])
                        # print("Acc:", accuracy)
                        accuracies.append(accuracy)
                    if "MCC" in line:
                        mcc = float(line.split(":")[1])
                        # print("MCC:", mcc)
                        mccs.append(mcc)
                    if "Spearman correlation coefficient" in line:
                        scc = float(line.split(":")[1])
                        # print("SCC:", scc)
                        sccs.append(scc)

                try:
                    acc_mean, acc_conf = confidence_interval(accuracies)
                    f1_mean, f1_conf = confidence_interval(f1s)
                    mcc_mean, mcc_conf = confidence_interval(mccs)
                    print(f"{acc_mean*100:<15.2f}\t{f1_mean*100:<15.2f}\t{mcc_mean*100:<15.2f}".replace('.', ','))
                    print(f"{acc_conf*100:.2f}\t\t\t{f1_conf*100:.2f}\t\t\t{mcc_conf*100:.2f}".replace('.', ','))
                except:
                    try:
                        scc_mean, scc_conf = confidence_interval(sccs)
                        print(f"{scc_mean:<15.2f}".replace('.', ','))
                        print(f"{scc_conf:.2f}".replace('.', ','))
                    except:
                        print("No results")

        print("\n*************************************************\n")
else:
    import os
    import re
    # Cartella che contiene i file
    cartella = 'results/BERT-tiny'
    
    
    # Funzione per estrarre i dati e calcolare la media di accuracy, MCC e F1
    def estrai_metriche_con_parsing_custom(file_path):
        accuracy_list = []
        mcc_list = []
        f1_list = []
        
        with open(file_path, 'r') as f:
            contenuto = f.read()
            
            # Trova tutti i blocchi di dati che contengono i valori
            blocchi = re.findall(r"\{.*?\}", contenuto, re.DOTALL)
            
            # Per ogni blocco, estrai i valori di accuracy, mcc e f1
            for blocco in blocchi:
                try:
                    eval_data = eval(blocco)  # Converte la stringa del blocco in un dizionario
                    accuracy_list.append(eval_data.get('eval_accuracy', 0))
                    mcc_list.append(eval_data.get('eval_matthews_correlation', 0))
                    f1_list.append(eval_data.get('eval_f1', 0))
                except:
                    continue
                
        # Calcola le medie per il file
        media_accuracy = sum(accuracy_list) / len(accuracy_list) if accuracy_list else 0
        media_mcc = sum(mcc_list) / len(mcc_list) if mcc_list else 0
        media_f1 = sum(f1_list) / len(f1_list) if f1_list else 0
        
        return media_accuracy, media_mcc, media_f1
    
    # Itera attraverso tutti i file nella cartella
    for filename in os.listdir(cartella):
        if filename.endswith(".txt"):  # Se i file sono in formato di testo
            file_path = os.path.join(cartella, filename)
            media_accuracy, media_mcc, media_f1 = estrai_metriche_con_parsing_custom(file_path)
    
            # Stampa i risultati per ogni file
            print(f"File: {filename}")
            print(f"Acc - F1 - MCC")
            print(f"{media_accuracy*100:.5f}\t{media_f1*100:.5f}\t{media_mcc*100:.5f}")

            print("*******************************************")