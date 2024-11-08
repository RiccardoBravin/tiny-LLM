#code to synthesize the results of the experiments

import os
import pathlib
import re

# models = ["BERT_original", "Mamba", "Embedder_+_conv", "Embedder_only", "Nano_Bert_Efficient", "Nano_Bert_Efficient_mh"]
models = ["IDEA1", "mlm_New_idea2_30"]

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

                try: #there are results for F1, accuracy and MCC
                    print(f"{sum(accuracies)/len(accuracies)*100:<15}\t{sum(f1s)/len(f1s)*100:<15}\t{sum(mccs)/len(mccs)*100:<15}".replace('.', ','))
                except:
                    try: #there are results just for spearman correlation coefficient
                        print(f"{sum(sccs)/len(sccs)*100:<15}".replace('.', ','))
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
            print(f"Media Accuracy: {media_accuracy*100:.5f}")
            print(f"Media MCC: {media_mcc*100:.5f}")
            print(f"Media F1: {media_f1*100:.5f}")
            print("*******************************************")