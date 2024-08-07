#code to synthesize the results of the experiments

import os
import pathlib
import re

models = ["BERT_original", "Embbert", "Embedder_+_conv", "Embedder_only", "Nano_Bert_Efficient"]
# models = ["Nano_Bert_Efficient_Nano_Bert_Efficient"]

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
            
            try:
                
                print(f"{sum(accuracies)/len(accuracies)*100:<15}\t{sum(f1s)/len(f1s)*100:<15}\t{sum(mccs)/len(mccs)*100:<15}".replace('.', ','))
                # print(f"F1 mean:\t\t{sum(f1s)/len(f1s)*100}".replace('.', ','))
                # print(f"Accuracy mean:\t{sum(accuracies)/len(accuracies)*100}".replace('.', ','))
                # print(f"MCC mean:\t\t{sum(mccs)/len(mccs)*100}".replace('.', ','))
            except:
                print("No results")

    print("\n*************************************************\n")
