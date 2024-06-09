#code to synthesize the results of the experiments

import os
import pathlib

models = ["Bert_efficient", "Brav", "Nano_Bert_Efficient"]

for model in models:
    print(f"Results for model: {model}")

    # For each file inside the results/model directory 
    for file in os.listdir(f"results/{model}"):
        #keep from the file name only what is after cls_ and before _classification...
        file_name = file.split("_")[0]
        print(f"\t\t{file_name}")

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

            print("F1 mean:", sum(f1s)/len(f1s))
            print("Accuracy mean:", sum(accuracies)/len(accuracies))
            print("MCC mean:", sum(mccs)/len(mccs))

    print("\n*************************************************\n")