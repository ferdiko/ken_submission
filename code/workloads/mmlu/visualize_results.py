import numpy as np


def get_accuracy(path):
    with open(path, "r") as f:
        lines = f.readlines()
    
    correct = 0
    total = 0

    for l in lines:
        pred = int(l.split(",")[1])
        corr = int(l.split(",")[2])
        total += 1

        if pred == corr:
            correct += 1

    return correct/total

def print_thresholds(model1_path, model2_path):

    cost_fact = runtimes_eager[model2_path] / runtimes_eager[model1_path]

    with open(model1_path, "r") as f:
        model1_lines = f.readlines()

    with open(model2_path, "r") as f:
        model2_lines = f.readlines()

    certainties = [float(l.split(",")[0]) for l in model1_lines]

    for t in np.linspace(np.min(certainties), np.max(certainties), 20):
        print(f"t {t:.2f}: ", end="\t")

        correct = 0
        total = 0
        cost = 0

        for l1, l2 in zip(model1_lines, model2_lines):
            total += 1
            cost += 1

            if float(l1.split(",")[0]) < t:
                # Cascade
                if int(l2.split(",")[2]) == int(l2.split(",")[1]):
                    correct += 1
                    cost += cost_fact
            else:
                if int(l1.split(",")[2]) == int(l1.split(",")[1]):
                    correct += 1

        print(f"{correct/total}\t{cost}")

    print("MAX:", runtimes_eager[model2_path]*1000)


print(get_accuracy("Meta-Llama-3.1-8B-Instruct-AWQ-INT4.csv"))
print(get_accuracy("Meta-Llama-3.1-8B-Instruct.csv"))
print(get_accuracy("Meta-Llama-3.1-70B-Instruct-AWQ-INT4.csv"))
print(get_accuracy("Llama-3.2-3B-Instruct.csv"))
print(get_accuracy("Llama-3.2-1B-Instruct.csv"))
print(get_accuracy("Llama-3.2-3B-Instruct-AWQ.csv"))


print()
print()

runtimes_eager = {
    "Meta-Llama-3.1-8B-Instruct-AWQ-INT4.csv": 29.483928442001343,
    "Meta-Llama-3.1-8B-Instruct.csv": 18.3103609085083,
    "Meta-Llama-3.1-70B-Instruct-AWQ-INT4.csv": 236.74053931236267,
    "Llama-3.2-3B-Instruct.csv": 13.064658164978027,
    "Llama-3.2-1B-Instruct.csv": 8.751412868499756,
    "Llama-3.2-3B-Instruct-AWQ.csv": 17.038991928100586
}


# print(print_thresholds("Meta-Llama-3.1-8B-Instruct-AWQ-INT4.csv", "Meta-Llama-3.1-70B-Instruct-AWQ-INT4.csv", 5))
print_thresholds("Llama-3.2-1B-Instruct.csv", "Meta-Llama-3.1-70B-Instruct-AWQ-INT4.csv")