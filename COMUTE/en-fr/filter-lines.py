filename = "/home/sami/mmt-eval/eval-datasets/commute/commute/en-fr/captions-commute-gemma3b.fr"

with open(filename, "r") as f:
    lines = f.readlines()

with open('commute-caps.germa.txt', "w") as f:
    for line in lines:
        if line.strip():  # keep only non-empty
            f.write(line)
