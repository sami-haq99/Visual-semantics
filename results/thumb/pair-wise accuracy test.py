import itertools
import pandas as pd
import numpy as np

def system_level_pairwise_accuracy(df, metric):
    # df must contain: system, human_mean, metric_mean columns
    
    systems = df["system"].unique()
    correct = 0
    total = 0
    
    for sysA, sysB in itertools.combinations(systems, 2):
        hA = df[df.system == sysA]["human_mean"].item()
        hB = df[df.system == sysB]["human_mean"].item()

        # skip ties as WMT does
        if hA == hB:
            continue
        
        mA = df[df.system == sysA][metric].item()
        mB = df[df.system == sysB][metric].item()
        
        human_delta = np.sign(hA - hB)
        metric_delta = np.sign(mA - mB)
        
        if human_delta == metric_delta:
            correct += 1
        
        total += 1
    
    return correct / total

