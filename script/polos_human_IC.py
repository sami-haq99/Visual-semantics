
from datasets import load_dataset

polaris = load_dataset("yuwd/Polaris")
print(polaris)

#extract the text data
test_texts = polaris["test"]

#print the first few examples
for i in range(5):
    print(test_texts[i])
    
    print(test_texts[i]["refs"])
    print(test_texts[i]["cand"])
    print(test_texts[i]["img"])
    print(test_texts[i]["human_score"])

