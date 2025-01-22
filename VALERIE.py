from datasets import load_dataset

dataset = load_dataset("Intel/VALERIE22", split="test")
print(dataset)