from random import sample
import json

with open('./dpo_ready_data.json', 'r') as f:
    data = json.load(f)

test_set_idx = set(sample(range(len(data)), 50))
test_set = [data[idx] for idx in test_set_idx]
train_set = [data[idx] for idx in range(len(data)) if idx not in test_set_idx]
print(len(train_set), len(test_set),len(data))
with open('./dpo_ready_data_train.json', 'w') as f:
    json.dump(train_set, f, indent=2)

with open('./dpo_ready_data_test.json', 'w') as f:
    json.dump(test_set, f, indent=2)