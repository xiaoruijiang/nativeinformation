import json
import codecs
from tqdm import tqdm

train_dict = {}
dev_dict = {}
test_dict = {}
with codecs.open('dataset_updated_final_v5.jsonl', 'r', 'utf-8') as out:
    with codecs.open('dataset_updated_final_train_v5.jsonl', 'w', 'utf-8') as train_out:
        with codecs.open('dataset_updated_final_dev_v5.jsonl', 'w', 'utf-8') as dev_out:
            with codecs.open('dataset_updated_final_test_v5.jsonl', 'w', 'utf-8') as test_out:
                for line in tqdm(out.readlines()):
                    line = line.strip()
                    try:
                        paper_info = json.loads(line)
                    except:
                        print("missing", line)
                    if int(paper_info['year']) in [2017, 2018, 2019]:
                        if paper_info['label'] not in test_dict:
                            test_dict[paper_info['label']] = 0
                        test_dict[paper_info['label']] += 1
                        test_out.write(line + '\n')
                    elif int(paper_info['year']) == 2016:
                        if paper_info['label'] not in dev_dict:
                            dev_dict[paper_info['label']] = 0
                        dev_dict[paper_info['label']] += 1
                        dev_out.write(line + '\n')
                    else:
                        if paper_info['label'] not in train_dict:
                            train_dict[paper_info['label']] = 0
                        train_dict[paper_info['label']] += 1
                        train_out.write(line + '\n')

print(train_dict)
print(dev_dict)
print(test_dict)