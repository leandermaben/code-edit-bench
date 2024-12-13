import json
import numpy as np

DATA_BASE_PATH = '/data/tir/projects/tir7/user_data/lmaben/code-edit-bench_new/data'
OUT_DIR = 'data/finetune'
SPLITS = [2,3,5,7,8]

TRAIN_NUM = 5200
VAL_NUM = 750

def consolidate_data(splits=SPLITS):
    rows = []
    count = 0
    for split in splits:
        input_file = f'{DATA_BASE_PATH}/sampled_commits_Qwen_Qwen2.5-72B-Instruct_split_{split}.jsonl'
        with open(input_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                for file in data['files']:
                    new_row = dict(data)
                    new_row.pop('files')
                    for file_key in file.keys():
                        if file_key == 'draft_from_diff':
                            file[file_key] = file[file_key].replace('<update_snippet>', '').replace('</update_snippet>', '')                                                                                                                                                                                                                                                            
                        new_row[f'file_{file_key}'] = file[file_key]
                    try:
                        file['previous_content'].encode('utf-8')
                        file['current_content'].encode('utf-8')
                        file['draft_from_diff'].encode('utf-8')
                    except:
                        print(f'Error encoding original_code for file {file_key}')
                        continue
                    rows.append(new_row)
    return rows

def train_val_test_split(rows):
    np.random.seed(42)
    np.random.shuffle(rows)
    train_rows = rows[:TRAIN_NUM]
    val_rows = rows[TRAIN_NUM:TRAIN_NUM+VAL_NUM]
    test_rows = rows[TRAIN_NUM+VAL_NUM:]
    return train_rows, val_rows, test_rows

def save_data(rows, out_file):
    with open(out_file, 'w') as f:
        for row in rows:
            json.dump(row, f)
            f.write('\n')
    print(f'Saved {len(rows)} rows to {out_file}')


rows = consolidate_data()
print(f'Total rows: {len(rows)}')
train_rows, val_rows, test_rows = train_val_test_split(rows)

save_data(train_rows, f'{OUT_DIR}/train.jsonl')
save_data(val_rows, f'{OUT_DIR}/val.jsonl')
save_data(test_rows, f'{OUT_DIR}/test.jsonl')


