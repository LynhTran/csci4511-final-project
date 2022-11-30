import sys
import os
import json

if len(sys.argv) != 2:
    print('Usage: python {} path_to_sms_dataset'.format(os.path.basename(__file__)))
    sys.exit(1)

path = sys.argv[1]

ham = []
spam = []

with open(path) as f:
    print('Extracting data from {}'.format(path))
    lines = f.readlines()
    for line in lines:
        if len(line) > 0:
            data_split = line.split('\t', 1)
            print(data_split)
            sms_class = data_split[0]
            sms_content = data_split[1]
            if sms_class == 'ham':
                ham.append({
                    'subject': None,
                    'content': sms_content,
                    'class': 0
                })
            elif sms_class == 'spam':
                spam.append({
                    'subject': None,
                    'content': sms_content,
                    'class': 1
                })

with open('uci_sms_ham.json', 'w') as f:
    json.dump(ham, f, ensure_ascii=True, indent=4)

with open('uci_sms_spam.json', 'w') as f:
    json.dump(spam, f, ensure_ascii=True, indent=4)