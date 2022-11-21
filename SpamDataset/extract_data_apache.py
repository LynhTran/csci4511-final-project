import sys
import os
import json

if len(sys.argv) != 4:
    print('Usage: python {} apache_spamassassin_email_directory json_output_name class'.format(os.path.basename(__file__)))
    sys.exit(1)

path = sys.argv[1]
output = sys.argv[2] + '.json'
category = int(sys.argv[3])
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

emails = []

for file in files:
    with open(os.path.join(path, file)) as f:
        print('Extracting data from {}'.format(file))
        text_data = None
        try:
            text_data = f.read()
        except:
            continue
        f.close()
        data_split = text_data.split('\n\n', 1)
        email_headers = data_split[0]
        email_subject = 'NONE'
        if 'Subject: ' in email_headers:
            email_subject = email_headers.split('Subject: ')[1].split('\n')[0]
        email_content = data_split[1]
        email_data = {
            'subject': email_subject,
            'content': email_content,
            'class': category
        }
        emails.append(email_data)

with open(output, 'w') as f:
    json.dump(emails, f, ensure_ascii=True, indent=4)