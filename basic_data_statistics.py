import json
import codecs
from string import punctuation
from urllib.parse import urlparse
from transformers import BertTokenizer

bert_dir = "scibert_scivocab_uncased"
tokenizer = BertTokenizer.from_pretrained(bert_dir)

total_instance_count = 0
paper_ids = set()
label_dict = {}
doi_missing = 0
title_missing = 0
no_doi_paper_ids = set()

def clean_up_venue(venue):
    venue = venue.lower()
    c_list = []
    for i, c in enumerate(venue.split()):
        if c == ' ':
            continue
        if i == 0 and c == 'the':
            continue
        if i == len(venue.split()) - 1 and len(c) == 1:
            continue
        c_list.append(c)
    venue = ' '.join(c_list)
    venue = ''.join([c for c in venue if (not c.isdigit()) and c not in punctuation])
    venue = venue.strip()
    if 'cvpr' in venue:
        venue = 'cvpr'
    if 'iccv' in venue:
        venue = 'iccv'
    if 'emnlp' in venue or 'acl' in venue or venue in ['anlp', 'sigdial workshop', 'coling', 'conll shared task']:
        venue = 'acl'
    if 'neurips' in venue:
        venue = 'nips'
    return venue

venue_to_doi = {}
paper_id_to_doi = {}
with codecs.open('new_scicite/all_with_section_name_updated_final_v4.jsonl', 'r', 'utf-8') as out:
    for line in out:
        line = line.strip()
        try:
            paper_info = json.loads(line)
        except:
            print('missing + ' + line)
            continue
        cleaned_v = clean_up_venue(paper_info['venue'])

        if paper_info['doi'] is not None and not paper_info['doi'] == 'empty':
            doi_key = paper_info['doi'].split('/')[0]
            if not (paper_info['venue'] is None or len(cleaned_v) == 0):
                venue_to_doi[cleaned_v] = doi_key
            paper_id_to_doi[paper_info['citingPaperId']] = paper_info['doi']

average_sentence_length = 0
average_title_length = 0
sentence_ratio = 0
title_ratio = 0

doi_distribution = {'empty': 0}
url_distribution = {}
section_distribution = {}
time_missing_count = 0
with codecs.open('new_scicite/all_with_section_name_updated_final_v4.jsonl', 'r', 'utf-8') as out:
    lines = out.readlines()
    for index, line in enumerate(lines):
        total_instance_count += 1
        line = line.strip()
        try:
            paper_info = json.loads(line)
        except:
            print(index, 'missing + ' + line)
            continue
        length = len(tokenizer.tokenize(paper_info['string']))
        average_sentence_length += length
        if length > 126: sentence_ratio += 1

        length = len(tokenizer.tokenize(paper_info['prev_sen']))
        average_sentence_length += length
        if length > 126: sentence_ratio += 1

        length = len(tokenizer.tokenize(paper_info['after_sen']))
        average_sentence_length += length
        if length > 126: sentence_ratio += 1

        length = len(tokenizer.tokenize(paper_info['title']))
        average_title_length += length
        if length > 80: title_ratio += 1

        length = len(tokenizer.tokenize(paper_info['cited_title']))
        average_title_length += length
        if length > 80: title_ratio += 1

        if paper_info['year'] is None or not len(str(paper_info['year'])) == 4 or paper_info['year'] == 'miss':
            print(paper_info['citingPaperId'], paper_info['title'])
            time_missing_count += 1
        paper_ids.add(paper_info['citingPaperId'])
        if paper_info['label'] not in label_dict:
            label_dict[paper_info['label']] = 0
        label_dict[paper_info['label']] += 1
        if paper_info['sectionName'] not in section_distribution:
            section_distribution[paper_info['sectionName']] = 0
        section_distribution[paper_info['sectionName']] += 1
        cleaned_v = clean_up_venue(paper_info['venue'])
        if paper_info['doi'] is None:
            paper_info['doi'] = 'empty'

        if paper_info['doi'] is None or not paper_info['doi'].startswith('10.'):
            if cleaned_v not in venue_to_doi:
                no_doi_paper_ids.add(paper_info['citingPaperId'])
                doi_missing += 1
                doi_distribution['empty'] += 1
            else:
                doi_key = venue_to_doi[cleaned_v]
                if doi_key not in doi_distribution:
                    doi_distribution[doi_key] = 0
                doi_distribution[doi_key] += 1
        else:
            doi_key = paper_info['doi'].split('/')[0]
            if doi_key not in doi_distribution:
                doi_distribution[doi_key] = 0
            doi_distribution[doi_key] += 1

        if paper_info['url'] is not None and len(paper_info['url']) > 0:
            root = urlparse(paper_info['url']).netloc
            if root not in url_distribution:
                url_distribution[root] = 0
            url_distribution[root] += 1


        if paper_info['title'] is None or len(paper_info['title']) == 0 or paper_info['cited_title'] is None or len(paper_info['cited_title']) == 0:
            title_missing += 1

doi_out = {'other': 0}
total_count = 0
sorted_doi_dist = sorted(doi_distribution.items(), key=lambda x: x[1], reverse=True)
print("===============doi===============")
for doi, count in sorted_doi_dist:
    if count >= 5:
        print(doi, count)
        total_count += count
        doi_out[doi] = len(doi_out)
print(total_count, total_instance_count, total_count / total_instance_count)
with codecs.open('new_scicite/doi_label.json', 'w', 'utf-8') as out:
    out.write(json.dumps(doi_out))

url_out = {'other': 0}
url_total_count = 0
sorted_url_dist = sorted(url_distribution.items(), key=lambda x: x[1], reverse=True)
print("===============url===============")
for url, count in sorted_url_dist:
    if count >= 5:
        print(url, count)
        url_total_count += count
        url_out[url] = len(url_out)
print(url_total_count, total_instance_count, url_total_count / total_instance_count)
with codecs.open('new_scicite/url_label.json', 'w', 'utf-8') as out:
    out.write(json.dumps(url_out))

# acl_paper_ids = set()
# acl_paper_count = 0
# with codecs.open('new_scicite/acl_added.jsonl', 'r', 'utf-8') as out:
#     lines = out.readlines()
#     for line in lines:
#         acl_paper_count += 1
#         line = line.strip()
#         paper_info = json.loads(line)
#         acl_paper_ids.add(paper_info['citingPaperId'])

print('total instances', total_instance_count)
print('total papers', len(paper_ids))
print("label distribution", label_dict)
print('section name', section_distribution)
print('doi missing', doi_missing, len(no_doi_paper_ids))
print('title missing', title_missing)
print('time missing', time_missing_count)
print('averaged sentence length', average_sentence_length / (3 * total_instance_count))
print('longer sentence ratio', sentence_ratio / (3 * total_instance_count))
print('averaged title length', average_title_length / (2 * total_instance_count))
print('longer title ratio', title_ratio / (2 * total_instance_count))


