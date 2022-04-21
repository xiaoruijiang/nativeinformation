import codecs
import json
from urllib.parse import quote
import requests
from string import punctuation

# count = 0
# with codecs.open('new_scicite/all_with_section_name_acl_updated.jsonl', 'r', 'utf-8') as out:
#     for line in out:
#         line = line.strip()
#         paper_info = json.loads(line)
#
#         if len(paper_info['title']) == 0:
#             count += 1
# print(count)


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


attr_name = 'venue'
count = 0
total = 0
venue_to_doi = {}
venue_without_doi = set()
with codecs.open('new_scicite/all_with_section_name_updated_v2.jsonl', 'r', 'utf-8') as out:
    for line in out:
        line = line.strip()
        paper_info = json.loads(line)
        cleaned_v = clean_up_venue(paper_info['venue'])

        if paper_info['doi'] is not None:
            doi_key = paper_info['doi'].split('/')[0]
            if not (paper_info['venue'] is None or len(cleaned_v) == 0):
                venue_to_doi[cleaned_v] = doi_key

paper_ids = set()
paper_title_id = []
label_dict = {}
time_missing = 0
with codecs.open('new_scicite/all_with_section_name_updated_v2.jsonl', 'r', 'utf-8') as out:
    with codecs.open('new_scicite/all_with_section_name_updated_final.jsonl', 'w', 'utf-8') as writer:
        lines = out.readlines()
        for line in lines:
            line = line.strip()
            paper_info = json.loads(line)
            total += 1
            if paper_info['doi'] is None:
                cleaned_v = clean_up_venue(paper_info['venue'])
                if not (paper_info['venue'] is None or len(cleaned_v) == 0):
                    if cleaned_v in venue_to_doi:
                        writer.write(json.dumps(paper_info) + '\n')
                        continue
                if paper_info['label'] == 'background':
                    continue
                count += 1
                writer.write(json.dumps(paper_info) + '\n')
                if paper_info['label'] not in label_dict:
                    label_dict[paper_info['label']] = 0
                label_dict[paper_info['label']] += 1
                if paper_info['citingPaperId'] not in paper_ids:
                    paper_ids.add(paper_info['citingPaperId'])
            else:
                writer.write(json.dumps(paper_info) + '\n')

print(label_dict)
print(count, len(paper_ids), len(venue_without_doi), len(venue_to_doi), len(lines))

