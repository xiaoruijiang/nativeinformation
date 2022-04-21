import spacy
import re, math
from collections import Counter
import json
import codecs
from string import punctuation
from tqdm import tqdm
import os.path

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
with codecs.open('new_scicite/all_with_section_name_updated_final_v2.jsonl', 'r', 'utf-8') as out:
    for line in out:
        line = line.strip()
        paper_info = json.loads(line)
        cleaned_v = clean_up_venue(paper_info['venue'])

        if paper_info['doi'] is not None:
            doi_key = paper_info['doi'].split('/')[0]
            if not (paper_info['venue'] is None or len(cleaned_v) == 0):
                venue_to_doi[cleaned_v] = doi_key


templates = ["introduction", "experiment", "conclusion", "related work", "method", "result and discussion", "abstract"]
def get_section(sectionName):
    if sectionName is None:
        return 'missing'
    s = str(sectionName)
    s = s.lower()
    s = ''.join([c for c in s if c not in [x for x in punctuation] + [str(i) for i in range(10)]])
    s = s.strip()
    if 'experiment' in s:
        s = 'experiment'
    if 'method' in s:
        s = 'method'
    if 'introduction' in s:
        s = 'introduction'
    if 'motivation' in s:
        s = 'introduction'
    if 'result' in s or 'discussion' in s:
        s = 'result and discussion'
    if 'background' in s:
        s = 'related work'
    if 'implementation' in s:
        s = 'method'
    if 'conclusion' in s:
        s = 'conclusion'
    if 'related work' in s:
        s = 'related work'
    if s not in templates:
        s = 'missing'
    return s

WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator


def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)

def ngrams(string, n=3):
    ngrams = zip(*[string[i:] for i in range(n)])
    return set([' '.join(ngram) for ngram in ngrams])

paper_ids = set()
with codecs.open('new_scicite/all_with_section_name_updated_final_v5.jsonl', 'r', 'utf-8') as out:
    lines = out.readlines()
    for l in lines:
        l = l.strip()
        record = json.loads(l)
        paper_ids.add(record['citingPaperId'])

nlp = spacy.load("en_core_sci_sm")
bad_case = 0
sec_or_doi = 0
no_file = 0
label_dict = {}
json_missing_list = []
with codecs.open('new_scicite/all_with_section_name_updated_final_v4.jsonl', 'r', 'utf-8') as out:
    with codecs.open('new_scicite/all_with_section_name_updated_final_v6.jsonl', 'w', 'utf-8') as good_out:
        with codecs.open('new_scicite/no_match_final_v4.txt', 'w', 'utf-8') as bad_out:
            lines = out.readlines()
            with tqdm(total=len(lines)) as pbar:
                for l in lines:
                    l = l.strip()
                    record = json.loads(l)
                    paper_id = record['citingPaperId']
                    if paper_id not in paper_ids:
                        sentence = record['string']
                        if sentence.startswith('…'):
                            sentence = sentence[1:]
                        if sentence.endswith('…'):
                            sentence = sentence[:-1]
                        sentence = sentence.replace('\n', ' ')
                        sentence = sentence.replace('( ', '(')
                        sentence = sentence.replace(' )', ')')
                        sentence = sentence.replace('\ud835', ' ')
                        sentence = sentence.replace('\udc5d', ' ')
                        sentence = sentence.replace('\udf14', ' ')
                        sentence = sentence.replace('\udc45', ' ')
                        #file_path = 'C:\\Users\\45068208\\Desktop\\citation\\pdfs\\%s.pdf.json' % paper_id
                        file_path = 'C:\\Users\\45068208\\Desktop\\citation\\new_tool_output\\new_json_%s.json' % paper_id
                        if os.path.exists(file_path):
                            with codecs.open(file_path, 'r','utf-8') as paper_file:
                                paper = json.loads(paper_file.read())
                                sec_list = paper['metadata']['sections']

                                if 'abstractText' in paper['metadata'] and paper['metadata']['abstractText'] is not None:
                                    if sec_list is None:
                                        sec_list = [{'text': paper['metadata']['abstractText'], 'heading': 'abstract'}]
                                    else:
                                        sec_list.append({'text': paper['metadata']['abstractText'], 'heading': 'abstract'})

                                section_list = [get_section(a['heading'] if 'heading' in a else None) for a in sec_list]
                                section_index = 0
                                sentence_index = 0
                                max_similarity = 0
                                sen_ngram = ngrams(sentence.split())
                                sec_sen_list = []
                                for i, (sec, name) in enumerate(zip(sec_list, section_list)):
                                    text = sec['text'].replace('\n', ' ')
                                    text = text.replace('\ud835', ' ')
                                    text = text.replace('\udc5d', ' ')
                                    text = text.replace('\udf14', ' ')
                                    text = text.replace('\udf14', ' ')

                                    sec_ngram = ngrams(text.split())
                                    recall = len(sen_ngram & sec_ngram) / len(sen_ngram)

                                    sen_list = []
                                    if recall > 0:
                                        doc = nlp(text)
                                        sen_list = [sen.text for sen in doc.sents]
                                        for index, sen in enumerate(sen_list):
                                            v1 = text_to_vector(sen)
                                            v2 = text_to_vector(sentence)
                                            similarity = get_cosine(v1, v2)
                                            if similarity > max_similarity:
                                                max_similarity = similarity
                                                sentence_index = index
                                                section_index = i
                                    sec_sen_list.append(sen_list)

                                if max_similarity > 0:
                                    if sentence_index == 0:
                                        record['prev_sen'] = 'This is the starting sentence of the section .'
                                    else:
                                        record['prev_sen'] = sec_sen_list[section_index][sentence_index - 1]

                                    if sentence_index == len(sec_sen_list[section_index]) - 1:
                                        record['after_sen'] = 'This is the ending sentence of the section .'
                                    else:
                                        record['after_sen'] = sec_sen_list[section_index][sentence_index + 1]

                                    record['mid_sen'] = sec_sen_list[section_index][sentence_index]

                                    selected_name = 'missing'
                                    start_index = section_index
                                    while start_index >= 0:
                                        if section_list[start_index] in templates:
                                            selected_name = section_list[start_index]
                                            break
                                        start_index -= 1

                                    # doi_ok = False
                                    # if record['doi'] is not None and record['doi'].startswith('10.'):
                                    #     doi_ok = True
                                    # else:
                                    #     cleaned_v = clean_up_venue(record['venue'])
                                    #     if len(cleaned_v) > 0:
                                    #         if cleaned_v in venue_to_doi:
                                    #             doi_ok = True
                                    #             record['doi'] = venue_to_doi

                                    if paper['metadata']['title'] is not None and len(paper['metadata']['title']) > 0:
                                        record['title'] = paper['metadata']['title']
                                    if selected_name in templates or record['sectionName'] in templates:
                                        record['sectionName'] = selected_name if not selected_name == 'missing' else record['sectionName']
                                        good_out.write(json.dumps(record) + '\n')
                                    else:
                                        bad_out.write(l + '\n')
                                        if record['label'] not in label_dict:
                                            label_dict[record['label']] = 0
                                        label_dict[record['label']] += 1
                                        sec_or_doi += 1
                                else:
                                    bad_out.write(l + '\n')
                                    if record['label'] not in label_dict:
                                        label_dict[record['label']] = 0
                                    label_dict[record['label']] += 1
                                    bad_case += 1
                        else:
                            json_missing_list.append(paper_id)
                            good_out.write(json.dumps(record) + '\n')
                            no_file += 1

                    pbar.update(1)
                    update_string = 'bad case %d json missing %d sec_or_doi %d ' % (bad_case, no_file, sec_or_doi)
                    for key in label_dict:
                        update_string += ' %s %d ' % (key, label_dict[key])
                    pbar.set_description(update_string)

for pid in json_missing_list:
    print(pid)

