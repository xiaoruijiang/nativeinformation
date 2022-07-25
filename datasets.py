import json, re
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import codecs
from urllib.parse import urlparse

TARGET_CITATION = '[@@CITATION@@]'

def clean(text):
	return ''.join([i if ord(i) < 128 else '' for i in text])

def process_tensor(tensor_list, token_padding_value=0, mask_padding_value=0, seg_id_padding_value=0):
    tensor_len = [d.shape[0] for d in tensor_list]
    tensor_np = np.ones((len(tensor_list), max(tensor_len)), dtype=np.int64) * token_padding_value
    mask_np = np.ones((len(tensor_list), max(tensor_len)), dtype=np.float32) * mask_padding_value
    seg_id_np = np.ones((len(tensor_list), max(tensor_len)), dtype=np.int64) * seg_id_padding_value
    for i, (d, l) in enumerate(zip(tensor_list, tensor_len)):
        if l > 0:
            tensor_np[i][:l] = d
            mask_np[i][:l] = 1
            seg_id_np[i][:l] = 0
    return torch.from_numpy(tensor_np), torch.from_numpy(mask_np), torch.from_numpy(seg_id_np)

re_string = [r"\s[a-zA-Z\-,]+?\sand\s[a-zA-Z\-,]+?\s\(\s*?\d{4}\s*?\)", r"\s[a-zA-Z\-,]+?\s\(\s*?\d{4}\s*?\)", r"\s[a-zA-Z\-,]+?\set\sal\.\s\(\s*?\d{4}\s*?\)", r"\s\([a-zA-Z.,;&\s\d\-]+?\d{4}[a-zA-Z.,;\s\d]*?\)"]

def process_citation_sentene(sentence, citation_string=None, always_mark_citation=False):
	text = clean(sentence)

	for res in re_string:
		for t in re.findall(res, text):
			if (citation_string is not None and citation_string in t) or always_mark_citation:
				text = text.replace(t, ' ' + TARGET_CITATION)

	if citation_string is not None and TARGET_CITATION not in text:
		text = clean(sentence)
		text = text.replace(citation_string, ' ' + TARGET_CITATION)
					
	return text.strip()

class BaseCitationDataset(Dataset):

	def __init__(self, pretained_LM_tokenizer, data_json_path, intent_json_path, section_json_path, journal_json_path = None, DOI_json_path = None, is_train=False):

		self.pretained_LM_tokenizer = pretained_LM_tokenizer
		self.is_train = is_train

		with open(intent_json_path) as out:
			self.intent_json = json.loads(out.read())

		with open(section_json_path) as out:
			self.section_json = json.loads(out.read())

		if journal_json_path :
			with open(journal_json_path) as out:
				self.journal_json = json.loads(out.read())
		else:
			self.journal_json = None

		if DOI_json_path:
			with open(DOI_json_path) as out:
				self.DOI_json = json.loads(out.read())
		else:
			self.DOI_json = None

		self.citation_data = []
		with codecs.open(data_json_path, 'r', 'utf-8') as out:
			for line in out:
				citation = json.loads(line)

				intent = self.get_intention(citation)
				section_name = self.get_section(citation)
				journal_name = self.get_journal(citation)
				DOI_name = self.get_DOI(citation)

				target_citation_string, text = self.get_string(citation)
				if target_citation_string is not None:
					target_citation_string = clean(target_citation_string)
					text = process_citation_sentene(text, target_citation_string)
				else:
					text = process_citation_sentene(text, always_mark_citation=True)

				tokens = self.pretained_LM_tokenizer.tokenize(text)
				if len(tokens) > 126:
					tokens = tokens[:126]
				tokens = [self.pretained_LM_tokenizer.cls_token] + tokens + [self.pretained_LM_tokenizer.sep_token]
				token_ids = np.array(self.pretained_LM_tokenizer.convert_tokens_to_ids(tokens))

				title, cited_title = self.get_title_pair(citation)

				title_tokens = self.pretained_LM_tokenizer.tokenize(title)
				if len(title_tokens) > 80:
					title_tokens = title_tokens[:80]
				title_tokens = [self.pretained_LM_tokenizer.cls_token] + title_tokens + [self.pretained_LM_tokenizer.sep_token]
				title_token_ids = np.array(self.pretained_LM_tokenizer.convert_tokens_to_ids(title_tokens))

				cited_title_tokens = self.pretained_LM_tokenizer.tokenize(cited_title)
				if len(cited_title_tokens) > 80:
					cited_title_tokens = cited_title_tokens[:80]
				cited_title_tokens = [self.pretained_LM_tokenizer.cls_token] + cited_title_tokens + [self.pretained_LM_tokenizer.sep_token]
				cited_title_token_ids = np.array(self.pretained_LM_tokenizer.convert_tokens_to_ids(cited_title_tokens))

				prev_sen, after_sen = self.get_context_sentence_pair(citation)

				prev_sen_tokens = self.pretained_LM_tokenizer.tokenize(prev_sen)
				if len(prev_sen_tokens) > 126:
					prev_sen_tokens = prev_sen_tokens[:126]
				prev_sen_tokens = [self.pretained_LM_tokenizer.cls_token] + prev_sen_tokens + [self.pretained_LM_tokenizer.sep_token]
				prev_sen_token_ids = np.array(self.pretained_LM_tokenizer.convert_tokens_to_ids(prev_sen_tokens))

				after_sen_tokens = self.pretained_LM_tokenizer.tokenize(after_sen)
				if len(after_sen_tokens) > 126:
					after_sen_tokens = after_sen_tokens[:126]
				after_sen_tokens = [self.pretained_LM_tokenizer.cls_token] + after_sen_tokens + [self.pretained_LM_tokenizer.sep_token]
				after_sen_tokens_ids = np.array(self.pretained_LM_tokenizer.convert_tokens_to_ids(after_sen_tokens))

				if self.journal_json and self.DOI_json:
					self.citation_data.append({
						"sentence_ids": token_ids,
						"prev_sen_token_ids": prev_sen_token_ids,
						"after_sen_tokens_ids": after_sen_tokens_ids,
						"title_token_ids": title_token_ids,
						"cited_title_token_ids": cited_title_token_ids,
						"intention": self.intent_json[intent],
						"section": self.section_json[section_name],
						"url": self.journal_json[journal_name],
						"doi": self.DOI_json[DOI_name]
					})
				else:
					self.citation_data.append({
						"sentence_ids": token_ids,
						"prev_sen_token_ids": prev_sen_token_ids,
						"after_sen_tokens_ids": after_sen_tokens_ids,
						"title_token_ids": title_token_ids,
						"cited_title_token_ids": cited_title_token_ids,
						"intention": self.intent_json[intent],
						"section": self.section_json[section_name]
						# ,
						# "url": self.journal_json[journal_name],
						# "doi": self.DOI_json[DOI_name]
					})


	def get_intention(self, citation):
		pass

	def get_section(self, citation):
		pass

	def get_journal(self, citation):
		pass

	def get_DOI(self, citation):
		pass

	def get_string(self, citation):
		pass

	def get_title_pair(self, citation):
		pass

	def get_context_sentence_pair(self, citation):
		pass

	def __len__(self):
		return len(self.citation_data)

	def __getitem__(self, index):
		return self.citation_data[index]

	def get_class_num(self):
		return len(self.intent_json)

	def get_section_num(self):
		return len(self.section_json)

	def get_journal_num(self):
		if self.journal_json:
			return len(self.journal_json)
		else:
			return 0

	def get_DOI_num(self):
		if self.DOI_json:
			return len(self.DOI_json)
		else:
			return 0

class SciDataset(BaseCitationDataset):

	def get_intention(self, citation):
		return citation['label']

	def get_section(self, citation):
		return citation['sectionName']

	def get_journal(self, citation):
		if 'url' in citation:
			url = citation['url']
			website = urlparse(url).netloc
			if website not in self.journal_json:
				return 'other'
			return website
		else:
			return None

	def get_title_pair(self, citation):
		title = citation['title'] if not citation['title'].endswith('.') else citation['title'][:-1]
		cited_title = citation['cited_title'] if not citation['cited_title'].endswith('.') else citation['cited_title'][:-1]
		return title, cited_title

	def get_context_sentence_pair(self, citation):
		if len(citation['prev_sen']) == 0:
			citation['prev_sen'] = 'This is the starting sentence of the section .'
		if len(citation['after_sen']) == 0:
			citation['after_sen'] = 'This is the ending sentence of the section .'
		return citation['prev_sen'], citation['after_sen']

	def get_DOI(self, citation):
		if 'doi' in citation:
			if citation['doi'] is None or citation['doi'] == 'miss':
				citation['doi'] = 'empty'

			if citation['doi'].startswith("10."):
				doi =  citation['doi'][:7]
				if doi in self.DOI_json:
					return doi
				else:
					return 'other'
			else:
				return citation['doi']
		else:
			return None

	def get_string(self, citation):
		text = citation['string']
		try:
			s, e = int(citation['citeStart']), int(citation['citeEnd'])
			string_cite = text[s:e]
			return string_cite, ' ' + text
		except:
			return None, ' ' + text


class ACLDataset(BaseCitationDataset):

	def get_intention(self, citation):
		return citation['intent']

	def get_section(self, citation):
		return citation['section_name'] if citation['section_name'] is not None else 'missing'

	def get_string(self, citation):
		text = citation['text']
		offset = citation['cite_marker_offset']
		text = citation['text']
		return text[offset[0]: offset[1]], ' ' + text


def data_wrapper(dataset: BaseCitationDataset, device):
	labels = torch.tensor([d['intention'] for d in dataset]).long()
	section_embedding = torch.tensor([d['section'] for d in dataset]).long()

	text_list = []
	for d in dataset:
		text_list += [d['sentence_ids']]
	text, mask, seg_ids = process_tensor(text_list)

	context_text_list = []
	for d in dataset:
		context_text_list += [d['prev_sen_token_ids'], d['after_sen_tokens_ids']]
	context_text, context_mask, context_seg_ids = process_tensor(context_text_list)

	title_list = []
	for d in dataset:
		title_list += [d['title_token_ids'], d['cited_title_token_ids']]
	title_text, title_mask, title_seg_ids = process_tensor(title_list)

	if dataset.journal_json and dataset.DOI_json:
		journal_embedding = torch.tensor([d['url'] for d in dataset]).long()
		DOI_embedding = torch.tensor([d['doi'] for d in dataset]).long()
		return {
			'text': text.to(device),
			'mask': mask.to(device),
			'seg_ids': seg_ids.to(device),
			'title_text': title_text.to(device),
			'title_mask': title_mask.to(device),
			'title_seg_ids': title_seg_ids.to(device),
			'context_text': context_text.to(device),
			'context_mask': context_mask.to(device),
			'context_seg_ids': context_seg_ids.to(device),
			'labels': labels.to(device),
			'sec_names': section_embedding.to(device),
			'jour_names': journal_embedding.to(device),
			'doi_names': DOI_embedding.to(device)
		}
	else:
		return {
			'text': text.to(device),
			'mask': mask.to(device),
			'seg_ids': seg_ids.to(device),
			'title_text': title_text.to(device),
			'title_mask': title_mask.to(device),
			'title_seg_ids': title_seg_ids.to(device),
			'context_text': context_text.to(device),
			'context_mask': context_mask.to(device),
			'context_seg_ids': context_seg_ids.to(device),
			'labels': labels.to(device),
			'sec_names': section_embedding.to(device)
		}

def get_data_loader(dataset, batch_size, num_worker, device):
	collate_fn = lambda d: data_wrapper(d, device)
	return DataLoader(dataset, 
	    batch_size=batch_size, 
	    shuffle=True, 
	    num_workers=num_worker,
	    collate_fn=collate_fn
	    )

def eval_model(predictions, group_truths, label_json):
	F1_dict = {}
	correct_dict = {k: 0 for k in label_json}
	p_count_dict = {k: 0 for k in label_json}
	g_count_dict = {k: 0 for k in label_json}
	reverse_label = {v: k for (k, v) in label_json.items()}
	for p, g in zip(predictions, group_truths):
		p_count_dict[reverse_label[p]] += 1
		g_count_dict[reverse_label[g]] += 1
		if p == g:
			correct_dict[reverse_label[g]] += 1
	for label in label_json:
		recall = correct_dict[label] / g_count_dict[label]
		precision = (correct_dict[label] / p_count_dict[label]) if p_count_dict[label] > 0 else 0.0
		F1_dict[label] = (200 * recall * precision / (recall + precision)) if recall > 0 and precision > 0 else 0.0

	F1_dict['ave_F1'] = sum([v for v in F1_dict.values()]) / len(label_json)

	return F1_dict



