from torch import nn
import torch

class BertClassifier(nn.Module):

	def __init__(self, bert_pretrained_model, _A, num_classes, num_section, num_journal=None, num_DOI=None):
		super().__init__()
		self.bert_pretrained_model = bert_pretrained_model
		# self.dropout = nn.Dropout(dropout)
		self.dropout = _A.dropout
		self.num_classes = num_classes
		# self.bert_output_size = bert_output_size
		self.bert_output_size = self.bert_pretrained_model.config.hidden_size
		self.feature_size = self.bert_output_size
		if _A.use_title:
			self.feature_size += self.bert_output_size * 2
		if _A.use_context:
			self.feature_size += self.bert_output_size * 2
		if _A.use_section:
			self.section_embedding = nn.Embedding(num_section, 256)
			self.feature_size += 256
		if _A.use_url:
			self.journal_embedding = nn.Embedding(num_journal, 256)
			self.feature_size += 256
		if _A.use_doi:
			self.DOI_embedding = nn.Embedding(num_DOI, 256)
			self.feature_size += 256

		# self.classifier = nn.Linear(bert_output_size, self.num_classes)
		self.classifier = nn.Linear(self.feature_size, self.num_classes)
		self.loss = nn.CrossEntropyLoss(reduction='mean')

		self.args = _A

	def forward(
        self,
        input_ids,
		title_ids,
		context_ids,
        sec_ids = None,
		jour_ids = None,
		doi_ids = None,
        attention_mask=None,
        token_type_ids=None,
		title_attention_mask=None,
		title_token_type_ids=None,
		context_attention_mask=None,
		context_token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):

		sen_outputs = self.bert_pretrained_model(
		    input_ids,
		    attention_mask=attention_mask,
		    token_type_ids=token_type_ids,
		    position_ids=position_ids,
		    head_mask=head_mask,
		    inputs_embeds=inputs_embeds,
		)

		title_outputs = self.bert_pretrained_model(
			title_ids,
			attention_mask=title_attention_mask,
			token_type_ids=title_token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
		)

		context_outputs = self.bert_pretrained_model(
			context_ids,
			attention_mask=context_attention_mask,
			token_type_ids=context_token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
		)

		feature_vectors = []
		sen_vec = sen_outputs[1]	# [cls] of sent
		feature_vectors.append(sen_vec)
		if self.args.use_title:
			# title_outputs[1]: [cls] of title, BxH
			title_vec = title_outputs[1].reshape(-1, 2 * self.bert_output_size)
			feature_vectors.append(title_vec)
		if self.args.use_context:
			# context_outputs[1]: [cls] of context	BxH
			context_vec = context_outputs[1].reshape(-1, 2 * self.bert_output_size)
			feature_vectors.append(context_vec)
		if self.use_url:
			j_embed = self.journal_embedding(jour_ids)
			feature_vectors.append(j_embed)
		if self.use_doi:
			doi_embed = self.DOI_embedding(doi_ids)
			feature_vectors.append(doi_embed)
		

		# feature_vec = torch.cat([sen_vec, sec_embed], dim=1)
		feature_vec = torch.cat(feature_vectors, dim=1)

		return self.classifier(self.dropout(feature_vec))

	def cal_loss(self, logits, labels):
		return self.loss(logits, labels)

	def prediction(self, logits):
		_, max_index = torch.max(logits, 1)
		return max_index





