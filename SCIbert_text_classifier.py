from torch import nn
import torch

class BertClassifier(nn.Module):

	def __init__(self, bert_pretrained_model, bert_output_size, num_classes, num_section, num_journal, num_DOI, dropout):
		super().__init__()
		self.bert_pretrained_model = bert_pretrained_model
		self.dropout = nn.Dropout(dropout)
		self.num_classes = num_classes
		self.bert_output_size = bert_output_size
		self.classifier = nn.Linear(bert_output_size, self.num_classes)
		#self.section_embedding = nn.Embedding(num_section, 256)
		# self.journal_embedding = nn.Embedding(num_journal, 256)
		# self.DOI_embedding = nn.Embedding(num_DOI, 256)
		self.loss = nn.CrossEntropyLoss(reduction='mean')

	def forward(
        self,
        input_ids,
		title_ids,
		context_ids,
        sec_ids,
		jour_ids,
		doi_ids,
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

		outputs = self.bert_pretrained_model(
		    input_ids,
		    attention_mask=attention_mask,
		    token_type_ids=token_type_ids,
		    position_ids=position_ids,
		    head_mask=head_mask,
		    inputs_embeds=inputs_embeds,
		)

		# title_outputs = self.bert_pretrained_model(
		# 	title_ids,
		# 	attention_mask=title_attention_mask,
		# 	token_type_ids=title_token_type_ids,
		# 	position_ids=position_ids,
		# 	head_mask=head_mask,
		# 	inputs_embeds=inputs_embeds,
		# )
		#
		# context_outputs = self.bert_pretrained_model(
		# 	context_ids,
		# 	attention_mask=context_attention_mask,
		# 	token_type_ids=context_token_type_ids,
		# 	position_ids=position_ids,
		# 	head_mask=head_mask,
		# 	inputs_embeds=inputs_embeds,
		# )

		sen_vec = outputs[1]
		# title_vec = title_outputs[1].reshape(-1, 2 * self.bert_output_size)
		# context_vec = context_outputs[1].reshape(-1, 2 * self.bert_output_size)
		# #j_embed = self.journal_embedding(jour_ids)
		# doi_embed = self.DOI_embedding(doi_ids)
		#sec_embed = self.section_embedding(sec_ids)
		#output = torch.cat([sen_vec, sec_embed], dim=1)

		return self.classifier(self.dropout(sen_vec))

	def cal_loss(self, logits, labels):
		return self.loss(logits, labels)

	def prediction(self, logits):
		_, max_index = torch.max(logits, 1)
		return max_index





