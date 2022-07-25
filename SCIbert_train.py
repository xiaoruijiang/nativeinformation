from datasets import SciDataset, ACLDataset, get_data_loader, TARGET_CITATION, eval_model
from bert_text_classifier import BertClassifier
from transformers import BertModel,BertTokenizer
from transformers.optimization import get_linear_schedule_with_warmup, AdamW
import torch
import argparse
import numpy as np
from tqdm import tqdm
from config import Config

parser = argparse.ArgumentParser("Train Bert for Citation Functions")
parser.add_argument(
    "--gpu-ids", required=True, nargs="+", type=int, help="List of GPU IDs to use (-1 for CPU)."
)
parser.add_argument(
    "--cpu-workers", type=int, default=0, help="Number of CPU workers to use for data loading."
)
parser.add_argument(
    "--serialization-dir",
    default="checkpoints/experiment",
    help="Path to a (non-existent) directory for serializing checkpoints and tensorboard logs.",
)
parser.add_argument(
    "--bert-dir",
    default="checkpoints/experiment",
    help="Path to a (non-existent) directory for serializing checkpoints and tensorboard logs.",
)
parser.add_argument('--seed',  type=int, default=8083, help='Init seed')
# parser.add_argument(
#     "--config", required=True, help="Path to a config file with all configuration parameters."
# )
# parser.add_argument(
#     "--config-override",
#     default=[],
#     nargs="*",
#     help="A sequence of key-value pairs specifying certain config arguments (with dict-like "
#     "nesting) using a dot operator. The actual config will be updated and recorded in "
#     "the serialization directory.",
# )
parser.add_argument('--data_type',  type=str, default="sci", help='SciDataset ot ACLDataset')
parser.add_argument('--batch_size',  type=int, default=16)
parser.add_argument('--eval_batch_size',  type=int, default=16)
parser.add_argument('--lr',  type=float, default=5e-4)
parser.add_argument('--accumulation_steps',  type=int, default=150)
parser.add_argument('--warmup_steps',  type=int, default=1500)
parser.add_argument('--epochs',  type=int, default=20)
parser.add_argument('--max_grad_norm',  type=float, default=1.0)

parser.add_argument("--use_section", action='store_true', default=False)
parser.add_argument("--use_title", action='store_true', default=False)
parser.add_argument("--use_context", action='store_true', default=False)
parser.add_argument("--use_url", action='store_true', default=False)
parser.add_argument("--use_doi", action='store_true', default=False)



if __name__ == "__main__":
	# --------------------------------------------------------------------------------------------
	#   INPUT ARGUMENTS AND CONFIG
	# --------------------------------------------------------------------------------------------
	_A = parser.parse_args()
	# _C = Config(_A.config, _A.config_override)

	np.random.seed(_A.seed)
	torch.manual_seed(_A.seed)
	torch.cuda.manual_seed_all(_A.seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

	device = torch.device(f"cuda:{_A.gpu_ids[0]}" if _A.gpu_ids[0] >= 0 else "cpu")

	tokenizer = BertTokenizer.from_pretrained(_A.bert_dir)
	model = BertModel.from_pretrained(_A.bert_dir)

	tokenizer.add_tokens([TARGET_CITATION])
	model.resize_token_embeddings(len(tokenizer))

	# D = SciDataset if _C.DATA.TYPE == 'sci' else ACLDataset
	Dataset = SciDataset if _A.data_type == 'sci' else ACLDataset

	# train_dataset = Dataset(_C.DATA.TRAIN_PATH, _C.DATA.LABEL_PATH, _C.DATA.CITATION_LABEL_PATH, _C.DATA.URL_LABEL_PATH, _C.DATA.DOI_LABEL_PATH, tokenizer, is_train=True)
	# train_loader = get_data_loader(train_dataset, _C.OPTIM.BATCH_SIZE, _A.cpu_workers, device)
	train_dataset = Dataset(tokenizer, _A.train_path, _A.intent_path, _A.section_path, is_train=True)
	train_loader = get_data_loader(train_dataset, _A.batch_size, _A.cpu_workers, device)

	# dev_dataset = Dataset(_C.DATA.DEV_PATH, _C.DATA.LABEL_PATH, _C.DATA.CITATION_LABEL_PATH, _C.DATA.URL_LABEL_PATH, _C.DATA.DOI_LABEL_PATH, tokenizer)
	# dev_loader = get_data_loader(dev_dataset, _C.OPTIM.EVAL_BATCH_SIZE, _A.cpu_workers, device)
	dev_dataset = Dataset(tokenizer, _A.dev_path, _A.intent_path, _A.section_path, is_train=False)
	dev_loader = get_data_loader(dev_dataset, _A.eval_batch_size, _A.cpu_workers, device)

	# test_dataset = Dataset(_C.DATA.TEST_PATH, _C.DATA.LABEL_PATH, _C.DATA.CITATION_LABEL_PATH, _C.DATA.URL_LABEL_PATH, _C.DATA.DOI_LABEL_PATH, tokenizer)
	# test_loader = get_data_loader(test_dataset, _C.OPTIM.EVAL_BATCH_SIZE, _A.cpu_workers, device)
	test_dataset = Dataset(tokenizer, _A.test_path, _A.intent_path, _A.section_path, is_train=False)
	test_loader = get_data_loader(test_dataset, _A.eval_batch_size, _A.cpu_workers, device)

	config = {"use_secttion": _A.use_section, "use_title": _A.use_title, "use_context": _A.use_context,
			  "use_url": _A.use_url, "use_doi": _A.use_doi}

	# bert_model = BertClassifier(model, _C.MODEL.BERT_OUT_DIM, train_dataset.get_class_num(), train_dataset.get_section_num(), train_dataset.get_journal_num(), train_dataset.get_DOI_num(), _C.MODEL.BERT_DROPOUT).to(device)
	if train_dataset.journal_json and train_dataset.DOI_json:
		bert_model = BertClassifier(model, config, train_dataset.get_class_num(), train_dataset.get_section_num(), train_dataset.get_journal_num(), train_dataset.get_DOI_num()).to(device)
	else:
		bert_model = BertClassifier(model, config, train_dataset.get_class_num(), train_dataset.get_section_num()).to(device)

	# optimizer = AdamW(bert_model.parameters(), lr=_C.OPTIM.LR, correct_bias=False)
	# num_training_steps = (len(train_loader) / _C.OPTIM.ACCUMULATION_STEPS) * _C.OPTIM.NUM_EPOCH
	# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=_C.OPTIM.NUM_WARMUP_STEPS, num_training_steps=num_training_steps)

	optimizer = AdamW(bert_model.parameters(), lr=_A.lr, correct_bias=False)
	num_training_steps = (len(train_loader) / _A.accumulation_steps) * _A.epochs
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=_A.warmup_steps, num_training_steps=num_training_steps)

	best_macro_f1 = 0.0
	test_macro_f1 = 0.0
	test_f1 = None
	# for i in range(_C.OPTIM.NUM_EPOCH):
	for i in range(_A.epochs):
		print('Epoch %d' % i)

		bert_model.train()
		for step, batch in enumerate(tqdm(train_loader)):
			# logits = bert_model(input_ids=batch['text'],
			# 					title_ids=batch['title_text'],
			# 					title_attention_mask=batch['title_mask'],
			# 					title_token_type_ids=batch['title_seg_ids'],
			# 					context_ids=batch['context_text'],
			# 					context_attention_mask=batch['context_mask'],
			# 					context_token_type_ids=batch['context_seg_ids'],
			# 					sec_ids=batch['sec_names'],
			# 					jour_ids=batch['jour_names'],
			# 					doi_ids=batch['doi_names'],
			# 					attention_mask=batch['mask'],
			# 					token_type_ids=batch['seg_ids'])

			loss = bert_model.cal_loss(logits, batch['labels'])
			# loss = loss / _C.OPTIM.ACCUMULATION_STEPS
			loss = loss / _A.accumulation_steps
			loss.backward()
			# torch.nn.utils.clip_grad_norm_(model.parameters(), _C.OPTIM.MAX_GRAD_NORM)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
			torch.nn.utils.clip_grad_norm_(model.parameters(), _A.max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
			
			# if (step+1) % _C.OPTIM.ACCUMULATION_STEPS == 0:
			if (step + 1) % _A.accumulation_steps == 0:
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad()

		bert_model.eval()
		prediction_list = []
		ground_truth_list = []
		for batch in tqdm(dev_loader):
			logits = bert_model(input_ids=batch['text'],
								title_ids=batch['title_text'],
								title_attention_mask=batch['title_mask'],
								title_token_type_ids=batch['title_seg_ids'],
								context_ids=batch['context_text'],
								context_attention_mask=batch['context_mask'],
								context_token_type_ids=batch['context_seg_ids'],
								sec_ids=batch['sec_names'],
								jour_ids=batch['jour_names'],
								doi_ids=batch['doi_names'],
								attention_mask=batch['mask'],
								token_type_ids=batch['seg_ids'])
			predictions = bert_model.prediction(logits)
			prediction_list += predictions.cpu().numpy().tolist()
			ground_truth_list += batch['labels'].cpu().numpy().tolist()
		F1_dict = eval_model(prediction_list, ground_truth_list, train_dataset.intent_json)
		
		if best_macro_f1 < F1_dict['ave_F1']:
			best_macro_f1 = F1_dict['ave_F1']
			prediction_list = []
			ground_truth_list = []
			for batch in tqdm(test_loader):
				logits = bert_model(input_ids=batch['text'],
									title_ids=batch['title_text'],
									title_attention_mask=batch['title_mask'],
									title_token_type_ids=batch['title_seg_ids'],
									context_ids=batch['context_text'],
									context_attention_mask=batch['context_mask'],
									context_token_type_ids=batch['context_seg_ids'],
									sec_ids=batch['sec_names'],
									jour_ids=batch['jour_names'],
									doi_ids=batch['doi_names'],
									attention_mask=batch['mask'],
									token_type_ids=batch['seg_ids'])
				predictions = bert_model.prediction(logits)
				prediction_list += predictions.cpu().numpy().tolist()
				ground_truth_list += batch['labels'].cpu().numpy().tolist()
			test_F1_dict = eval_model(prediction_list, ground_truth_list, train_dataset.intent_json)
			test_macro_f1 = test_F1_dict['ave_F1']
			test_f1 = test_F1_dict

		print('Macro F1 %.2f Best F1 %.2f Test F1 %.2f' % (F1_dict['ave_F1'], best_macro_f1, test_macro_f1))
		print(test_f1)


