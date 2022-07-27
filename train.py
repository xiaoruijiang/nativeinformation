from datasets import SciDataset, ACLDataset, get_data_loader, TARGET_CITATION, eval_model
from bert_text_classifier import BertClassifier
from transformers import BertModel,BertTokenizer
from transformers.optimization import get_linear_schedule_with_warmup, AdamW
import torch
import argparse
import numpy as np
from tqdm import tqdm
# from config import Config

parser = argparse.ArgumentParser("Train Bert for Citation Functions")
parser.add_argument(
    "--gpu-ids", required=True, nargs="+", type=int, help="List of GPU IDs to use (-1 for CPU)."
)
parser.add_argument(
    "--cpu-workers", type=int, default=0, help="Number of CPU workers to use for data loading."
)
parser.add_argument(
    "--serialization-dir",
    default="./checkpoints/experiment",
    help="Path to a (non-existent) directory for serializing checkpoints and tensorboard logs.",
)
parser.add_argument(
    "--bert-dir",
    default="allenai/scibert_scivocab_uncased"
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
parser.add_argument('--accumulation_steps',  type=int, default=5)
parser.add_argument('--lr_warmup',  type=float, default=0.1)
parser.add_argument('--epochs',  type=int, default=10)
parser.add_argument('--max_grad_norm',  type=float, default=1.0)
parser.add_argument('--dropout',  type=float, default=0.1)

import os
data_root = "./data/"
data_root = "./data_one_citstr_per_citseg/"
data_root = "./jiang2021_data_one_citstr_per_citseg/processed_final_one_citstr_per_citseg/citation_function_11class/"
# data_root = "./jiang2021_data_one_citstr_per_citseg/processed_final_one_citstr_per_citseg/citation_function_6class_jurgens/"
# data_root = "./jiang2021_data_one_citstr_per_citseg/processed_final_one_citstr_per_citseg/citation_function_9class/"
# data_root = "./jiang2021_data_one_citstr_per_citseg/processed_final_one_citstr_per_citseg/citation_function_7class/"

parser.add_argument('--train_path',  type=str, default=os.path.join(data_root, "dataset_updated_final_processed_final_train_v5.jsonl"))
parser.add_argument('--dev_path',  type=str, default=os.path.join(data_root, "dataset_updated_final_processed_final_valid_v5.jsonl"))
parser.add_argument('--test_path',  type=str, default=os.path.join(data_root, "dataset_updated_final_processed_final_test_v5.jsonl"))
# parser.add_argument('--train_path',  type=str, default="dataset_updated_final_train_v5.jsonl")
# parser.add_argument('--dev_path',  type=str, default="dataset_updated_final_dev_v5.jsonl")
# parser.add_argument('--test_path',  type=str, default="dataset_updated_final_test_v5.jsonl")
parser.add_argument('--intent_path',  type=str, default=os.path.join(data_root, "intent.json"))
parser.add_argument('--section_path',  type=str, default=os.path.join(data_root, "section.json"))

parser.add_argument("--use_section", action='store_true', default=False)
parser.add_argument("--use_title", action='store_true', default=False)
parser.add_argument("--use_context", action='store_true', default=False)
parser.add_argument("--use_url", action='store_true', default=False)
parser.add_argument("--use_doi", action='store_true', default=False)



if __name__ == "__main__":
	# --------------------------------------------------------------------------------------------
	#   INPUT ARGUMENTS AND CONFIG
	# --------------------------------------------------------------------------------------------
	run_args = parser.parse_args()
	# _C = Config(run_args.config, run_args.config_override)

	np.random.seed(run_args.seed)
	torch.manual_seed(run_args.seed)
	torch.cuda.manual_seed_all(run_args.seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

	device = torch.device(f"cuda:{run_args.gpu_ids[0]}" if run_args.gpu_ids[0] >= 0 else "cpu")

	tokenizer = BertTokenizer.from_pretrained(run_args.bert_dir)
	model = BertModel.from_pretrained(run_args.bert_dir)

	tokenizer.add_tokens([TARGET_CITATION])
	model.resize_token_embeddings(len(tokenizer))

	# D = SciDataset if _C.DATA.TYPE == 'sci' else ACLDataset
	Dataset = SciDataset if run_args.data_type == 'sci' else ACLDataset

	print("loading training dataset ...")
	# train_dataset = Dataset(_C.DATA.TRAIN_PATH, _C.DATA.LABEL_PATH, _C.DATA.CITATION_LABEL_PATH, _C.DATA.URL_LABEL_PATH, _C.DATA.DOI_LABEL_PATH, tokenizer, is_train=True)
	# train_loader = get_data_loader(train_dataset, _C.OPTIM.BATCH_SIZE, run_args.cpu_workers, device)
	train_dataset = Dataset(tokenizer, run_args.train_path, run_args.intent_path, run_args.section_path, is_train=True)
	train_loader = get_data_loader(train_dataset, run_args.batch_size, run_args.cpu_workers, device, run_args.use_url, run_args.use_doi)

	print("loading development dataset ...")
	# dev_dataset = Dataset(_C.DATA.DEV_PATH, _C.DATA.LABEL_PATH, _C.DATA.CITATION_LABEL_PATH, _C.DATA.URL_LABEL_PATH, _C.DATA.DOI_LABEL_PATH, tokenizer)
	# dev_loader = get_data_loader(dev_dataset, _C.OPTIM.EVAL_BATCH_SIZE, run_args.cpu_workers, device)
	dev_dataset = Dataset(tokenizer, run_args.dev_path, run_args.intent_path, run_args.section_path, is_train=False)
	dev_loader = get_data_loader(dev_dataset, run_args.eval_batch_size, run_args.cpu_workers, device, run_args.use_url, run_args.use_doi)

	print("loading test dataset ...")
	# test_dataset = Dataset(_C.DATA.TEST_PATH, _C.DATA.LABEL_PATH, _C.DATA.CITATION_LABEL_PATH, _C.DATA.URL_LABEL_PATH, _C.DATA.DOI_LABEL_PATH, tokenizer)
	# test_loader = get_data_loader(test_dataset, _C.OPTIM.EVAL_BATCH_SIZE, run_args.cpu_workers, device)
	test_dataset = Dataset(tokenizer, run_args.test_path, run_args.intent_path, run_args.section_path, is_train=False)
	test_loader = get_data_loader(test_dataset, run_args.eval_batch_size, run_args.cpu_workers, device, run_args.use_url, run_args.use_doi)

	# config = {"use_secttion": run_args.use_section, "use_title": run_args.use_title, "use_context": run_args.use_context,
	# 		  "use_url": run_args.use_url, "use_doi": run_args.use_doi}

	# bert_model = BertClassifier(model, _C.MODEL.BERT_OUT_DIM, train_dataset.get_class_num(), train_dataset.get_section_num(), train_dataset.get_journal_num(), train_dataset.get_DOI_num(), _C.MODEL.BERT_DROPOUT).to(device)
	if train_dataset.journal_json and train_dataset.DOI_json:
		bert_model = BertClassifier(model, run_args, train_dataset.get_class_num(), train_dataset.get_section_num(), train_dataset.get_journal_num(), train_dataset.get_DOI_num()).to(device)
	else:
		bert_model = BertClassifier(model, run_args, train_dataset.get_class_num(), train_dataset.get_section_num()).to(device)

	# optimizer = AdamW(bert_model.parameters(), lr=_C.OPTIM.LR, correct_bias=False)
	# num_training_steps = (len(train_loader) / _C.OPTIM.ACCUMULATION_STEPS) * _C.OPTIM.NUM_EPOCH
	# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=_C.OPTIM.NUM_WARMUP_STEPS, num_training_steps=num_training_steps)

	optimizer = AdamW(bert_model.parameters(), lr=run_args.lr, correct_bias=False)
	num_training_steps = (len(train_loader) / run_args.accumulation_steps) * run_args.epochs
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=run_args.lr_warmup * num_training_steps,
												num_training_steps=num_training_steps)
	
	best_macro_f1 = 0.0
	test_macro_f1 = 0.0
	test_f1 = None
	# for i in range(_C.OPTIM.NUM_EPOCH):
	for i in range(run_args.epochs):
		print('Epoch %d' % i)

		bert_model.train()
		for step, batch in enumerate(tqdm(train_loader)):
			if run_args.use_url and run_args.use_doi:
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
			else:
				logits = bert_model(input_ids=batch['text'],
									title_ids=batch['title_text'],
									title_attention_mask=batch['title_mask'],
									title_token_type_ids=batch['title_seg_ids'],
									context_ids=batch['context_text'],
									context_attention_mask=batch['context_mask'],
									context_token_type_ids=batch['context_seg_ids'],
									sec_ids=batch['sec_names'],
									attention_mask=batch['mask'],
									token_type_ids=batch['seg_ids'])
			loss = bert_model.cal_loss(logits, batch['labels'])
			# loss = loss / _C.OPTIM.ACCUMULATION_STEPS
			loss = loss / run_args.accumulation_steps
			loss.backward()
			# torch.nn.utils.clip_grad_norm_(model.parameters(), _C.OPTIM.MAX_GRAD_NORM)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
			torch.nn.utils.clip_grad_norm_(model.parameters(), run_args.max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
			
			# if (step+1) % _C.OPTIM.ACCUMULATION_STEPS == 0:
			if (step + 1) % run_args.accumulation_steps == 0:
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad()

		bert_model.eval()
		prediction_list = []
		ground_truth_list = []
		for batch in tqdm(dev_loader):
			if dev_dataset.journal_json and dev_dataset.DOI_json:
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
			else:
				logits = bert_model(input_ids=batch['text'],
									title_ids=batch['title_text'],
									title_attention_mask=batch['title_mask'],
									title_token_type_ids=batch['title_seg_ids'],
									context_ids=batch['context_text'],
									context_attention_mask=batch['context_mask'],
									context_token_type_ids=batch['context_seg_ids'],
									sec_ids=batch['sec_names'],
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
				if test_dataset.journal_json and test_dataset.DOI_json:
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
				else:
					logits = bert_model(input_ids=batch['text'],
										title_ids=batch['title_text'],
										title_attention_mask=batch['title_mask'],
										title_token_type_ids=batch['title_seg_ids'],
										context_ids=batch['context_text'],
										context_attention_mask=batch['context_mask'],
										context_token_type_ids=batch['context_seg_ids'],
										sec_ids=batch['sec_names'],
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


