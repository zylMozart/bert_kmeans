import torch
from torch import nn
import json
import random
import numpy as np
from collections import namedtuple
from tempfile import TemporaryDirectory

from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from pytorch_pretrained_bert.modeling import BertForPreTraining
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from xkg import *

InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids is_next")

def convert_example_to_features(example, tokenizer, max_seq_length):
	tokens = example["tokens"]
	segment_ids = example["segment_ids"]
	is_random_next = example["is_random_next"]
	masked_lm_positions = example["masked_lm_positions"]
	masked_lm_labels = example["masked_lm_labels"]

	assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
	input_ids = tokenizer.convert_tokens_to_ids(tokens)
	masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

	input_array = np.zeros(max_seq_length, dtype=np.int)
	input_array[:len(input_ids)] = input_ids

	mask_array = np.zeros(max_seq_length, dtype=np.bool)
	mask_array[:len(input_ids)] = 1

	segment_array = np.zeros(max_seq_length, dtype=np.bool)
	segment_array[:len(segment_ids)] = segment_ids

	lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
	lm_label_array[masked_lm_positions] = masked_label_ids

	features = InputFeatures(input_ids=input_array,
							 input_mask=mask_array,
							 segment_ids=segment_array,
							 lm_label_ids=lm_label_array,
							 is_next=is_random_next)
	return features

class PregeneratedDataset(Dataset):
	def __init__(self, training_path, epoch, tokenizer, num_data_epochs, reduce_memory=False):
		self.vocab = tokenizer.vocab
		self.tokenizer = tokenizer
		self.epoch = epoch
		self.data_epoch = epoch % num_data_epochs
		data_file = training_path/f'epoch_{self.data_epoch}.json'
		metrics_file = training_path/f'epoch_{self.data_epoch}_metrics.json'
		assert data_file.isFile() and metrics_file.isFile()
		metrics = json.loads(metrics_file.file().read())
		num_samples = metrics['num_training_examples']
		seq_len = metrics['max_seq_len']
		self.temp_dir = None
		self.working_dir = None
		if reduce_memory:
			self.temp_dir = TemporaryDirectory()
			self.working_dir = Path(self.temp_dir.name)
			input_ids = np.memmap(filename=(self.working_dir/'input_ids.memmap').pathstr(),
								  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
			input_masks = np.memmap(filename=(self.working_dir/'input_masks.memmap').pathstr(),
									shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
			segment_ids = np.memmap(filename=(self.working_dir/'segment_ids.memmap').pathstr(),
									shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
			lm_label_ids = np.memmap(filename=(self.working_dir/'lm_label_ids.memmap').pathstr(),
									 shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
			lm_label_ids[:] = -1
			is_nexts = np.memmap(filename=(self.working_dir/'is_nexts.memmap').pathstr(),
								 shape=(num_samples,), mode='w+', dtype=np.bool)
		else:
			input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
			input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
			segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
			lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
			is_nexts = np.zeros(shape=(num_samples,), dtype=np.bool)
		pr(f'loading training examples for epoch {epoch}')
		i = NA
		with data_file.file().open('rb') as f:
			for i, line in enumerate(tqdm(f, total=num_samples, desc='training examples')):
				line = line.decode().strip()
				example = json.loads(line)
				features = convert_example_to_features(example, tokenizer, seq_len)
				input_ids[i] = features.input_ids
				segment_ids[i] = features.segment_ids
				input_masks[i] = features.input_mask
				lm_label_ids[i] = features.lm_label_ids
				is_nexts[i] = features.is_next
		assert i == num_samples - 1  # Assert that the sample count metric was true
		pr('loading complete!')
		self.num_samples = num_samples
		self.seq_len = seq_len
		self.input_ids = input_ids
		self.input_masks = input_masks
		self.segment_ids = segment_ids
		self.lm_label_ids = lm_label_ids
		self.is_nexts = is_nexts

	def __len__(self):
		return self.num_samples

	def __getitem__(self, item):
		return (torch.tensor(self.input_ids[item].astype(np.int64)),
				torch.tensor(self.input_masks[item].astype(np.int64)),
				torch.tensor(self.segment_ids[item].astype(np.int64)),
				torch.tensor(self.lm_label_ids[item].astype(np.int64)),
				torch.tensor(self.is_nexts[item].astype(np.int64)))

def main(dn, dev, batch_size, epochs):
	pregenerated_data = Dir(f'data/{dn}.pretrain.temp')
	output_dir = Dir(f'temp/{dn}.bert.pt')
	bert_model = 'bert-base-uncased'
	do_lower_case = TRUE
	reduce_memory = TRUE
	epochs = epochs
	local_rank = -1
	no_cuda = (dev=='cpu')
	gradient_accumulation_steps = 1
	train_batch_size = batch_size
	fp16 = FALSE
	loss_scale = 0
	warmup_proportion = 0.1
	learning_rate = 3e-5
	seed = 42

	samples_per_epoch = []
	for i in range(epochs):
		epoch_file = pregenerated_data/f'epoch_{i}.json'
		metrics_file = pregenerated_data/f'epoch_{i}_metrics.json'
		if epoch_file.isFile() and metrics_file.isFile():
			metrics = json.loads(metrics_file.file().read())
			samples_per_epoch.append(metrics['num_training_examples'])
		else:
			if i == 0: exit("No training data was found!")
			print(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({epochs}).")
			print("This script will loop over the available data, but training diversity may be negatively impacted.")
			num_data_epochs = i
			break
	else: num_data_epochs = epochs

	if no_cuda: device, n_gpu = 'cpu', 0
	elif local_rank==-1: device, n_gpu = 'cuda', torch.cuda.device_count()
	else:
		torch.cuda.set_device(local_rank)
		device, n_gpu = f'cuda:{local_rank}', 1
		# Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		torch.distributed.init_process_group(backend='nccl')
	pr(device=device, n_gpu=n_gpu, distributed=(local_rank!=-1), float16=fp16)

	train_batch_size = train_batch_size//gradient_accumulation_steps

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if n_gpu>0: torch.cuda.manual_seed_all(seed)

	tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

	total_train_examples = 0
	for i in range(epochs):
		# The modulo takes into account the fact that we may loop over limited epochs of data
		total_train_examples += samples_per_epoch[i%len(samples_per_epoch)]

	num_train_optimization_steps = int(
		total_train_examples/train_batch_size/gradient_accumulation_steps
	)
	if local_rank!=-1:
		num_train_optimization_steps = num_train_optimization_steps//torch.distributed.get_world_size()

	# Prepare model
	model = BertForPreTraining.from_pretrained(bert_model)
	if fp16: model.half()
	model.to(device)
	if local_rank != -1:
		try: from apex.parallel import DistributedDataParallel as DDP
		except ImportError:
			raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
		model = DDP(model)
	elif n_gpu > 1: model = nn.DataParallel(model)

	# Prepare optimizer
	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
		 'weight_decay': 0.01},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	]
	warmup_linear = NA
	if fp16:
		try:
			from apex.optimizers import FP16_Optimizer
			from apex.optimizers import FusedAdam
		except ImportError:
			raise ImportError(
				"Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

		optimizer = FusedAdam(optimizer_grouped_parameters,
							  lr=learning_rate,
							  bias_correction=False,
							  max_grad_norm=1.0)
		if loss_scale == 0:
			optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
		else:
			optimizer = FP16_Optimizer(optimizer, static_loss_scale=loss_scale)
		warmup_linear = WarmupLinearSchedule(warmup=warmup_proportion,
											 t_total=num_train_optimization_steps)
	else:
		optimizer = BertAdam(optimizer_grouped_parameters,
							 lr=learning_rate,
							 warmup=warmup_proportion,
							 t_total=num_train_optimization_steps)

	global_step = 0
	pr('***** Running training *****')
	pr(num_examples=total_train_examples)
	pr(batch_size=train_batch_size)
	pr(num_steps=num_train_optimization_steps)
	model.train()
	for epoch in range(epochs):
		epoch_dataset = PregeneratedDataset(
			epoch=epoch, training_path=pregenerated_data, tokenizer=tokenizer,
			num_data_epochs=num_data_epochs, reduce_memory=reduce_memory,
		)
		if local_rank == -1:
			train_sampler = RandomSampler(epoch_dataset)
		else:
			train_sampler = DistributedSampler(epoch_dataset)
		train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler,
			batch_size=train_batch_size)
		tr_loss = 0
		nb_tr_examples, nb_tr_steps = 0, 0
		with tqdm(total=len(train_dataloader), desc=f"epoch-{epoch}") as pbar:
			for step, batch in enumerate(train_dataloader):
				batch = tuple(t.to(device) for t in batch)
				input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch
				loss = model(input_ids, segment_ids, input_mask, lm_label_ids, is_next)
				if n_gpu > 1:
					loss = loss.mean() # mean() to average on multi-gpu.
				if gradient_accumulation_steps > 1:
					loss = loss / gradient_accumulation_steps
				if fp16:
					optimizer.backward(loss)
				else:
					loss.backward()
				tr_loss += loss.item()
				nb_tr_examples += input_ids.size(0)
				nb_tr_steps += 1
				pbar.update(1)
				mean_loss = tr_loss * gradient_accumulation_steps / nb_tr_steps
				pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")
				if (step + 1) % gradient_accumulation_steps == 0:
					if fp16:
						# modify learning rate with special warm up BERT uses
						# if fp16 is False, BertAdam is used that handles this automatically
						lr_this_step = learning_rate * warmup_linear.get_lr(global_step/num_train_optimization_steps,
																				 warmup_proportion)
						for param_group in optimizer.param_groups:
							param_group['lr'] = lr_this_step
					optimizer.step()
					optimizer.zero_grad()
					global_step += 1

	# Save a trained model
	pr('***** Saving fine-tuned model *****')
	model_to_save = model.module if hasattr(model, 'module') else model # Only save the model it-self
	output_model_file = output_dir.add().div('pytorch_model.bin').file()
	torch.save(model_to_save.state_dict(), output_model_file.pathstr())

pr.params({'exts': '0'}).file(NA)
main('so', 'cpu', 16, 12)
