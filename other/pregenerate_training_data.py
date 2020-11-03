from tqdm import trange
from random import random, randrange, randint, shuffle, choice, sample
from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np
import json

from xkg import *

class DocumentDatabase:
	def __init__(self):
		self.documents = NA
		self.doc_lengths = NA
		self.doc_cumsum = NA
		self.cumsum_max = NA

	def set_docs(self, docs):
		self.documents = docs
		self.doc_lengths = [len(doc) for doc in docs]

	def _precalculate_doc_weights(self):
		self.doc_cumsum = np.cumsum(self.doc_lengths)
		self.cumsum_max = self.doc_cumsum[-1]

	def sample_doc(self, current_idx, sentence_weighted=True):
		# Uses the current iteration counter to ensure we don't sample the same doc twice
		if sentence_weighted:
			# With sentence weighting, we sample docs proportionally to their sentence length
			if self.doc_cumsum is None or len(self.doc_cumsum) != len(self.doc_lengths):
				self._precalculate_doc_weights()
			rand_start = self.doc_cumsum[current_idx]
			rand_end = rand_start + self.cumsum_max - self.doc_lengths[current_idx]
			sentence_index = randrange(rand_start, rand_end) % self.cumsum_max
			sampled_doc_index = np.searchsorted(self.doc_cumsum, sentence_index, side='right')
		else:
			# If we don't use sentence weighting, then every doc has an equal chance to be chosen
			sampled_doc_index = (current_idx + randrange(1, len(self.doc_lengths))) % len(self.doc_lengths)
		assert sampled_doc_index != current_idx
		return self.documents[sampled_doc_index]

	def __len__(self):
		return len(self.doc_lengths)

	def __getitem__(self, item):
		return self.documents[item]

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
	"""Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_num_tokens:
			break

		trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
		assert len(trunc_tokens) >= 1

		# We want to sometimes truncate from the front and sometimes from the
		# back to add more randomness and avoid biases.
		if random() < 0.5:
			del trunc_tokens[0]
		else:
			trunc_tokens.pop()

def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_list):
	"""Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
	with several refactors to clean it up and remove a lot of unnecessary variables."""
	cand_indices = []
	for (i, token) in enumerate(tokens):
		if token == "[CLS]" or token == "[SEP]":
			continue
		cand_indices.append(i)

	num_to_mask = min(max_predictions_per_seq,
					  max(1, int(round(len(tokens) * masked_lm_prob))))
	shuffle(cand_indices)
	mask_indices = sorted(sample(cand_indices, num_to_mask))
	masked_token_labels = []
	for index in mask_indices:
		# 80% of the time, replace with [MASK]
		if random() < 0.8:
			masked_token = "[MASK]"
		else:
			# 10% of the time, keep original
			if random() < 0.5:
				masked_token = tokens[index]
			# 10% of the time, replace with random word
			else:
				masked_token = choice(vocab_list)
		masked_token_labels.append(tokens[index])
		# Once we've saved the true label for that token, we can overwrite it with the masked version
		tokens[index] = masked_token

	return tokens, mask_indices, masked_token_labels

def create_instances_from_document(
		doc_database, doc_idx, max_seq_length, short_seq_prob,
		masked_lm_prob, max_predictions_per_seq, vocab_list):
	"""This code is mostly a duplicate of the equivalent function from Google BERT's repo.
	However, we make some changes and improvements. Sampling is improved and no longer requires a loop in this function.
	Also, documents are sampled proportionally to the number of sentences they contain, which means each sentence
	(rather than each document) has an equal chance of being sampled as a false example for the NextSentence task."""
	document = doc_database[doc_idx]
	# Account for [CLS], [SEP], [SEP]
	max_num_tokens = max_seq_length - 3

	# We *usually* want to fill up the entire sequence since we are padding
	# to `max_seq_length` anyways, so short sequences are generally wasted
	# computation. However, we *sometimes*
	# (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
	# sequences to minimize the mismatch between pre-training and fine-tuning.
	# The `target_seq_length` is just a rough target however, whereas
	# `max_seq_length` is a hard limit.
	target_seq_length = max_num_tokens
	if random() < short_seq_prob:
		target_seq_length = randint(2, max_num_tokens)

	# We DON'T just concatenate all of the tokens from a document into a long
	# sequence and choose an arbitrary split point because this would make the
	# next sentence prediction task too easy. Instead, we split the input into
	# segments "A" and "B" based on the actual "sentences" provided by the user
	# input.
	instances = []
	current_chunk = []
	current_length = 0
	i = 0
	while i < len(document):
		segment = document[i]
		current_chunk.append(segment)
		current_length += len(segment)
		if i == len(document) - 1 or current_length >= target_seq_length:
			if current_chunk:
				# `a_end` is how many segments from `current_chunk` go into the `A`
				# (first) sentence.
				a_end = 1
				if len(current_chunk) >= 2:
					a_end = randrange(1, len(current_chunk))

				tokens_a = []
				for j in range(a_end):
					tokens_a.extend(current_chunk[j])

				tokens_b = []

				# Random next
				if len(current_chunk) == 1 or random() < 0.5:
					is_random_next = True
					target_b_length = target_seq_length - len(tokens_a)

					# Sample a random document, with longer docs being sampled more frequently
					random_document = doc_database.sample_doc(current_idx=doc_idx, sentence_weighted=True)

					random_start = randrange(0, len(random_document))
					for j in range(random_start, len(random_document)):
						tokens_b.extend(random_document[j])
						if len(tokens_b) >= target_b_length:
							break
					# We didn't actually use these segments so we "put them back" so
					# they don't go to waste.
					num_unused_segments = len(current_chunk) - a_end
					i -= num_unused_segments
				# Actual next
				else:
					is_random_next = False
					for j in range(a_end, len(current_chunk)):
						tokens_b.extend(current_chunk[j])
				truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

				assert len(tokens_a) >= 1
				assert len(tokens_b) >= 1

				tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
				# The segment IDs are 0 for the [CLS] token, the A tokens and the first [SEP]
				# They are 1 for the B tokens and the final [SEP]
				segment_ids = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]

				tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
					tokens, masked_lm_prob, max_predictions_per_seq, vocab_list)

				instance = {
					"tokens": tokens,
					"segment_ids": segment_ids,
					"is_random_next": is_random_next,
					"masked_lm_positions": masked_lm_positions,
					"masked_lm_labels": masked_lm_labels}
				instances.append(instance)
			current_chunk = []
			current_length = 0
		i += 1

	return instances

def main(dn, epochs, max_seq_len):
	train_corpus = File(f'data/{dn}.pretrain.tok')
	output_dir = Dir(f'data/{dn}.pretrain.temp')
	bert_model = 'bert-base-uncased'
	do_lower_case = TRUE
	epochs_to_generate = epochs
	short_seq_prob = 0.1
	masked_lm_prob = 0.15
	max_predictions_per_seq = 20

	tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
	vocab_list = list(tokenizer.vocab.keys())
	docdb = DocumentDatabase()
	docs = train_corpus.load()

	docdb.set_docs(docs)

	output_dir.add()
	for epoch in trange(epochs_to_generate, desc='epoch'):
		epoch_filename = (output_dir/f'epoch_{epoch}.json').file()
		num_instances = 0
		with epoch_filename.open('wb') as epoch_file:
			for doc_idx in trange(len(docdb), desc='document'):
				doc_instances = create_instances_from_document(
					docdb, doc_idx, max_seq_length=max_seq_len, short_seq_prob=short_seq_prob,
					masked_lm_prob=masked_lm_prob, max_predictions_per_seq=max_predictions_per_seq,
					vocab_list=vocab_list)
				doc_instances = [json.dumps(instance) for instance in doc_instances]
				for instance in doc_instances:
					epoch_file.write((instance+'\n').encode())
					num_instances += 1
		metrics_file = (output_dir/f'epoch_{epoch}_metrics.json').file()
		with metrics_file.open('wb') as metrics_file:
			metrics = {
				"num_training_examples": num_instances,
				"max_seq_len": max_seq_len
			}
			metrics_file.write(json.dumps(metrics).encode())

main('so', 12, 128)
