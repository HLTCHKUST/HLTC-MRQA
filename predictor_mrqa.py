# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import collections
import os
import time
import math
import json
import six
import random
import gc
import sys

import numpy as np

if six.PY2:
  import cPickle as pickle
else:
  import pickle
import jsonlines

import tensorflow as tf
import sentencepiece as spm
from src.prepro_utils import preprocess_text, encode_ids, encode_pieces, printable_text
import src.function_builder as function_builder
import src.model_utils as model_utils
import src.mrqa_utils
from src.data_utils import SEP_ID, CLS_ID, VOCAB_SIZE

SPIECE_UNDERLINE = u'▁'

SEG_ID_P   = 0
SEG_ID_Q   = 1
SEG_ID_CLS = 2
SEG_ID_PAD = 3

# Preprocessing
flags.DEFINE_bool("do_prepro", default=False,
      help="Perform preprocessing only.")
flags.DEFINE_integer("num_proc", default=16,
      help="Number of preprocessing processes.")
flags.DEFINE_integer("proc_id", default=0,
      help="Process id for preprocessing.")

# Model
flags.DEFINE_string("model_config_path", default=None,
      help="Model config path.")
flags.DEFINE_float("dropout", default=0.1,
      help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
      help="Attention dropout rate.")
flags.DEFINE_integer("clamp_len", default=-1,
      help="Clamp length.")
flags.DEFINE_string("summary_type", default="last",
      help="Method used to summarize a sequence into a vector.")
flags.DEFINE_bool("use_bfloat16", default=False,
      help="Whether to use bfloat16.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
                  enum_values=["normal", "uniform"],
                  help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
                   help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
                   help="Initialization std when init is uniform.")

# I/O paths
flags.DEFINE_bool("overwrite_data", default=False,
                  help="If False, will use cached data if available.")
flags.DEFINE_string("init_checkpoint", default=None,
                    help="checkpoint path for initializing the model. "
                    "Could be a pretrained model or a finetuned model.")
flags.DEFINE_bool("init_global_vars", default=False,
                  help="If true, init all global vars. If false, init "
                  "trainable vars only.")
flags.DEFINE_string("output_dir", default="",
                    help="Output dir for TF records.")
flags.DEFINE_string("predict_dir", default="",
                    help="Dir for predictions.")
flags.DEFINE_string("spiece_model_file", default="",
                    help="Sentence Piece model path.")
flags.DEFINE_string("model_dir", default="",
                    help="Directory for saving the finetuned model.")
flags.DEFINE_string("train_dir", default="",
                    help="Path of train directory.")
flags.DEFINE_string("dev_dir", default="",
                    help="Path of development directory.")

# Data preprocessing config
flags.DEFINE_integer("max_seq_length",
                     default=512, help="Max sequence length")
flags.DEFINE_integer("max_query_length",
                     default=64, help="Max query length")
flags.DEFINE_integer("doc_stride",
                     default=128, help="Doc stride")
flags.DEFINE_integer("max_answer_length",
                     default=64, help="Max answer length")
flags.DEFINE_bool("uncased", default=False, help="Use uncased data.")

# TPUs and machines
flags.DEFINE_bool("use_tpu", default=False, help="whether to use TPU.")
flags.DEFINE_integer("num_hosts", default=1, help="How many TPU hosts.")
flags.DEFINE_integer("num_core_per_host", default=8,
      help="8 for TPU v2 and v3-8, 16 for larger TPU v3 pod. In the context "
      "of GPU training, it refers to the number of GPUs used.")
flags.DEFINE_string("tpu_job_name", default=None, help="TPU worker job name.")
flags.DEFINE_string("tpu", default=None, help="TPU name.")
flags.DEFINE_string("tpu_zone", default=None, help="TPU zone.")
flags.DEFINE_string("gcp_project", default=None, help="gcp project.")
flags.DEFINE_string("master", default=None, help="master")
flags.DEFINE_integer("iterations", default=1000,
                     help="number of iterations per TPU training loop.")

# Training
flags.DEFINE_bool("do_train", default=True, help="whether to do training")
flags.DEFINE_integer("train_batch_size", default=48,
                     help="batch size for training")
flags.DEFINE_integer("train_steps", default=8000,
                     help="Number of training steps")
flags.DEFINE_integer("warmup_steps", default=0, help="number of warmup steps")
flags.DEFINE_integer("save_steps", default=None,
                     help="Save the model for every save_steps. "
                     "If None, not to save any model.")
flags.DEFINE_integer("max_save", default=0,
                     help="Max number of checkpoints to save. "
                     "Use 0 to save all.")
flags.DEFINE_integer("shuffle_buffer", default=2048,
                     help="Buffer size used for shuffle.")

# Optimization
flags.DEFINE_float("learning_rate", default=3e-5, help="initial learning rate")
flags.DEFINE_float("min_lr_ratio", default=0.0,
                   help="min lr ratio for cos decay.")
flags.DEFINE_float("clip", default=1.0, help="Gradient clipping")
flags.DEFINE_float("weight_decay", default=0.00, help="Weight decay rate")
flags.DEFINE_float("adam_epsilon", default=1e-6, help="Adam epsilon")
flags.DEFINE_string("decay_method", default="poly", help="poly or cos")
flags.DEFINE_float("lr_layer_decay_rate", default=0.75,
                   help="Top layer: lr[L] = FLAGS.learning_rate."
                   "Lower layers: lr[l-1] = lr[l] * lr_layer_decay_rate.")

# Eval / Prediction
flags.DEFINE_bool("do_predict", default=False, help="whether to do predict")
flags.DEFINE_integer("predict_batch_size", default=32,
                     help="batch size for prediction")
flags.DEFINE_integer("n_best_size", default=5,
                     help="n best size for predictions")
flags.DEFINE_integer("start_n_top", default=5, help="Beam size for span start.")
flags.DEFINE_integer("end_n_top", default=5, help="Beam size for span end.")
flags.DEFINE_string("target_eval_key", default="best_f1",
                    help="Use has_ans_f1 for Model I.")
flags.DEFINE_string("export_dir_base", default="",
                    help="Path of exported models.")


FLAGS = flags.FLAGS

class MRQAExample(object):
  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               qas_id,
               question_text,
               paragraph_text,
               orig_answer_text=None,
               start_position=None,
               send_position=None,
               is_impossible=False):
    self.qas_id = qas_id
    self.question_text = question_text
    self.paragraph_text = paragraph_text
    self.orig_answer_text = orig_answer_text
    self.start_position = start_position
    self.send_position = send_position
    self.is_impossible = is_impossible

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (printable_text(self.qas_id))
    s += ", question_text: %s" % (
        printable_text(self.question_text))
    s += ", paragraph_text: [%s]" % (" ".join(self.paragraph_text))
    if self.start_position:
      s += ", start_position: %d" % (self.start_position)
    if self.start_position:
      s += ", is_impossible: %r" % (self.is_impossible)
    if self.start_position:
      s += ", send_position: %d" % (self.send_position)
    return s

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tok_start_to_orig_index,
               tok_end_to_orig_index,
               token_is_max_context,
               input_ids,
               input_mask,
               p_mask,
               segment_ids,
               paragraph_len,
               cls_index,
               start_position=None,
               end_position=None,
               is_impossible=None):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tok_start_to_orig_index = tok_start_to_orig_index # paragraph
    self.tok_end_to_orig_index = tok_end_to_orig_index     # paragraph
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids                             # context+question
    self.input_mask = input_mask
    self.p_mask = p_mask
    self.segment_ids = segment_ids
    self.paragraph_len = paragraph_len
    self.cls_index = cls_index
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible

def read_mrqa_data(input_file, is_training):
  """Read a QA data jsonl file into a list of Examples."""
  with tf.gfile.Open(input_file, "r") as reader:
    input_data = jsonlines.Reader(reader)

  examples = []
  for entry in input_data:
    if u'header' in entry:
      continue

    assert type(entry) == dict
    assert u'context' in entry
    assert u'qas' in entry

    paragraph_text = entry["context"]

    for qa in entry["qas"]:
      assert u'qid' in qa
      assert u'question' in qa
      assert u'detected_answers' in qa
      assert u'text' in qa[u'detected_answers'][0]
      assert u'char_spans' in qa[u'detected_answers'][0]

      qas_id = qa["qid"]
      question_text = qa["question"]
      start_position = None
      send_position = None
      orig_answer_text = None
      is_impossible = False
      
      if is_training:
        is_impossible = False if qa["detected_answers"]!=[] else True
        # if (len(qa["answers"]) != 1) and (not is_impossible):  # TriviaQA may have several answers, choose the first one
        #     raise ValueError(
        #         "For training, each question should have exactly 1 answer.")
        if not is_impossible:
          answer = qa["detected_answers"][0]
          orig_answer_text = answer["text"]
          start_position = answer["char_spans"][0][0]
          send_position = answer["char_spans"][0][1]
          assert type(start_position)==int
        else:
          print(qa["answers"])
          print("="*80)
          start_position = -1
          send_position = -1
          orig_answer_text = ""        

      example = MRQAExample(
          qas_id=qas_id,
          question_text=question_text,
          paragraph_text=paragraph_text,
          orig_answer_text=orig_answer_text,
          start_position=start_position,
          send_position=send_position,
          is_impossible=is_impossible)
      examples.append(example)
  return examples

def arrange_mrqa_data(input_data, is_training):
  """Read a QA data jsonl file into a list of Examples."""
  examples = []
  entry = input_data

  assert type(entry) == dict
  assert u'context' in entry
  assert u'qas' in entry

  paragraph_text = entry["context"]
  print(input_data)
  print("="*80)
  print(len(entry["qas"]))
  for qa in entry["qas"]:
    assert u'qid' in qa
    assert u'question' in qa
    assert u'detected_answers' in qa
    assert u'text' in qa[u'detected_answers'][0]
    assert u'char_spans' in qa[u'detected_answers'][0]

    qas_id = qa["qid"]
    question_text = qa["question"]
    start_position = None
    send_position = None
    orig_answer_text = None
    is_impossible = False
    
    if is_training:
      is_impossible = False if qa["detected_answers"]!=[] else True
      # if (len(qa["answers"]) != 1) and (not is_impossible):  # TriviaQA may have several answers, choose the first one
      #     raise ValueError(
      #         "For training, each question should have exactly 1 answer.")
      if not is_impossible:
        answer = qa["detected_answers"][0]
        orig_answer_text = answer["text"]
        start_position = answer["char_spans"][0][0]
        send_position = answer["char_spans"][0][1]
        assert type(start_position)==int
      else:
        start_position = -1
        send_position = -1
        orig_answer_text = ""        

    example = MRQAExample(
        qas_id=qas_id,
        question_text=question_text,
        paragraph_text=paragraph_text,
        orig_answer_text=orig_answer_text,
        start_position=start_position,
        send_position=send_position,
        is_impossible=is_impossible)
    examples.append(example)
    print(len(examples))
    print("="*80)
  return examples

def _convert_index(index, pos, M=None, is_start=True):
  if index[pos] is not None:
    return index[pos]
  N = len(index)
  rear = pos
  while rear < N - 1 and index[rear] is None:
    rear += 1
  front = pos
  while front > 0 and index[front] is None:
    front -= 1
  assert index[front] is not None or index[rear] is not None
  if index[front] is None:
    if index[rear] >= 1:
      if is_start:
        return 0
      else:
        return index[rear] - 1
    return index[rear]
  if index[rear] is None:
    if M is not None and index[front] < M - 1:
      if is_start:
        return index[front] + 1
      else:
        return M - 1
    return index[front]
  if is_start:
    if index[rear] > index[front] + 1:
      return index[front] + 1
    else:
      return index[rear]
  else:
    if index[rear] > index[front] + 1:
      return index[rear] - 1
    else:
      return index[front]

def convert_examples_to_features(examples, sp_model, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn):
  """Loads a data file into a list of `InputBatch`s."""

  cnt_pos, cnt_neg = 0, 0
  unique_id = 1000000000
  max_N, max_M = 1024, 1024
  f = np.zeros((max_N, max_M), dtype=np.float32)

  for (example_index, example) in enumerate(examples):

    if example_index % 100 == 0:
      tf.logging.info('Converting {}/{} pos {} neg {}'.format(
          example_index, len(examples), cnt_pos, cnt_neg))

    query_tokens = encode_ids(
        sp_model,
        preprocess_text(example.question_text, lower=FLAGS.uncased))

    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]

    paragraph_text = example.paragraph_text
    para_tokens = encode_pieces(
        sp_model,
        preprocess_text(example.paragraph_text, lower=FLAGS.uncased))

    chartok_to_tok_index = []
    tok_start_to_chartok_index = []
    tok_end_to_chartok_index = []
    char_cnt = 0
    for i, token in enumerate(para_tokens):
      chartok_to_tok_index.extend([i] * len(token))
      tok_start_to_chartok_index.append(char_cnt)
      char_cnt += len(token)
      tok_end_to_chartok_index.append(char_cnt - 1)

    tok_cat_text = ''.join(para_tokens).replace(SPIECE_UNDERLINE, ' ')
    N, M = len(paragraph_text), len(tok_cat_text)

    if N > max_N or M > max_M:
      max_N = max(N, max_N)
      max_M = max(M, max_M)
      f = np.zeros((max_N, max_M), dtype=np.float32)
      gc.collect()

    g = {}

    def _lcs_match(max_dist):
      f.fill(0)
      g.clear()

      ### longest common sub sequence
      # f[i, j] = max(f[i - 1, j], f[i, j - 1], f[i - 1, j - 1] + match(i, j))
      for i in range(N):

        # note(zhiliny):
        # unlike standard LCS, this is specifically optimized for the setting
        # because the mismatch between sentence pieces and original text will
        # be small
        for j in range(i - max_dist, i + max_dist):
          if j >= M or j < 0: continue

          if i > 0:
            g[(i, j)] = 0
            f[i, j] = f[i - 1, j]

          if j > 0 and f[i, j - 1] > f[i, j]:
            g[(i, j)] = 1
            f[i, j] = f[i, j - 1]

          f_prev = f[i - 1, j - 1] if i > 0 and j > 0 else 0
          if (preprocess_text(paragraph_text[i], lower=FLAGS.uncased,
              remove_space=False)
              == tok_cat_text[j]
              and f_prev + 1 > f[i, j]):
            g[(i, j)] = 2
            f[i, j] = f_prev + 1

    max_dist = abs(N - M) + 5
    for _ in range(2):
      _lcs_match(max_dist)
      if f[N - 1, M - 1] > 0.8 * N: break
      max_dist *= 2

    orig_to_chartok_index = [None] * N
    chartok_to_orig_index = [None] * M
    i, j = N - 1, M - 1
    while i >= 0 and j >= 0:
      if (i, j) not in g: break
      if g[(i, j)] == 2:
        orig_to_chartok_index[i] = j
        chartok_to_orig_index[j] = i
        i, j = i - 1, j - 1
      elif g[(i, j)] == 1:
        j = j - 1
      else:
        i = i - 1

    if all(v is None for v in orig_to_chartok_index) or f[N - 1, M - 1] < 0.8 * N:
      print('MISMATCH DETECTED!')
      continue

    tok_start_to_orig_index = []
    tok_end_to_orig_index = []
    for i in range(len(para_tokens)):
      start_chartok_pos = tok_start_to_chartok_index[i]
      end_chartok_pos = tok_end_to_chartok_index[i]
      start_orig_pos = _convert_index(chartok_to_orig_index, start_chartok_pos,
                                      N, is_start=True)
      end_orig_pos = _convert_index(chartok_to_orig_index, end_chartok_pos,
                                    N, is_start=False)

      tok_start_to_orig_index.append(start_orig_pos)
      tok_end_to_orig_index.append(end_orig_pos)

    if not is_training:
      tok_start_position = tok_end_position = None

    if is_training and example.is_impossible:
      tok_start_position = -1
      tok_end_position = -1

    if is_training and not example.is_impossible:
      start_position = example.start_position
      # end_position = start_position + len(example.orig_answer_text) - 1
      end_position = example.send_position

      start_chartok_pos = _convert_index(orig_to_chartok_index, start_position,
                                         is_start=True)
      tok_start_position = chartok_to_tok_index[start_chartok_pos]

      end_chartok_pos = _convert_index(orig_to_chartok_index, end_position,
                                       is_start=False)
      tok_end_position = chartok_to_tok_index[end_chartok_pos]
      assert tok_start_position <= tok_end_position

    def _piece_to_id(x):
      if six.PY2 and isinstance(x, unicode):
        x = x.encode('utf-8')
      return sp_model.PieceToId(x)

    all_doc_tokens = list(map(_piece_to_id, para_tokens))

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset
      if length > max_tokens_for_doc:
        length = max_tokens_for_doc
      doc_spans.append(_DocSpan(start=start_offset, length=length))
      if start_offset + length == len(all_doc_tokens):
        break
      start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
      tokens = []
      token_is_max_context = {}
      segment_ids = []
      p_mask = []

      cur_tok_start_to_orig_index = []
      cur_tok_end_to_orig_index = []

      for i in range(doc_span.length):
        split_token_index = doc_span.start + i

        cur_tok_start_to_orig_index.append(
            tok_start_to_orig_index[split_token_index])
        cur_tok_end_to_orig_index.append(
            tok_end_to_orig_index[split_token_index])

        is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                               split_token_index)
        token_is_max_context[len(tokens)] = is_max_context
        tokens.append(all_doc_tokens[split_token_index])
        segment_ids.append(SEG_ID_P)
        p_mask.append(0)

      paragraph_len = len(tokens)

      tokens.append(SEP_ID)
      segment_ids.append(SEG_ID_P)
      p_mask.append(1)

      # note(zhiliny): we put P before Q
      # because during pretraining, B is always shorter than A
      for token in query_tokens:
        tokens.append(token)
        segment_ids.append(SEG_ID_Q)
        p_mask.append(1)
      tokens.append(SEP_ID)
      segment_ids.append(SEG_ID_Q)
      p_mask.append(1)

      cls_index = len(segment_ids)
      tokens.append(CLS_ID)
      segment_ids.append(SEG_ID_CLS)
      p_mask.append(0)

      input_ids = tokens

      # The mask has 0 for real tokens and 1 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [0] * len(input_ids)

      # Zero-pad up to the sequence length.
      while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(1)
        segment_ids.append(SEG_ID_PAD)
        p_mask.append(1)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length
      assert len(p_mask) == max_seq_length

      span_is_impossible = example.is_impossible
      start_position = None
      end_position = None
      if is_training and not span_is_impossible:
        # For training, if our document chunk does not contain an annotation
        # we throw it out, since there is nothing to predict.
        doc_start = doc_span.start
        doc_end = doc_span.start + doc_span.length - 1
        out_of_span = False
        if not (tok_start_position >= doc_start and
                tok_end_position <= doc_end):
          # print("out of span")
          # print("{}|{}|{}|{}".format(doc_start,tok_start_position,tok_end_position,doc_end))
          out_of_span = True
        if out_of_span:
          # continue
          start_position = 0
          end_position = 0
          span_is_impossible = True
        else:
          # note(zhiliny): we put P before Q, so doc_offset should be zero.
          # doc_offset = len(query_tokens) + 2
          doc_offset = 0
          start_position = tok_start_position - doc_start + doc_offset
          end_position = tok_end_position - doc_start + doc_offset

      if is_training and span_is_impossible:
        start_position = cls_index
        end_position = cls_index

          # note(zhiliny): With multi processing,
          # the example_index is actually the index within the current process
          # therefore we use example_index=None to avoid being used in the future.
          # The current code does not use example_index of training data.
      if is_training:
        feat_example_index = None
      else:
        feat_example_index = example_index

      feature = InputFeatures(
          unique_id=unique_id,
          example_index=feat_example_index,
          doc_span_index=doc_span_index,
          tok_start_to_orig_index=cur_tok_start_to_orig_index,
          tok_end_to_orig_index=cur_tok_end_to_orig_index,
          token_is_max_context=token_is_max_context,
          input_ids=input_ids,
          input_mask=input_mask,
          p_mask=p_mask,
          segment_ids=segment_ids,
          paragraph_len=paragraph_len,
          cls_index=cls_index,
          start_position=start_position,
          end_position=end_position,
          is_impossible=span_is_impossible)

      # Run callback
      output_fn(feature)

      unique_id += 1
      if span_is_impossible:
        cnt_neg += 1
      else:
        cnt_pos += 1

  tf.logging.info("Total number of instances: {} = pos {} neg {}".format(
      cnt_pos + cnt_neg, cnt_pos, cnt_neg))
	

def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index

class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, is_training):
    self.is_training = is_training
    self.num_features = 0

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_float_feature(feature.input_mask)
    features["p_mask"] = create_float_feature(feature.p_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)

    features["cls_index"] = create_int_feature([feature.cls_index])

    if self.is_training:
      features["start_positions"] = create_int_feature([feature.start_position])
      features["end_positions"] = create_int_feature([feature.end_position])
      impossible = 0
      if feature.is_impossible:
        impossible = 1
      features["is_impossible"] = create_float_feature([impossible])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    return tf_example.SerializeToString()

RawResult = collections.namedtuple("RawResult",
    ["unique_id", "start_top_log_probs", "start_top_index",
    "end_top_log_probs", "end_top_index", "cls_logits"])

_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction",
    ["feature_index", "start_index", "end_index",
    "start_log_prob", "end_log_prob"])

_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "NbestPrediction", ["text", "start_log_prob", "end_log_prob"])

def get_predictions(all_examples, all_features, all_results, n_best_size,
                    max_answer_length):
  """Write final predictions to the json file and log-odds of null if needed."""
  tf.logging.info("Getting predictions")

  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  all_predictions = collections.OrderedDict()

  for (example_index, example) in enumerate(all_examples):
    features = example_index_to_features[example_index]

    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = 1000000  # large and positive

    for (feature_index, feature) in enumerate(features):
      result = unique_id_to_result[feature.unique_id]

      cur_null_score = result.cls_logits

      # if we could have irrelevant answers, get the min score of irrelevant
      score_null = min(score_null, cur_null_score)

      for i in range(FLAGS.start_n_top):
        for j in range(FLAGS.end_n_top):
          start_log_prob = result.start_top_log_probs[i]
          start_index = result.start_top_index[i]

          j_index = i * FLAGS.end_n_top + j

          end_log_prob = result.end_top_log_probs[j_index]
          end_index = result.end_top_index[j_index]

          # We could hypothetically create invalid predictions, e.g., predict
          # that the start of the span is in the question. We throw out all
          # invalid predictions.
          if start_index >= feature.paragraph_len - 1:
            continue
          if end_index >= feature.paragraph_len - 1:
            continue

          if not feature.token_is_max_context.get(start_index, False):
            continue
          if end_index < start_index:
            continue
          length = end_index - start_index + 1
          if length > max_answer_length:
            continue

          prelim_predictions.append(
              _PrelimPrediction(
                  feature_index=feature_index,
                  start_index=start_index,
                  end_index=end_index,
                  start_log_prob=start_log_prob,
                  end_log_prob=end_log_prob))

    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_log_prob + x.end_log_prob),
        reverse=True)

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      feature = features[pred.feature_index]

      tok_start_to_orig_index = feature.tok_start_to_orig_index
      tok_end_to_orig_index = feature.tok_end_to_orig_index
      start_orig_pos = tok_start_to_orig_index[pred.start_index]
      end_orig_pos = tok_end_to_orig_index[pred.end_index]

      paragraph_text = example.paragraph_text
      final_text = paragraph_text[start_orig_pos: end_orig_pos + 1].strip()

      if final_text in seen_predictions:
        continue

      seen_predictions[final_text] = True

      nbest.append(
          _NbestPrediction(
              text=final_text,
              start_log_prob=pred.start_log_prob,
              end_log_prob=pred.end_log_prob))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          _NbestPrediction(text="", start_log_prob=-1e6,
          end_log_prob=-1e6))

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_log_prob + entry.end_log_prob)
      if not best_non_null_entry:
        best_non_null_entry = entry

    assert best_non_null_entry is not None

    all_predictions[example.qas_id] = best_non_null_entry.text

  return all_predictions


def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs


def input_fn_builder(input_glob, seq_length, is_training, drop_remainder,
                     num_hosts, num_threads=8):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.float32),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "cls_index": tf.FixedLenFeature([], tf.int64),
      "p_mask": tf.FixedLenFeature([seq_length], tf.float32)
  }

  if is_training:
    name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["is_impossible"] = tf.FixedLenFeature([], tf.float32)

  tf.logging.info("Input tfrecord file glob {}".format(input_glob))
  global_input_paths = tf.gfile.Glob(input_glob)
  tf.logging.info("Find {} input paths {}".format(
      len(global_input_paths), global_input_paths))

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    if FLAGS.use_tpu:
      batch_size = params["batch_size"]
    elif is_training:
      batch_size = FLAGS.train_batch_size
    else:
      batch_size = FLAGS.predict_batch_size

    # Split tfrecords across hosts
    if num_hosts > 1:
      host_id = params["context"].current_host
      num_files = len(global_input_paths)
      if num_files >= num_hosts:
        num_files_per_host = (num_files + num_hosts - 1) // num_hosts
        my_start_file_id = host_id * num_files_per_host
        my_end_file_id = min((host_id + 1) * num_files_per_host, num_files)
        input_paths = global_input_paths[my_start_file_id: my_end_file_id]
      tf.logging.info("Host {} handles {} files".format(host_id,
                                                        len(input_paths)))
    else:
      input_paths = global_input_paths

    if len(input_paths) == 1:
      d = tf.data.TFRecordDataset(input_paths[0])
      # For training, we want a lot of parallel reading and shuffling.
      # For eval, we want no shuffling and parallel reading doesn't matter.
      if is_training:
        d = d.shuffle(buffer_size=FLAGS.shuffle_buffer)
        d = d.repeat()
    else:
      d = tf.data.Dataset.from_tensor_slices(input_paths)
      # file level shuffle
      d = d.shuffle(len(input_paths)).repeat()

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_threads, len(input_paths))

      d = d.apply(
          tf.contrib.data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))

      if is_training:
        # sample level shuffle
        d = d.shuffle(buffer_size=FLAGS.shuffle_buffer)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_threads,
            drop_remainder=drop_remainder))
    d = d.prefetch(1024)

    return d

  return input_fn

def get_model_fn():
  def model_fn(features, labels, mode, params):
    #### Training or Evaluation
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    #### Get loss from inputs
    outputs = function_builder.get_qa_outputs(FLAGS, features, is_training)

    #### Check model parameters
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info('#params: {}'.format(num_params))

    scaffold_fn = None

    #### Evaluation mode
    if mode == tf.estimator.ModeKeys.PREDICT:
      if FLAGS.init_checkpoint:
        tf.logging.info("init_checkpoint not being used in predict mode.")

      predictions = {
          "unique_ids": features["unique_ids"],
          "start_top_index": outputs["start_top_index"],
          "start_top_log_probs": outputs["start_top_log_probs"],
          "end_top_index": outputs["end_top_index"],
          "end_top_log_probs": outputs["end_top_log_probs"],
          "cls_logits": outputs["cls_logits"]
      }

      if FLAGS.use_tpu:
        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
      else:
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions)
      return output_spec

    ### Compute loss
    seq_length = tf.shape(features["input_ids"])[1]
    def compute_loss(log_probs, positions):
      one_hot_positions = tf.one_hot(
          positions, depth=seq_length, dtype=tf.float32)

      loss = - tf.reduce_sum(one_hot_positions * log_probs, axis=-1)
      loss = tf.reduce_mean(loss)
      return loss

    start_loss = compute_loss(
        outputs["start_log_probs"], features["start_positions"])
    end_loss = compute_loss(
        outputs["end_log_probs"], features["end_positions"])

    total_loss = (start_loss + end_loss) * 0.5

    cls_logits = outputs["cls_logits"] ## answerability loss
    is_impossible = tf.reshape(features["is_impossible"], [-1])
    regression_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=is_impossible, logits=cls_logits)
    regression_loss = tf.reduce_mean(regression_loss)

    # note(zhiliny): by default multiply the loss by 0.5 so that the scale is
    # comparable to start_loss and end_loss
    total_loss += regression_loss * 0.5

    #### Configuring the optimizer
    train_op, learning_rate, _ = model_utils.get_train_op(FLAGS, total_loss)

    monitor_dict = {}
    monitor_dict["lr"] = learning_rate

    #### load pretrained models
    scaffold_fn = model_utils.init_from_checkpoint(FLAGS)

    #### Constucting training TPUEstimatorSpec with new cache.
    if FLAGS.use_tpu:
      host_call = function_builder.construct_scalar_host_call(
          monitor_dict=monitor_dict,
          model_dir=FLAGS.model_dir,
          prefix="train/",
          reduce_fn=tf.reduce_mean)

      train_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, loss=total_loss, train_op=train_op, host_call=host_call,
          scaffold_fn=scaffold_fn)
    else:
      train_spec = tf.estimator.EstimatorSpec(
          mode=mode, loss=total_loss, train_op=train_op)

    return train_spec

  return model_fn

def _get_spm_basename():
  spm_basename = os.path.basename(FLAGS.spiece_model_file)
  return spm_basename

def mrqa_predictor(FLAGS, json_data):
  """
  Get prediction with the data got fron mrqa official request.
  """
  tf.logging.set_verbosity(tf.logging.INFO)

  sp_model = spm.SentencePieceProcessor()
  sp_model.Load(FLAGS.spiece_model_file)

  ### TPU Configuration
  run_config = model_utils.configure_tpu(FLAGS)

  model_fn = get_model_fn()  # model configs, load the trained model
  spm_basename = _get_spm_basename()

  tf.logging.info("Got Data from Server...")
  eval_data = arrange_mrqa_data(json_data, is_training=False)

  eval_writer = FeatureWriter(is_training=False)
  eval_features = []
  eval_features_inp = []

  def append_feature(feature):
    eval_features.append(feature)
    eval_features_inp.append(eval_writer.process_feature(feature))

  convert_examples_to_features(
			examples=eval_data,
			sp_model=sp_model,
			max_seq_length=FLAGS.max_seq_length,
			doc_stride=FLAGS.doc_stride,
			max_query_length=FLAGS.max_query_length,
			is_training=False,
			output_fn=append_feature) 

  predict_fn = tf.contrib.predictor.from_saved_model(FLAGS.export_dir_base)

  cur_results = []

  for num, eval_feature in enumerate(eval_features_inp):
    result = predict_fn({"examples":[eval_feature]})

    if len(cur_results) % 1000 == 0:
			tf.logging.info("Processing example: %d" % (len(cur_results)))

    unique_id = int(result["unique_ids"])
    start_top_log_probs = (
				[float(x) for x in result["start_top_log_probs"].flat])
    start_top_index = [int(x) for x in result["start_top_index"].flat]
    end_top_log_probs = (
				[float(x) for x in result["end_top_log_probs"].flat])
    end_top_index = [int(x) for x in result["end_top_index"].flat]

    cls_logits = float(result["cls_logits"].flat[0])

    cur_results.append(
				RawResult(
						unique_id=unique_id,
						start_top_log_probs=start_top_log_probs,
						start_top_index=start_top_index,
						end_top_log_probs=end_top_log_probs,
						end_top_index=end_top_index,
						cls_logits=cls_logits))

  ret = get_predictions(eval_data, eval_features, cur_results,
                        FLAGS.n_best_size, FLAGS.max_answer_length)
  print(ret)
  return dict(ret)


def mrqa_tester():
  """
  Get prediction with the data got fron mrqa official request.
  """
  tf.logging.set_verbosity(tf.logging.INFO)

  sp_model = spm.SentencePieceProcessor()
  sp_model.Load(FLAGS.spiece_model_file)

  ### TPU Configuration
  run_config = model_utils.configure_tpu(FLAGS)

  model_fn = get_model_fn()  # model configs, load the trained model
  spm_basename = _get_spm_basename()

  dev_file = FLAGS.dev_dir+"/DROP-piece.jsonl"
  tf.logging.info("Read data from {}".format(dev_file))
  eval_data = read_mrqa_data(dev_file, is_training=False)

  eval_writer = FeatureWriter(is_training=False)
  eval_features = []
  eval_features_inp = []

  def append_feature(feature):
    eval_features.append(feature)
    eval_features_inp.append(eval_writer.process_feature(feature))

  convert_examples_to_features(
			examples=eval_data,
			sp_model=sp_model,
			max_seq_length=FLAGS.max_seq_length,
			doc_stride=FLAGS.doc_stride,
			max_query_length=FLAGS.max_query_length,
			is_training=False,
			output_fn=append_feature) 

  predict_fn = tf.contrib.predictor.from_saved_model(FLAGS.export_dir_base)

  cur_results = []
  N = 0
  for num, eval_feature in enumerate(eval_features_inp):
    result = predict_fn({"examples":[eval_feature]})

    if len(cur_results) % 1000 == 0:
			tf.logging.info("Processing example: %d" % (len(cur_results)))

    unique_id = int(result["unique_ids"])
    start_top_log_probs = (
				[float(x) for x in result["start_top_log_probs"].flat])
    start_top_index = [int(x) for x in result["start_top_index"].flat]
    end_top_log_probs = (
				[float(x) for x in result["end_top_log_probs"].flat])
    end_top_index = [int(x) for x in result["end_top_index"].flat]

    cls_logits = float(result["cls_logits"].flat[0])

    cur_results.append(
				RawResult(
						unique_id=unique_id,
						start_top_log_probs=start_top_log_probs,
						start_top_index=start_top_index,
						end_top_log_probs=end_top_log_probs,
						end_top_index=end_top_index,
						cls_logits=cls_logits))

  ret = get_predictions(eval_data, eval_features, cur_results,
                        FLAGS.n_best_size, FLAGS.max_answer_length)
  print(ret)

if __name__ == "__main__":
  FLAGS(sys.argv)
  mrqa_tester()
