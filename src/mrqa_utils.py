import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys

OPTS = None

def make_qid_to_has_ans(dataset):
  qid_to_has_ans = {}
  for entry in dataset:
    qid_to_has_ans[entry['qid']] = bool(entry['answers'])
  return qid_to_has_ans

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def get_raw_scores(dataset, preds):
  exact_scores = {}
  f1_scores = {}

  for qa in dataset:
    qid = qa['qid']
    gold_answers = [a['text'] for a in qa['detected_answers']
                    if normalize_answer(a['text'])]
    if not gold_answers:
      # For unanswerable questions, only correct answer is empty string
      gold_answers = ['']
    if qid not in preds:
      print('Missing prediction for %s' % qid)
      continue
    a_pred = preds[qid]
    # Take max over all gold answers
    exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
    f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
  return exact_scores, f1_scores

def find_best_thresh_v2(preds, scores, na_probs, qid_to_has_ans):
  num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
  cur_score = num_no_ans
  best_score = cur_score
  best_thresh = 0.0
  qid_list = sorted(na_probs, key=lambda k: na_probs[k])
  for i, qid in enumerate(qid_list):
    if qid not in scores: continue
    if qid_to_has_ans[qid]:
      diff = scores[qid]
    else:
      if preds[qid]:
        diff = -1
      else:
        diff = 0
    cur_score += diff
    if cur_score > best_score:
      best_score = cur_score
      best_thresh = na_probs[qid]

  has_ans_score, has_ans_cnt = 0, 0
  for qid in qid_list:
    if not qid_to_has_ans[qid]: continue
    has_ans_cnt += 1

    if qid not in scores: continue
    has_ans_score += scores[qid]
  return 100.0 * best_score / len(scores), best_thresh, 1.0 * has_ans_score / has_ans_cnt

def find_all_best_thresh_v2(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
  best_exact, exact_thresh, has_ans_exact = find_best_thresh_v2(preds, exact_raw, na_probs, qid_to_has_ans)
  best_f1, f1_thresh, has_ans_f1 = find_best_thresh_v2(preds, f1_raw, na_probs, qid_to_has_ans)
  main_eval['best_exact'] = best_exact
  main_eval['best_exact_thresh'] = exact_thresh
  main_eval['best_f1'] = best_f1
  main_eval['best_f1_thresh'] = f1_thresh
  main_eval['has_ans_exact'] = has_ans_exact
  main_eval['has_ans_f1'] = has_ans_f1