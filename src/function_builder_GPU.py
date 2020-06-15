"""doc."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import tensorflow as tf
import .modeling
import .xlnet


def construct_scalar_host_call(
    monitor_dict,
    model_dir,
    prefix="",
    reduce_fn=None):
  """
  Construct host calls to monitor training progress on TPUs.
  """

  metric_names = list(monitor_dict.keys())

  def host_call_fn(global_step, *args):
    """actual host call function."""
    step = global_step[0]
    with tf.contrib.summary.create_file_writer(
        logdir=model_dir, filename_suffix=".host_call").as_default():
      with tf.contrib.summary.always_record_summaries():
        for i, name in enumerate(metric_names):
          if reduce_fn is None:
            scalar = args[i][0]
          else:
            scalar = reduce_fn(args[i])
          with tf.contrib.summary.record_summaries_every_n_global_steps(
              100, global_step=step):
            tf.contrib.summary.scalar(prefix + name, scalar, step=step)

        return tf.contrib.summary.all_summary_ops()

  global_step_tensor = tf.reshape(tf.train.get_or_create_global_step(), [1])
  other_tensors = [tf.reshape(monitor_dict[key], [1]) for key in metric_names]

  return host_call_fn, [global_step_tensor] + other_tensors


def get_classification_loss(
    FLAGS, features, n_class, is_training):
  """Loss for downstream classification tasks."""

  bsz_per_core = tf.shape(features["input_ids"])[0]

  inp = tf.transpose(features["input_ids"], [1, 0])
  seg_id = tf.transpose(features["segment_ids"], [1, 0])
  inp_mask = tf.transpose(features["input_mask"], [1, 0])
  label = tf.reshape(features["label_ids"], [bsz_per_core])

  xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)
  run_config = xlnet.create_run_config(is_training, True, FLAGS)

  xlnet_model = xlnet.XLNetModel(
      xlnet_config=xlnet_config,
      run_config=run_config,
      input_ids=inp,
      seg_ids=seg_id,
      input_mask=inp_mask)

  summary = xlnet_model.get_pooled_out(FLAGS.summary_type, FLAGS.use_summ_proj)

  with tf.variable_scope("model", reuse=tf.AUTO_REUSE):

    if FLAGS.cls_scope is not None and FLAGS.cls_scope:
      cls_scope = "classification_{}".format(FLAGS.cls_scope)
    else:
      cls_scope = "classification_{}".format(FLAGS.task_name.lower())

    per_example_loss, logits = modeling.classification_loss(
        hidden=summary,
        labels=label,
        n_class=n_class,
        initializer=xlnet_model.get_initializer(),
        scope=cls_scope,
        return_logits=True)

    total_loss = tf.reduce_mean(per_example_loss)

    return total_loss, per_example_loss, logits


def get_qa_outputs(FLAGS, features, is_training):
  """Loss for downstream span-extraction QA tasks such as SQuAD."""

  inp = tf.transpose(features["input_ids"], [1, 0])
  seg_id = tf.transpose(features["segment_ids"], [1, 0])
  inp_mask = tf.transpose(features["input_mask"], [1, 0])
  cls_index = tf.reshape(features["cls_index"], [-1])

  seq_len = tf.shape(inp)[0]

  xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)
  run_config = xlnet.create_run_config(is_training, True, FLAGS)
  
  seg_id = tf.cast(seg_id, tf.int32)
  xlnet_model = xlnet.XLNetModel(
      xlnet_config=xlnet_config,
      run_config=run_config,
      input_ids=inp,
      seg_ids=seg_id,
      input_mask=inp_mask)
  output = xlnet_model.get_sequence_output()   # the output of xlnet
  initializer = xlnet_model.get_initializer()

  return_dict = {}

  # invalid position mask such as query and special symbols (PAD, SEP, CLS)
  p_mask = features["p_mask"]

  keep_prob = 1.0
  if is_training:
    keep_prob = 0.8
  else:
    keep_prob = 1.0

  # logit of the start position: linear(512)->relu->dropout->linear(384)->relu->dropout->linear(1)->softmax
  with tf.variable_scope("start_logits"):
    start_logits = tf.layers.dense(
        output,
        512,
        kernel_initializer=initializer,
        name="start_dense_1")
    start_logits = tf.nn.relu(start_logits)
    start_logits = tf.nn.dropout(start_logits, keep_prob)
    
    start_logits = tf.layers.dense(
        start_logits,
        384,
        kernel_initializer=initializer,
        name="start_dense_2")
    start_logits = tf.nn.relu(start_logits)
    start_logits = tf.nn.dropout(start_logits, keep_prob)
    
    start_logits = tf.layers.dense(
        start_logits,
        1,
        kernel_initializer=initializer,
        name="start_dense_3")
    
    start_logits = tf.transpose(tf.squeeze(start_logits, -1), [1, 0])
    start_logits_masked = start_logits * (1 - p_mask) - 1e30 * p_mask
    start_log_probs = tf.nn.log_softmax(start_logits_masked, -1)

  # logit of the end position: tanh(linear([output,start features]))->norm->linear(512)->relu->dropout->linear(1)->softmax
  with tf.variable_scope("end_logits"):
    if is_training:
      # during training, compute the end logits based on the
      # ground truth of the start position

      start_positions = tf.reshape(features["start_positions"], [-1])
      start_index = tf.one_hot(start_positions, depth=seq_len, axis=-1,
                               dtype=tf.float32)
      start_features = tf.einsum("lbh,bl->bh", output, start_index) # start_features[b,h]=output[l,b,h]xstart_index[b,l]
      start_features = tf.tile(start_features[None], [seq_len, 1, 1])
      
      end_logits = tf.layers.dense(
          tf.concat([output, start_features], axis=-1), xlnet_config.d_model,
          kernel_initializer=initializer, activation=tf.tanh, name="dense_0")
      end_logits = tf.contrib.layers.layer_norm(end_logits, begin_norm_axis=-1)

      end_logits = tf.layers.dense(
          end_logits, 512,
          kernel_initializer=initializer,
          name="end_dense_1")
      end_logits = tf.nn.relu(end_logits)  
      end_logits = tf.nn.dropout(end_logits, keep_prob)

      end_logits = tf.layers.dense(
          end_logits, 384,
          kernel_initializer=initializer,
          name="end_dense_2")
      end_logits = tf.nn.relu(end_logits)  
      end_logits = tf.nn.dropout(end_logits, keep_prob)

      end_logits = tf.layers.dense(
          end_logits, 1,
          kernel_initializer=initializer,
          name="end_dense_3")


      end_logits = tf.transpose(tf.squeeze(end_logits, -1), [1, 0])
      end_logits_masked = end_logits * (1 - p_mask) - 1e30 * p_mask
      end_log_probs = tf.nn.log_softmax(end_logits_masked, -1)
    else:
      # during inference, compute the end logits based on beam search

      start_top_log_probs, start_top_index = tf.nn.top_k(                 # get top k start indexes and probs
          start_log_probs, k=FLAGS.start_n_top)
      start_index = tf.one_hot(start_top_index,
                               depth=seq_len, axis=-1, dtype=tf.float32)
      start_features = tf.einsum("lbh,bkl->bkh", output, start_index) # start_features[b,k,h]=output[l,b,h]xstart_index[b,k,l]
      end_input = tf.tile(output[:, :, None],
                          [1, 1, FLAGS.start_n_top, 1])
      start_features = tf.tile(start_features[None],
                               [seq_len, 1, 1, 1])
     
     
      end_input = tf.concat([end_input, start_features], axis=-1) # tanh(linear([end_input,start_feature]))->norm
      end_logits = tf.layers.dense(
          end_input,
          xlnet_config.d_model,
          kernel_initializer=initializer,
          activation=tf.tanh,
          name="dense_0")
      end_logits = tf.contrib.layers.layer_norm(end_logits, begin_norm_axis=-1)
      
      end_logits = tf.layers.dense(                               # ->linear(512)->relu->dropout
          end_logits, 512,
          kernel_initializer=initializer,
          name="end_dense_1")
      end_logits = tf.nn.relu(end_logits)  
      end_logits = tf.nn.dropout(end_logits, 1.0)

      end_logits = tf.layers.dense(                               # ->linear(384)->relu->dropout
          end_logits, 384,
          kernel_initializer=initializer,
          name="end_dense_2")
      end_logits = tf.nn.relu(end_logits)  
      end_logits = tf.nn.dropout(end_logits, 1.0)

      end_logits = tf.layers.dense(                               # ->linear(1)->softmax
          end_logits,
          1,
          kernel_initializer=initializer,
          name="end_dense_3")

      end_logits = tf.reshape(end_logits, [seq_len, -1, FLAGS.start_n_top])
      end_logits = tf.transpose(end_logits, [1, 2, 0])
      end_logits_masked = end_logits * (
          1 - p_mask[:, None]) - 1e30 * p_mask[:, None]
      end_log_probs = tf.nn.log_softmax(end_logits_masked, -1)
      end_top_log_probs, end_top_index = tf.nn.top_k(             # get top k end indexs and probs
          end_log_probs, k=FLAGS.end_n_top)
      end_top_log_probs = tf.reshape(
          end_top_log_probs,
          [-1, FLAGS.start_n_top * FLAGS.end_n_top])
      end_top_index = tf.reshape(
          end_top_index,
          [-1, FLAGS.start_n_top * FLAGS.end_n_top])

  if is_training:
    return_dict["start_log_probs"] = start_log_probs
    return_dict["end_log_probs"] = end_log_probs
  else:
    return_dict["start_top_log_probs"] = start_top_log_probs
    return_dict["start_top_index"] = start_top_index
    return_dict["end_top_log_probs"] = end_top_log_probs
    return_dict["end_top_index"] = end_top_index

  # an additional layer to predict answerability
  with tf.variable_scope("answer_class"): # use start feature and cls feature to check the ans
    # get the representation of CLS
    cls_index = tf.one_hot(cls_index, seq_len, axis=-1, dtype=tf.float32)
    cls_feature = tf.einsum("lbh,bl->bh", output, cls_index)

    # get the representation of START
    start_p = tf.nn.softmax(start_logits_masked, axis=-1,
                            name="softmax_start")
    start_feature = tf.einsum("lbh,bl->bh", output, start_p)

    # note(zhiliny): no dependency on end_feature so that we can obtain
    # one single `cls_logits` for each sample
    ans_feature = tf.concat([start_feature, cls_feature], -1)
    ans_feature = tf.layers.dense(
        ans_feature,
        xlnet_config.d_model,
        activation=tf.tanh,
        kernel_initializer=initializer, name="dense_0")
    ans_feature = tf.layers.dropout(ans_feature, FLAGS.dropout,
                                    training=is_training)
    
    cls_logits = tf.layers.dense(
        ans_feature, 256,
        kernel_initializer=initializer,
        name="cls_dense_256")
    cls_logits = tf.nn.relu(cls_logits)  
    cls_logits = tf.nn.dropout(cls_logits, keep_prob)
    
    cls_logits = tf.layers.dense(
        cls_logits,
        1,
        kernel_initializer=initializer,
        name="dense_1",
        use_bias=False)
    cls_logits = tf.squeeze(cls_logits, -1)

    return_dict["cls_logits"] = cls_logits

  return return_dict