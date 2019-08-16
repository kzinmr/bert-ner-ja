# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
tf.enable_eager_execution()

import modeling
import tokenization
import optimization

import tf_metrics

import pickle
from collections import OrderedDict
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, # kblabel
                 label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        # self.kblabel = kblabel
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                #  kblabel_ids,
                 label_ids):

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        # self.kblabel_ids = kblabel_ids
        self.label_ids = label_ids
        # self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file, eos='。'):
        """Reads a IOB data.
        FIXME: it's buggy:
         - input_file must contain eos character ('。')
         - input_file must end with '\n'
        """
        with open(input_file) as f:
            lines = []
            words = []
            # kblabels = []
            labels = []
            for line in f:
                contents = line.strip()
                word = line.strip().split(' ')[0]
                # kblabel = line.strip().split(' ')[2]
                label = line.strip().split(' ')[-1]
                if contents.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                if len(words) > 0 and len(contents) == 0 and words[-1] == eos:
                    label_str = ' '.join(
                        [label for label in labels if len(label) > 0])
                    # kblabel_str = ' '.join(
                    #     [kblabel for kblabel in kblabels if len(kblabel) > 0])
                    word_str = ' '.join(
                        [word for word in words if len(word) > 0])
                    lines.append([label_str, word_str])  # kblabel_str
                    words = []
                    # kblabels = []
                    labels = []
                    continue
                words.append(word)
                # kblabels.append(kblabel)
                labels.append(label)

            return lines


class NerProcessor(DataProcessor):

    def __init__(self, label_vocab_path: str, data_dir: str):
        self.data_dir = data_dir

        labels = []
        with open(label_vocab_path) as f:
            labels = [l.strip() for l in f.read().split('\n') if l.strip()]
        assert len(labels) > 0
        assert all(l.startswith('B-') or l.startswith('I-') for l in labels)
        labels.extend(['O', 'X', '[CLS]', '[SEP]'])
        self.labels = labels

    def get_labels(self):
        return self.labels

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_data(os.path.join(self.data_dir, "train.txt")), "train")
    
    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_data(os.path.join(self.data_dir, "dev.txt")), "dev")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_data(os.path.join(self.data_dir, "test.txt")), "test")

    def get_examples(self, sentences, labels_list=None):
        """sentences: wakati list"""
        # [sent_str] ->  [ [label_str, word_str] ]
        lines = []
        if labels_list is None:
            for sentence in sentences:
                label_str = ' '.join('O' for _ in sentence.split(' '))
                lines.append([label_str, sentence])
        else:
            assert len(sentences) == len(labels_list)
            for sentence, label_str in zip(sentences, labels_list):
                lines.append([label_str, sentence])
        return self._create_examples(lines, "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        # set_type: train/dev/test
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            # kblabel = tokenization.convert_to_unicode(line[2])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))  # kblabel
        return examples


class DataBuilder:

    def __init__(self, data_dir, labels_path, vocab_path, output_dir, max_seq_length, drop_remainder=True, mode="train"):
        self.processor = NerProcessor(labels_path, data_dir)  # .txt -> InputExamples
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=False)
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.drop_remainder = drop_remainder

        # labels_path は学習と予測時と同じものである必要がある
        label_list = self.processor.get_labels()
        self.label2id = {label: idx for idx, label in enumerate(label_list, 1)}
        # with open(os.path.join(output_dir, "label2id.pkl"), 'wb') as f:
        #     pickle.dump(self.label2id, f)
        # NOTE: [NULL] も考慮
        self.id2label = {value: key for key, value in self.label2id.items()}
        self.num_labels = len(self.id2label) + 1

        self.num_examples = 0
        self.word_tokenizer = None
        if mode == "train":
            self.num_examples = len(self.processor.get_train_examples())
        elif mode == "evaluate":
            self.num_examples = len(self.processor.get_dev_examples())
        elif mode == "predict":
            if data_dir is not None:
                self.num_examples = len(self.processor.get_test_examples())
            else:
                self.word_tokenizer = KnpBase(jumanpp=True)

            export_file = os.path.join(self.output_dir, "token_label_pred.txt")
            self.swc = SubwordWordConverter(self.tokenizer,
                                            self.id2label,
                                            export_file)

    def convert_single_example(self, ex_index, example):
        """Converts a single `InputExample` into a single `InputFeatures`."""

        textlist = example.text.split(' ')
        # kblabellist = example.kblabel.split(' ')
        labellist = example.label.split(' ')
        tokens = []
        # kblabels = []
        labels = []
        for i, word in enumerate(textlist):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            # kblabel_1 = kblabellist[i]
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    # kblabels.append(kblabel_1)
                    labels.append(label_1)
                else:
                    # kblabels.append("X")
                    labels.append("X")

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > self.max_seq_length - 2:
            tokens = tokens[0:(self.max_seq_length - 2)]
            # kblabels = kblabels[0:(self.max_seq_length - 2)]
            labels = labels[0:(self.max_seq_length - 2)]

        ntokens = []
        segment_ids = []
        kblabel_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        # append("O") or append("[CLS]") not sure!
        # kblabel_ids.append(self.label2id["[CLS]"])
        label_ids.append(self.label2id["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            # kblabel_ids.append(self.label2id.get(kblabels[i], 'O'))
            label_ids.append(self.label2id.get(labels[i], self.label2id['O']))
        ntokens.append("[SEP]")
        segment_ids.append(0)
        # append("O") or append("[SEP]") not sure!
        # kblabel_ids.append(self.label2id["[SEP]"])
        label_ids.append(self.label2id["[SEP]"])

        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # kblabel_ids.append(0)
            label_ids.append(0)
            ntokens.append("[NULL]")

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        # assert len(kblabel_ids) == self.max_seq_length
        assert len(label_ids) == self.max_seq_length

        if ex_index < 5:
            tf.compat.v1.logging.info("*** Example ***")
            tf.compat.v1.logging.info("guid: %s" % (example.guid))
            tf.compat.v1.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            tf.compat.v1.logging.info("input_ids: %s" %
                            " ".join([str(x) for x in input_ids]))
            tf.compat.v1.logging.info("input_mask: %s" %
                            " ".join([str(x) for x in input_mask]))
            tf.compat.v1.logging.info("segment_ids: %s" %
                            " ".join([str(x) for x in segment_ids]))
            # tf.compat.v1.logging.info("kblabel_ids: %s" %
            #                 " ".join([str(x) for x in kblabel_ids]))
            tf.compat.v1.logging.info("label_ids: %s" %
                            " ".join([str(x) for x in label_ids]))

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            # kblabel_ids=kblabel_ids,
            label_ids=label_ids)

        return feature


    def file_based_convert_examples_to_features(self, examples, output_file):
        """Convert a set of `InputExample`s to `InputFeatures`
           and save them into a TFRecord file."""

        def create_int_feature(values):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

        writer = tf.io.TFRecordWriter(output_file)

        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                tf.compat.v1.logging.info(f"Writing example {ex_index} of {len(examples)}")

            feature = self.convert_single_example(ex_index, example)

            features = OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["input_mask"] = create_int_feature(feature.input_mask)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            # features["kblabel_ids"] = create_int_feature(feature.kblabel_ids)
            features["label_ids"] = create_int_feature(feature.label_ids)

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        writer.close()


    def file_based_input_fn_builder(self, input_file, is_training):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""

        name_to_features = {
            "input_ids": tf.io.FixedLenFeature([self.max_seq_length], tf.int64),
            "input_mask": tf.io.FixedLenFeature([self.max_seq_length], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([self.max_seq_length], tf.int64),
            # "kblabel_ids": tf.io.FixedLenFeature([self.max_seq_length], tf.int64),
            "label_ids": tf.io.FixedLenFeature([self.max_seq_length], tf.int64),
        }

        def _decode_record(record, name_to_features):
            """Decodes a record to a TensorFlow example."""
            example = tf.io.parse_single_example(record, name_to_features)

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
            batch_size = params["batch_size"]

            # For training, we want a lot of parallel reading and shuffling.
            # For eval, we want no shuffling and parallel reading doesn't matter.
            d = tf.data.TFRecordDataset(input_file)
            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100)
            d = d.map(lambda record: _decode_record(record, name_to_features))\
                 .batch(batch_size, self.drop_remainder)

            return d

        return input_fn


    def make_input_fn(self, mode='train'):
        if mode=='train':
            examples = self.processor.get_train_examples()
            output_file = os.path.join(self.output_dir, "train.tf_record")
            is_training = True
        elif mode=='eval':
            examples = self.processor.get_dev_examples()
            output_file = os.path.join(self.output_dir, "eval.tf_record")
            is_training = False
        else:
            examples = self.processor.get_test_examples()
            output_file = os.path.join(self.output_dir, "test.tf_record")
            is_training = False

        # List[InputExample] -> InputFeatures as .tfrecord
        self.file_based_convert_examples_to_features(examples, output_file)

        input_fn = self.file_based_input_fn_builder(
            input_file=output_file,
            is_training=is_training)
        return input_fn


    @staticmethod
    def __insert_eos(sentences, labels_list=None):
        if labels_list is not None:
            sentences_n, labels_list_n = [], []
            for ss, ls in zip(sentences, labels_list):
                if ss[-1] != '。':
                    sentences_n.append(ss + ['。'])
                    labels_list_n.append(ls + ['O'])
            return sentences_n, labels_list_n
        else:
            return [ss + ['。'] if ss[-1] != '。' else ss for ss in sentences]

            
    def make_input_fn_from_sentences(self, sentences):
        # NOTE: (subword->word復元に必要な) ダミーの labels_list はfile生成時に作ることにしている
        # word_tokenizer はtokenizerに含めたい気持ちがあるが、ダミーラベル生成を併せて行いたいのが厄介
        sentences = [self.word_tokenizer.wakati(s).split(' ')
                        for s in sentences]
        sentences = self.__insert_eos(sentences)
        tokenized_sentences = list(map(lambda x: ' '.join(x), sentences))

        examples = self.processor.get_examples(tokenized_sentences)

        output_file = os.path.join(self.output_dir, "predict.tf_record")
        is_training = False

        # List[InputExample] -> InputFeatures as .tfrecord
        self.file_based_convert_examples_to_features(examples, output_file, mode="test")

        input_fn = self.file_based_input_fn_builder(
            input_file=output_file,
            is_training=is_training)
        return input_fn


    def export_result(self, result):
        output_eval_file = os.path.join(self.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            tf.compat.v1.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.compat.v1.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


class SubwordWordConverter:

    def __init__(self, tokenizer, id2label, export_file=None):
        self.tokenizer = tokenizer
        self.id2label = id2label
        label2id = {v: k for k, v in id2label.items()}
        self.LABELID_PAD = 0
        self.LABELID_CLS = label2id['[CLS]']
        self.LABELID_SEP = label2id['[SEP]']
        self.LABELID_X = label2id['X']
        self.ignore_label_ids = {self.LABELID_PAD,
                                 self.LABELID_CLS, self.LABELID_SEP}
        self.TOKENID_PAD = tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
        self.TOKENID_CLS = tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.TOKENID_SEP = tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        self.ignore_token_ids = {self.TOKENID_PAD,
                                 self.TOKENID_CLS, self.TOKENID_SEP}

        self.export_file = export_file

    @staticmethod
    def convert_subword_to_word_by_label(subwords, labels_gold):
        # subword.startswith('##') == True だけがsubwordとは限らない
        # 'X' label を subword　-> word の復元に用いる
        words, labels = [], []
        for sw, lb in zip(subwords, labels_gold):
            if lb == 'X':
                assert len(words) > 0
                prev = words[-1]
                words = words[:-1]
                word = prev + sw[2:]
            else:
                word = sw
                labels.append(lb)
            words.append(word)
        return words, labels

    def check_separator_aligned(self, inputs, labels):
        for i, label in zip(inputs, labels):
            if label == self.LABELID_CLS:
                if i != self.TOKENID_CLS:
                    return False
            elif label == self.LABELID_SEP:
                if i != self.TOKENID_SEP:
                    return False
        return True

    def filter_token_ids(self, token_ids):
        return [i for i in token_ids if i not in self.ignore_token_ids]

    def filter_label_ids(self, label_ids):
        return [l for l in label_ids if l not in self.ignore_label_ids]

    def convert_id_to_surface_token(self, token_ids):
        token_ids = self.filter_token_ids(token_ids)
        return self.tokenizer.convert_ids_to_tokens(token_ids)

    def convert_id_to_surface_label(self, label_ids):
        label_ids = self.filter_label_ids(label_ids)
        return [self.id2label[i] for i in label_ids]

    def convert_tokens_to_words(self, token_ids, label_ids_gold, subword=False):
        # subword　-> word の復元
        subwords = self.convert_id_to_surface_token(token_ids)
        labels = self.convert_id_to_surface_label(label_ids_gold)
        if subword:
            return subwords, labels
        else:
            words, labels = self.convert_subword_to_word_by_label(
                zip(subwords, labels))
            return words, labels

    def filter_label_ids_by_gold(self, label_ids_pred, label_ids_gold):
        label_ids_pred = self.filter_label_ids(label_ids_pred)
        label_ids_gold = self.filter_label_ids(label_ids_gold)
        label_ids_pred = [l for l, lg in zip(label_ids_pred, label_ids_gold)
                          if lg != self.LABELID_X]
        return label_ids_pred

    def convert_ids_to_surfaces(self, token_ids, label_ids_pred, label_ids_gold, subword=False):
        # gold label が 'X' であるか否かを基点に subword かどうかを認識する
        subwords = self.convert_id_to_surface_token(token_ids)
        labels_gold = self.convert_id_to_surface_label(label_ids_gold)
        if subword:
            words = subwords
        else:
            # subwords => words
            words, labels_gold = self.convert_subword_to_word_by_label(
                subwords, labels_gold)
            # subword_labels => word_labels
            label_ids_pred = self.filter_label_ids_by_gold(
                label_ids_pred, label_ids_gold)
        labels_pred = self.convert_id_to_surface_label(label_ids_pred)

        return words, labels_pred, labels_gold

    def convert_ids_to_surfaces_list(self, token_ids_list, label_ids_list_pred, label_ids_list_gold, subword=False):
        tf.compat.v1.logging.info("***** Predict results *****")
        output_sentences = []
        tokens_list, labels_list_pred, labels_list_gold = [], [], []
        for token_ids, label_ids_pred, label_ids_gold in zip(token_ids_list, label_ids_list_pred, label_ids_list_gold):
            if self.check_separator_aligned(token_ids, label_ids_pred):
                words, labels_pred, labels_gold = self.convert_ids_to_surfaces(
                    token_ids, label_ids_pred, label_ids_gold, subword=subword)
                tokens_list.append(words)
                labels_list_pred.append(labels_pred)
                labels_list_gold.append(labels_gold)

                # export
                if self.export_file is not None:
                    output_lines = [f'{word}\t{label}\t{label_gold}'
                                    for word, label, label_gold in zip(words, labels_pred, labels_gold)]
                    output_line = "\n".join(output_lines)
                    output_line += "\n\n"
                    output_sentences.append(output_line)
        if self.export_file is not None:
            with open(self.export_file, 'w') as writer:
                for output_sentence in output_sentences:
                    writer.write(output_sentence)

        return tokens_list, labels_list_pred, labels_list_gold


class ModelBuilder:

    def __init__(self, num_labels, model_dir, bert_config_file, init_checkpoint, max_seq_length, 
                 save_checkpoints_steps=None,
                 num_train_steps=None, warmup_proportion=None, learning_rate=None,
                 train_batch_size=16, eval_batch_size=8, predict_batch_size=8):
        bert_config = modeling.BertConfig.from_json_file(bert_config_file)
        if max_seq_length > bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (max_seq_length, bert_config.max_position_embeddings))
        self.max_seq_length = max_seq_length
    
        run_config = tf.contrib.tpu.RunConfig(
            model_dir=model_dir,
            save_checkpoints_steps=save_checkpoints_steps,
        )

        # num_train_steps = int(num_examples / train_batch_size * num_train_epochs)
        if num_train_steps is not None and warmup_proportion is not None and learning_rate is not None:

            num_warmup_steps = int(num_train_steps * warmup_proportion)
            model_fn = self.model_fn_builder(
                bert_config=bert_config,
                num_labels=num_labels,
                init_checkpoint=init_checkpoint,
                learning_rate=learning_rate,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps)
        else:
            model_fn = self.model_fn_builder(
                bert_config=bert_config,
                num_labels=num_labels,
                init_checkpoint=init_checkpoint)
    
        self.estimator = tf.contrib.tpu.TPUEstimator(
                use_tpu=None,
                model_fn=model_fn,
                config=run_config,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                predict_batch_size=predict_batch_size
                )
                

    def create_model(self, bert_config, is_training, features, num_labels):
        """Creates a token-level classification model."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        # kblabel_ids = features["kblabel_ids"]

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=None)  # use_tpu=None

        # 出力層を改造する際、ここをいじる
        # hidden_size -> num_labels
        # NEW: hidden_size+num_labels -> num_labels

        # kbmatch_ids = features["kbmatch_ids"]  # [batch_size, seq_length, 1]
        # one_hot_kbmatches = tf.one_hot(kbmatch_ids, depth=num_labels, dtype=tf.float32)  # [batch_size, seq_length, num_labels]

        output_layer = model.get_sequence_output()  # [batch_size, seq_length, hidden_size]
        
        # output_layer = tf.concat([output_layer, one_hot_kbmatches], 2)  # [batch_size, seq_length, hidden_size + num_labels]
        hidden_size = output_layer.shape[-1].value

        output_weights = tf.compat.v1.get_variable("output_weights", [num_labels, hidden_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
        # output_weights = tf.compat.v1.get_variable("output_weights", [num_labels, hidden_size+num_labels], initializer=tf.truncated_normal_initializer(stddev=0.02))
        output_bias = tf.compat.v1.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())

        label_ids = features["label_ids"]
        with tf.compat.v1.variable_scope("loss"):

            # ここもいじる
            if is_training:
                output_layer = tf.nn.dropout(output_layer, rate=0.1)
            output_layer = tf.reshape(output_layer, [-1, hidden_size])  # hidden_size+num_labels

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            logits = tf.reshape(logits, [-1, self.max_seq_length, num_labels])
            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            one_hot_labels = tf.one_hot(label_ids, depth=num_labels, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)

            predict = tf.argmax(probabilities, axis=-1)

            return (loss, per_example_loss, logits, predict)

    def model_fn_builder(self, bert_config, num_labels, init_checkpoint,
                         learning_rate=None, num_train_steps=None, num_warmup_steps=None):
        """Returns `model_fn` closure for TPUEstimator."""

        def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""

            tf.compat.v1.logging.info("*** Features ***")
            for name in sorted(features.keys()):
                tf.compat.v1.logging.info(f"  name = {name}, shape = {features[name].shape}")

            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            (total_loss, per_example_loss, logits, predicts) = self.create_model(bert_config, is_training, features, num_labels)

            tvars = tf.compat.v1.trainable_variables()
            initialized_variable_names = {}

            if init_checkpoint:
                (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

            tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
            tf.compat.v1.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

            output_spec = None
            if learning_rate is not None and num_train_steps is not None and num_warmup_steps is not None:
                if mode == tf.estimator.ModeKeys.TRAIN:
                    # use_tpu=None
                    
                    train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps, None)

                    output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                        mode=mode,
                        loss=total_loss,
                        train_op=train_op,
                        scaffold_fn=None)

            if mode == tf.estimator.ModeKeys.EVAL:

                def metric_fn(per_example_loss, label_ids, logits):  # is_real_example
                    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

                    pos_labels_idx = range(1, 17)

                    precision = tf_metrics.precision(labels=label_ids, predictions=predictions, num_classes=num_labels, pos_indices=pos_labels_idx, average="macro")
                    recall = tf_metrics.recall(labels=label_ids, predictions=predictions, num_classes=num_labels, pos_indices=pos_labels_idx, average="macro")
                    f = tf_metrics.f1(labels=label_ids, predictions=predictions, num_classes=num_labels, pos_indices=pos_labels_idx, average="macro")

                    loss = tf.compat.v1.metrics.mean(values=per_example_loss)
                    return {
                        "eval_precision":precision,
                        "eval_recall":recall,
                        "eval_f": f,
                        "eval_loss": loss,
                    }

                label_ids = features["label_ids"]
                eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
                        
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metrics=eval_metrics,
                    scaffold_fn=None)

            elif mode == tf.estimator.ModeKeys.PREDICT:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions=predicts,  # {"probabilities": probabilities},
                    scaffold_fn=None)

            return output_spec

        return model_fn


class BERTNERTrainer:

    def __init__(self, data_dir, labels_path, output_dir, bert_dir, model_dir, max_seq_length=128,
                 save_checkpoints_steps=1000, learning_rate=5e-5,
                 train_batch_size=8, eval_batch_size=8, num_train_epochs=1, warmup_proportion=0.1,
                 drop_remainder=True
                 ):

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        bert_config_file = os.path.join(bert_dir, 'bert_config.json')
        init_checkpoint = os.path.join(bert_dir, 'bert_model.ckpt')
        vocab_path = os.path.join(bert_dir, 'vocab.txt')

        # Load training data
        self.db = DataBuilder(data_dir, labels_path, vocab_path, output_dir, max_seq_length, drop_remainder, mode="train")

        assert not drop_remainder or drop_remainder and self.db.num_examples > train_batch_size

        # Build model for training
        self.num_train_steps = int(self.db.num_examples / train_batch_size * num_train_epochs)
        __mb = ModelBuilder(
            self.db.num_labels, model_dir, bert_config_file, init_checkpoint, max_seq_length,
            save_checkpoints_steps,
            self.num_train_steps, warmup_proportion, learning_rate,
            train_batch_size, eval_batch_size)
        self.estimator = __mb.estimator

    def train(self):
        train_input_fn = self.db.make_input_fn(mode='train')
        self.estimator.train(input_fn=train_input_fn, max_steps=self.num_train_steps)

    def evaluate(self):
        eval_input_fn = self.db.make_input_fn(mode='eval')    
        result = self.estimator.evaluate(input_fn=eval_input_fn)
        self.db.export_result(result)


class BERTNERPredictor:

    def __init__(self, labels_path, output_dir, bert_dir, model_dir,
                    data_dir=None,
                    max_seq_length=128, predict_batch_size=8,
                    drop_remainder=True):


        bert_config_file = os.path.join(bert_dir, 'bert_config.json')
        init_checkpoint = os.path.join(bert_dir, 'bert_model.ckpt')
        vocab_path = os.path.join(bert_dir, 'vocab.txt')

        self.db = DataBuilder(data_dir, labels_path, vocab_path, output_dir, max_seq_length, drop_remainder, mode="predict")
        assert drop_remainder and self.db.num_examples > predict_batch_size

        # Build model for prediction
        
        __mb = ModelBuilder(self.db.num_labels, model_dir, bert_config_file, init_checkpoint, max_seq_length,
                            predict_batch_size=predict_batch_size)
        self.estimator = __mb.estimator
        self.predict_batch_size = predict_batch_size

    def predict(self, sentences=None, subword=False):
        if sentences is None:
            predict_input_fn = self.db.make_input_fn(mode="predict")
            gold = True
        else:
            predict_input_fn = self.db.make_input_fn_from_sentences(sentences)
            gold = False

        # prediction & make sequence tags word-wise
        label_ids_pred = self.estimator.predict(input_fn=predict_input_fn)
        token_ids, label_ids_gold = [], []
        for input_batch in predict_input_fn({'batch_size': self.predict_batch_size}):
            for ids in input_batch['input_ids'].numpy():
                token_ids.append(ids)
            for labels in input_batch['label_ids'].numpy():
                label_ids_gold.append(labels)
        assert len(token_ids) == len(label_ids_gold)

        # convert subwords-unit to words-unit
        tokens_list, labels_list_pred, labels_list_gold =\
            self.db.swc.convert_ids_to_surfaces_list(token_ids, label_ids_pred, label_ids_gold, subword=subword)

        if gold:
            return [[{'token': token, 'pred': label_pred, 'gold': label_gold}
                        for token, label_pred, label_gold in zip(tokens, labels_pred, labels_gold)]
                    for tokens, labels_pred, labels_gold in zip(tokens_list, labels_list_pred, labels_list_gold)]
        else:
            return [[{'token': token, 'pred': label_pred}
                        for token, label_pred in zip(tokens, labels_pred)]
                    for tokens, labels_pred in zip(tokens_list, labels_list_pred)]


if __name__=='__main__':
    data_dir = '../input'  # must contain 'train.txt', 'dev.txt', 'test.txt'
    labels_path = '../input/labels_enesub.txt'
    output_dir = '../output_result'
    model_dir = '../model_result'
    bert_dir = '../Japanese_L-12_H-768_A-12_E-30_BPE'

    bert_trainer = BERTNERTrainer(
                    data_dir,
                    labels_path,
                    output_dir,
                    bert_dir,
                    model_dir,
                    max_seq_length=128,
                    save_checkpoints_steps=1000, learning_rate=5e-5,
                    train_batch_size=8, eval_batch_size=8, warmup_proportion=0.1,
                    num_train_epochs=1,
                    drop_remainder=True
                    )
    bert_trainer.train()
    bert_trainer.evaluate()

    bert_predictor = BERTNERPredictor(
                    labels_path,
                    output_dir,
                    bert_dir,
                    model_dir,
                    data_dir=data_dir,
                    max_seq_length=128,
                    predict_batch_size=8,
                    drop_remainder=True
                    )
    bert_predictor.predict()