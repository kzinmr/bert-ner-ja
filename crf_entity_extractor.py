from tqdm import tqdm

import pickle
import multiprocessing
# from pyknp import KNP
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
import numpy as np
from typing import List, Dict

with open('input_kb/labels_enesub.txt') as f:
    LABELS = f.read().split('\n')


class Sentence:

    def __init__(self, sentence_str: str):
        self.entries = []
        self.labels = []
        for line in sentence_str.split('\n'):
            entry = line.split(' ')[:-1]
            label = line.split(' ')[-1]
            if not entry:
                continue
            self.entries.append(entry)
            self.labels.append(label)

        self.n = len(self.labels)


class FeatureExtractor:
    
    def __init__(self):
        pass

    @staticmethod
    def __select(knowledgebase_enes):
        return knowledgebase_enes.split('##')[0]

    def features(self, entry):
        features = {
            "bias": 1.0,
            "word": entry[0],
            "postag": entry[1],
            "chunktag": entry[2],
            "kb_match": entry[3]
        }
        # for enetype in entry[3].split('##')[:1]:
        #     features[enetype] = 1
        return features

    def window_features(self, entry, w):
        features = {
            f"{w}:word": entry[0],
            f"{w}:postag": entry[1],
            f"{w}:chunktag": entry[2],
            f"{w}:kb_match": entry[3]
        }
        # for enetype in entry[3].split('##')[:1]:
        #     features[f"{w}:{enetype}"] = 1
        return features

    def extract_feature(self, sentence: Sentence):
        features = []
        for i, entry in enumerate(sentence.entries):
            fe = self.features(entry)
            if i > 0:
                entry_prev = sentence.entries[i - 1]
                fe.update(self.window_features(entry_prev, '-1'))
            else:
                fe["BOS"] = True
            if i > 1:
                entry_preprev = sentence.entries[i - 2]
                fe.update(self.window_features(entry_preprev, '-2'))

            if i < sentence.n - 1:
                entry_next = sentence.entries[i + 1]
                fe.update(self.window_features(entry_next, '+1'))
            else:
                fe["EOS"] = True
            if i < sentence.n - 2:
                entry_postnext = sentence.entries[i + 2]
                fe.update(self.window_features(entry_postnext, '+2'))

            features.append(fe)

        return features

def _beam_search_decoder(args):
    probability, k = args
    topk_candidates = [(list(), 1.0)]
    for row in probability:
        topk_candidates = sorted([(seq + [i], score * -np.log(prob))  # NOTE: prob==0.?
                                    for seq, score in topk_candidates
                                    for i, prob in enumerate(row)],
                                    key=lambda x: x[1])[:k]
    return topk_candidates

def _is_valid(labels):
    prev_bio, prev_netype, netype = '', '', ''
    for l in labels:
        if len(l.split('-')) == 2:
            bio, netype = l.split('-')
        else:
            bio = l
        # check two bad patterns like ['O', 'I-A', 'O'] or ['B-A', 'I-B', 'O']
        if prev_bio == 'O' and bio == 'I' or prev_bio == 'B' and bio == 'I' and prev_netype != netype:
            return False
        prev_bio = bio
        prev_netype = netype
    return True

def _filter_valid(args):
    for labels in args:
        if _is_valid(labels):
            return labels
    return args[0]

def _remove_invalid_labels(labels):
    if _is_valid(labels):
        return labels
    prev_bio, prev_netype, netype = '', '', ''
    i = 0
    remove_indices = []
    while i < len(labels):
        l = labels[i]
        if len(l.split('-')) == 2:
            bio, netype = l.split('-')
        else:
            bio = l
        # check two bad patterns like ['O', 'I-A', 'O'] or ['B-A', 'I-B', 'O']
        if prev_bio == 'O' and bio == 'I':
            remove_from = i
            j = i + 1
            while labels[j].split('-')[0] != 'O':
                j += 1
            remove_to = j
            remove_indices.append((remove_from, remove_to))
            i = j
        elif prev_bio == 'B' and bio == 'I' and prev_netype != netype:
            remove_from = i
            j = i + 1
            while labels[j].split('-')[0] == 'I':
                j += 1
            remove_to = j
            remove_indices.append((remove_from, remove_to))
            i = j
        else:
            i += 1
        prev_bio = bio
        prev_netype = netype

    for (rm_from, rm_to) in remove_indices:
        labels[rm_from: rm_to] = ['[NULL]' for _ in range(rm_to - rm_from)]
    return labels

def _fix_valid(args):
    return [_remove_invalid_labels(labels) if _is_valid(labels) else labels for labels in args]

class EntityExtractor:

    def __init__(
        self,
        hyper_params: Dict[str, float] = None,
        model_path: str = None,
     ):
        if model_path:
            self.load_model(model_path=model_path)
        else:
            algorithm = (
                hyper_params["algorithm"]
                if hyper_params and "algorithm" in hyper_params
                else "lbfgs"
            )
            c1 = hyper_params["c1"] if hyper_params and "c1" in hyper_params else 0.1
            c2 = hyper_params["c2"] if hyper_params and "c2" in hyper_params else 0.1
            max_iters = (
                hyper_params["max_iterations"]
                if hyper_params and "max_iterations" in hyper_params
                else 100
            )
            apt = (
                hyper_params["all_possible trainsitions"]
                if hyper_params and "max_iterations" in hyper_params
                else True
            )

            self.fe = FeatureExtractor()

            self.crf = CRF(
                algorithm=algorithm,
                c1=c1,
                c2=c2,
                max_iterations=max_iters,
                all_possible_transitions=apt,
            )

    def save_model(self, output_path: str):
        """
        save model
        """
        with open(output_path, "wb") as f:
            pickle.dump(self.crf, f)

    def load_model(self, model_path: str):
        """
        load model
        """
        with open(model_path, "rb") as f:
            self.crf = pickle.load(f)

    def train(self, sentences: List[Sentence]):
        """
        execute training
        """
        x = [self.fe.extract_feature(s) for s in sentences]
        y = [s.labels for s in sentences]
        self.crf.fit(x, y)

    def evaluate(self, sentences: List[Sentence], fix_invalid_labels=True, decoder='greedy', k=5, debug=False):
        """
        return predicted class
        """
        x = [self.fe.extract_feature(s) for s in sentences]
        y_test = [s.labels for s in sentences]
        y_pred = self.__predict(x, decoder, k)
        if debug:
            for t, p in list(zip(y_test, y_pred))[:10]:
                print(t)
                print(p)

        print('raw, ', decoder)
        print(metrics.flat_classification_report(
            y_test, y_pred, labels=sorted(LABELS), digits=3
        ))        
        if fix_invalid_labels:
            print('fix-invalid, ', decoder)
            y_pred = self.fix_labels(y_pred)
            if debug:
                for t, p in list(zip(y_test, y_pred))[:10]:
                    print(t)
                    print(p)
            print(metrics.flat_classification_report(
                y_test, y_pred, labels=sorted(LABELS), digits=3
            ))
        return metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=LABELS)

    def fix_labels(self, labels):
        labels_fixed = []
        with multiprocessing.Pool() as p:
            labels_fixed.append(p.map(_fix_valid, labels))
        return labels_fixed[0]

    def __predict(self, x, decoder='greedy', k=5):
        if decoder == 'greedy':
            return self.crf.predict(x)
        else:
            y_probs = self.__predict_prob(x)
            # (n_data, max_sequence_length, n_labels) -> (n_data, k, max_sequence_length)
            args = [(probability, k) for probability in y_probs]
            results = []
            with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                results.append(p.map(_beam_search_decoder, args))
            beams_list = results[0]

            # validのみ残す
            args = [[[LABELS[idx-1] if idx > 0 else 'O' for idx in beam[0]] for beam in beams] for beams in beams_list]
            labels_selected = []
            with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                labels_selected.append(p.map(_filter_valid, args))
                labels_selected = labels_selected[0]

            return labels_selected

    def predict(self, sentences: List[Sentence], decoder='greedy', k=5):
        """
        return predicted class
        """
        x = [self.fe.extract_feature(s) for s in sentences]
        return self.__predict(x, decoder, k)

    def __predict_prob(self, x):
        probs = self.crf.predict_marginals(x)
        # (n_data, max_sequence_length, n_labels)
        return [[[p_token['O']] + [p_token[label] for label in LABELS]
                 for p_token in prob]
                for prob in probs]

    def predict_prob(self, sentences: List[Sentence]):
        """
        return probabilities for each class
        :param data:
        :return:
        """
        x = [self.fe.extract_feature(s) for s in sentences]
        return self.__predict_prob(x)
        


def read_data(filepath):
    with open(filepath) as f:
        return [Sentence(sentence_str) for sentence_str in f.read().split('\n\n')]

if __name__ == "__main__":
    model_path = 'crf_gazetter_multi.pkl'

    train = False
    if train:
        train_path = 'input_kb/train.txt'  # 'NERdata_ene_single_cv_5/NERdata_0/train.txt'
        sentences_train = read_data(train_path)
        print('data loaded')
        ee = EntityExtractor()
        ee.train(sentences_train)
        ee.save_model(model_path)

    test_path = 'input_kb/test.txt'  # 'NERdata_ene_single_cv_5/NERdata_0/test.txt'
    sentences_test = read_data(test_path)
    ee = EntityExtractor()
    ee.load_model(model_path)
    ee.evaluate(sentences_test)
    ee.evaluate(sentences_test, decoder='beam')
    # result = ee.predict(sentences_test, decoder='greedy')
    # with open('crfresult_test_gazetter_multi.pkl', 'wb') as f:
    #     pickle.dump(result, f)
