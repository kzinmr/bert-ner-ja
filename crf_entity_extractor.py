from tqdm import tqdm

import pickle
# from pyknp import KNP
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from typing import List, Dict

LABELS = ['Person',
'Organization_Other',
'International_Organization',
'Show_Organization',
'Family',
'Ethnic_Group_Other',
'Nationality',
'Sports_Organization_Other',
'Pro_Sports_Organization',
'Sports_League',
'Corporation_Other',
'Company',
'Company_Group',
'Political_Organization_Other',
'Government',
'Political_Party',
'Cabinet',
'Military',
'GPE_Other',
'City',
'County',
'Province',
'Country',
'GOE_Other',
'Public_Institution',
'School',
'Research_Institute',
'Market',
'Park',
'Sports_Facility',
'Museum',
'Zoo',
'Amusement_Park',
'Theater',
'Worship_Place',
'Car_Stop',
'Station',
'Airport',
'Port',
'Position_Vocation']


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
            # "kb_match": self.__select(entry[2])
        }
        for enetype in entry[2].split('##')[:1]:
            features[enetype] = 1
        return features

    def window_features(self, entry, w):
        features = {
            f"{w}:word": entry[0],
            f"{w}:postag": entry[1],
            # f"{w}:kb_match": self.__select(entry[2])
        }
        for enetype in entry[2].split('##')[:1]:
            features[f"{w}:{enetype}"] = 1
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

    def evaluate(self, sentences: List[Sentence]):
        """
        return predicted class
        """
        x = [self.fe.extract_feature(s) for s in sentences]
        y_test = [s.labels for s in sentences]
        y_pred = self.crf.predict(x)
        print(metrics.flat_classification_report(
            y_test, y_pred, labels=sorted(LABELS), digits=3
        ))        
        return metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=LABELS)

    def predict(self, sentences: List[Sentence]):
        """
        return predicted class
        """
        x = [self.fe.extract_feature(s) for s in sentences]
        return self.crf.predict(x)

    def predict_prob(self, sentences: List[Sentence]):
        """
        return probabilities for each class
        :param data:
        :return:
        """
        x = [self.fe.extract_feature(s) for s in sentences]
        return self.crf.predict_marginals(x)


def read_data(filepath):
    with open(filepath) as f:
        return [Sentence(sentence_str) for sentence_str in f.read().split('\n\n')]


if __name__ == "__main__":
    model_path = 'crf_gazetter_single.pkl'

    train_path = 'NERdata_ene_single_cv_5/NERdata_0/train.txt'
    sentences_train = read_data(train_path)
    print('data loaded')
    ee = EntityExtractor()
    ee.train(sentences_train)
    ee.save_model(model_path)

    test_path = 'NERdata_ene_single_cv_5/NERdata_0/test.txt'
    sentences_test = read_data(test_path)
    ee = EntityExtractor()
    ee.load_model(model_path)
    result = ee.predict(sentences_test)
    with open('crfresult_test_gazetter_single.pkl', 'wb') as f:
        pickle.dump(result, f)
