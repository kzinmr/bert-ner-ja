# import pysnooper
from knp_base import KnpBase


def convert_conll_format(tokens, delimiter=' '):
    return '\n'.join(delimiter.join([t["word"], t["pos"], t["tag"]])
                     for t in tokens)


class Span2BIO:

    def __init__(self, tokenizer, text):
        self.tokenizer = tokenizer
        assert self.tokenizer.wakati('') == ''
        # with pysnooper.snoop():
        words = self.tokenizer.juman_parse(text)
        # 単語のspanを取得 (単語がskipされていることも考慮するのでspanは必ずしも連続しない)
        spans = list(self.generate_spans(text, words))
        self.tokens = [{'span': span,
                        'word': mrph.midasi,
                        'pos': mrph.hinsi,
                        'tag': 'O'}
                       for span, mrph in zip(spans, words)]

        self.word_list = [wd['word'] for wd in self.tokens]
        self.begin2wordid = {wd['span'][0]: i
                             for i, wd in enumerate(self.tokens)}
        self.end2wordid = {wd['span'][1]: i
                           for i, wd in enumerate(self.tokens)}

    def generate_spans(self, text, words):
        offset = 0
        text_tmp = text
        for word in words:
            word_surf = word.midasi
            while text_tmp and not text_tmp.startswith(word_surf):
                text_tmp = text_tmp[1:]
                offset += 1

            if text_tmp.startswith(word_surf):
                yield (offset, offset+len(word_surf))

    def register_tag(self, span_query, token_type):
        if span_query[0] in self.begin2wordid and\
           span_query[1] in self.end2wordid:
            begin_id = self.begin2wordid[span_query[0]]
            end_id = self.end2wordid[span_query[1]]
            if token_type:
                token = self.tokens[begin_id]
                token['tag'] = f'B-{token_type}'

                for token in self.tokens[begin_id: end_id+1][1:]:
                    token['tag'] = f'I-{token_type}'

        # 解析結果の形態素単位が始点・終点に合わないとNEタグ付けができない
        # 例. 始点や終点が単語内部
#         elif token_type:
#             print(span_query, token_type)
#             if span_query[0] in self.begin2wordid:
#                 print(self.word_list[self.begin2wordid[span_query[0]]])
#             elif span_query[1] in self.end2wordid:
#                 print(self.word_list[self.end2wordid[span_query[1]]])


def tokenize_and_convert_sentence(tokenizer, sentence, entities, type_key='type',
                         begin_at=0):
    # sentenceになりindexが0から始まることへのspan修正
    # sentenceは１文字の区切り文字で区切られると想定
    end_at = begin_at + len(sentence)
    entities_sentence = [
        {
            'span': (e['span'][0] - begin_at, e['span'][1] - begin_at),
            type_key: e[type_key],
        }
        for e in entities
        if e['span'][0] >= begin_at and e['span'][0] < end_at
    ]
    # 文ごとにデータ登録
    try:
        span2bio = Span2BIO(tokenizer, sentence)
    except Exception:
        return []

    for e in entities_sentence:
        span2bio.register_tag(e['span'], e[type_key])
    return span2bio.tokens


def tokenize_and_convert(tokenizer, text, entities, type_key='type', delimiter=' '):

    # sentence splitter
    sentences = text.split('。')
    sentences = [s + '。' for s in sentences]

    conll_lines = []
    for i, sentence in enumerate(sentences):
        try:
            # spanの絶対位置を合わせるため各文の文字offsetを保持
            offset = len(''.join(sentences[:i])) if i > 0 else 0
            tokens = tokenize_and_convert_sentence(tokenizer, sentence, entities, type_key, offset)
            conll_line = convert_conll_format(tokens, delimiter)
            if conll_line.strip():
                conll_lines.append(conll_line)
        except ValueError:
            print('pass', len(text.encode('utf8')))
            continue

    return '\n\n'.join(conll_lines)


if __name__ == '__main__':
    from tqdm import tqdm
    import json

    # {'id': '', 'text': '', 'entities': [] }
    # {'entity': '', 'type': '', 'irex_type': '', 'span': ()}

    from pprint import PrettyPrinter

    pp = PrettyPrinter()

    from pathlib import Path

    i = 0
    # set where `gsk-ene-1.1-bccwj-json` is
    inpath = Path('gsk-ene-1.1-bccwj-json/')
    assert inpath.exists()

    outpath = Path('gsk-ene-1.1-bccwj-json-jumanpp-type/')
    outpath.mkdir(exist_ok=True)
    knp = KnpBase()
    #   dict_format='unidic', dictionary_path='/usr/local/lib/mecab/dic/unidic')
    import sys
    conll_txts = ''
    for f in tqdm(inpath.glob('*/*.json'), file=sys.stdout):
        with open(f) as fi:
            jd = json.load(fi)
        text = jd['text']
        entities = jd['entities']

        if not any(bool(e['type']) for e in entities):  # IREX_NETYPEが少なくとも１つ含まれる文のみ対象
            continue
        else:
            i += 1
            # try:
            conll_txt = tokenize_and_convert(knp, text, entities, 'type')
            if conll_txt:
                p = Path(f)
                op = outpath / p.parent.stem
                op.mkdir(exist_ok=True)
                out_filepath = op / Path(f'{p.stem}.txt')
                with open(out_filepath, 'wt') as of:
                    of.write(conll_txt)
                    of.write('\n')  # 単純にcatすればよくするため
                conll_txts += conll_txt + '\n'
            # except Exception as e:
            #     print(e)
            #     print('boo')
    out_filepath = outpath / Path(f'bccwj-ene-jumanpp-type.txt')
    conll_txts = '\n\n'.join([s for s in conll_txts.split('\n\n') if s])
    with open(out_filepath, 'wt') as of:
        of.write(conll_txts)
        of.write('\n')
