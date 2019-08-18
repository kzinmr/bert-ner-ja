import sys
import regex as re
import mojimoji
from pyknp import KNP, BList, MList, Juman, Bunsetsu, Tag, TList
from typing import Tuple, Set, Iterable, Generator, Optional, Union, Dict

class KnpError(Exception):
    pass


class MorphTooLongError(KnpError):
    pass


class CharTypeTooLongError(KnpError):
    pass


class InputTooLongError(KnpError):
    pass


class NoEOSError(KnpError):
    pass


class NoResultError(KnpError):
    pass



class KnpBase:
    """KNPでのパース結果&汎用的な処理をまとめたベースクラス。特化した処理はこれを継承して書いてください。"""

    MAX_CONSECUTIVENESS: int = 42
    MAX_JUMAN_BYTELENGTH: int = 4096
    MAX_JUMAN_MORPHLENGTH: int = 200
    text: str = ""
    juman_result = None
    result: Optional[BList] = None
    bnst_list: Optional[BList] = None
    tag_list: Optional[TList] = None
    tag_id2bnst: Dict[int, Bunsetsu] = {}

    def __init__(
        self,
        knp: Optional[KNP] = None,
        jumanpp: bool = True,
        fallback_juman: bool = True,
    ):
        self.knp = KNP(jumanpp=jumanpp) if knp is None else knp
        self.juman = self.knp.juman
        self.knp.parse("。")  # self.knp.socketやsubprocessを生成させるため
        self.fallback_juman = fallback_juman

    @classmethod
    def __contains_toolong_character(cls, s: str, character_pattern: str) -> bool:
        rs = re.findall(character_pattern, s)
        return rs and max([len(r) for r in rs]) > cls.MAX_CONSECUTIVENESS

    def contains_toolong_alphabet(self, s: str) -> bool:
        return self.__contains_toolong_character(
            s, r"[^\p{Hiragana}\p{Katakana}\p{Han}\p{N}]+"
        )

    def contains_toolong_number(self, s: str) -> bool:
        return self.__contains_toolong_character(s, r"[\p{N}.．]+")

    def contains_toolong_han(self, s: str) -> bool:
        return self.__contains_toolong_character(s, r"[\p{Han}]+")

    def contains_toolong_kana(self, s: str) -> bool:
        return self.__contains_toolong_character(s, r"[\p{Katakana}ー]+")

    def contains_toolong_hira(self, s: str) -> bool:
        return self.__contains_toolong_character(s, r"[\p{Hiragana}ー]+")

    @staticmethod
    def generate_knp_lines(line_iter: Iterable[str]) -> Generator[str, None, None]:
        data = ""
        for line in line_iter:
            data += line
            if line.strip() == "EOS":
                yield data.strip()
                data = ""

    @staticmethod
    def prenormalize(text: str) -> str:
        """
        今はとりあえず全角化だけ
        """
        return mojimoji.han_to_zen(text.replace("\t", " ").replace("\r", ""))

    def __juman_lines(self, text: str, jumanpp: Optional[bool] = None) -> str:

        if len(text.encode("utf8")) > self.MAX_JUMAN_BYTELENGTH:
            raise InputTooLongError(
                f'input size is {len(text.encode("utf8"))} > {self.MAX_JUMAN_BYTELENGTH}'
            )
        if self.contains_toolong_alphabet(text):
            raise CharTypeTooLongError(f"too long alphabets")
        elif self.contains_toolong_number(text):
            raise CharTypeTooLongError(f"too long numbers")
        elif self.contains_toolong_han(text):
            raise CharTypeTooLongError(f"too long han")
        elif self.contains_toolong_kana(text):
            raise CharTypeTooLongError(f"too long katakana")
        elif self.contains_toolong_hira(text):
            raise CharTypeTooLongError(f"too long hiragana")
        if jumanpp is None:
            return self.juman.juman_lines(text)
        else:
            return Juman(jumanpp=jumanpp).juman_lines(text)

    def juman_parse(self, text: str, jumanpp: Optional[bool] = None) -> MList:
        normalized_text = self.prenormalize(text)
        return MList(self.__juman_lines(normalized_text, jumanpp))

    def wakati(self, text: str) -> str:
        return " ".join(m.midasi if m else "" for m in self.juman_parse(text))

    def __juman_parse(self, text: str, jumanpp: Optional[bool] = None) -> str:
        juman_str = self.__juman_lines(text, jumanpp)
        m_length = len(MList(juman_str))
        if m_length < self.MAX_JUMAN_MORPHLENGTH:
            return "%s%s" % (juman_str, self.knp.pattern)
        else:
            raise MorphTooLongError(f"{m_length} morphs > {self.MAX_JUMAN_MORPHLENGTH}")

    def __knp_parse(self, juman_str: str) -> BList:
        if self.knp.socket:
            knp_lines = self.knp.socket.query(
                juman_str, pattern=r"^%s$" % self.knp.pattern
            )
        else:
            knp_lines = self.knp.subprocess.query(
                juman_str, pattern=r"^%s$" % self.knp.pattern
            )
        return BList(knp_lines, self.knp.pattern)

    def __register_attributes(self, blist: BList):
        self.result = blist
        self.bnst_list = self.result.bnst_list()
        self.tag_list = self.result.tag_list()
        self.mrph_list = self.result.mrph_list()
        self.tag_id2bnst = {t.tag_id: b for b in self.bnst_list for t in b.tag_list()}

    def parse_juman_result(self, juman_str: str) -> BList:
        blist = self.__knp_parse(juman_str)
        self.__register_attributes(blist)
        return blist

    def load_knp_result(self, knp_lines: str) -> BList:
        if knp_lines.strip().endswith("EOS"):
            blist = BList(knp_lines.strip(), self.knp.pattern)
            self.__register_attributes(blist)
            return blist
        else:
            raise NoEOSError

    def reparse_knp_result(self, knp_lines: str) -> BList:
        if knp_lines.strip().endswith("EOS"):
            blist = self.__knp_parse(knp_lines.strip())
            self.__register_attributes(blist)
            return blist
        else:
            raise NoEOSError

    def parse(self, text: str) -> BList:
        self.text = text
        normalized_text = self.prenormalize(text)
        juman_str = self.__juman_parse(normalized_text)

        if self.knp.jumanpp and self.fallback_juman:
            try:
                blist = self.__knp_parse(juman_str)
            except Exception as e:
                print(e, file=sys.stderr)
                juman_str = self.__juman_parse(normalized_text, jumanpp=False)
                blist = self.__knp_parse(juman_str)
        else:
            blist = self.__knp_parse(juman_str)

        self.__register_attributes(blist)
        return blist

    @staticmethod
    def unit_midasi(
        unit: Union[Bunsetsu, Tag],
        ignore_pos: Set[str] = None,
        ignore_genkei: Set[str] = None,
        genkei: bool = False,
    ) -> str:
        """
        Args:
            unit (pyknp.Bunsetsu|pyknp.Tag): 見出し語を抽出する単位
            ignore_pos: 抽出しない品詞 (空ならすべて抽出)
            ignore_genkei: 抽出しない語の原形
            genkei: 用言を原形化して返すオプション
                原形化した際にどの範囲の付属語を残すかは ignore_pos/ignore_genkei で制御
        Returns:
            unit に含まれる見出し語の結合
        """
        if ignore_pos is None:
            ignore_pos = set()
        if ignore_genkei is None:
            ignore_genkei = set()
        mrphs = [
            m
            for m in unit.mrph_list()
            if m.hinsi not in ignore_pos and m.genkei not in ignore_genkei
        ]
        if not mrphs:
            return ""
        if genkei:
            if len(mrphs) == 1:
                return unit.mrph_list()[0].genkei
            return "".join([m.midasi for m in mrphs[:-1]]) + mrphs[-1].genkei
        else:
            return "".join([m.midasi for m in mrphs])

    @classmethod
    def unit_genkei(
        cls,
        unit: Union[Bunsetsu, Tag],
        ignore_pos: Set[str] = None,
        ignore_genkei: Set[str] = None,
    ) -> str:
        return cls.unit_midasi(
            unit, ignore_pos=ignore_pos, ignore_genkei=ignore_genkei, genkei=True
        )

    def get_tag_span(self, tag_id: int) -> Tuple[int, int]:
        """ tag_idで指定される基本句(tag)がまたがる、元文内における文字列範囲(スパン)を返す
        注. 以下のようにスパンは終端を含む:
            例. '日本経済新聞の記者' において、 tag_id('新聞の') のスパン -> (0,2)
            例. '風邪をひく' において、 tag_id('ひく') のスパン -> (3,4)
        """
        if self.result is None:
            raise NoResultError
        return self.result.get_tag_span(tag_id)

    def get_bnst_span(self, bnst: Bunsetsu) -> Tuple[int, int]:
        """ 文節内の基本句すべてがまたがる、元文内における文字列範囲(スパン)を返す
            例. '日本経済新聞の記者' において、 bnst('日本経済新聞の') のスパン -> (0,6)
        """

        def merge_spans(continuous_spans):
            if len(continuous_spans) < 2:
                return continuous_spans[0]
            return continuous_spans[0][0], continuous_spans[-1][1]

        return merge_spans([self.get_tag_span(t.tag_id) for t in bnst.tag_list()])

    @staticmethod
    def is_verbal(tag: Tag) -> bool:
        """
         - サ変名詞(増加、転換、推奨)
         - 用言連用形名詞化(怒り、陰り、明るみ)
         - とれない: １０％増 (増: 接尾辞 and 内容語)
         - とれない: 名詞性述語接尾辞: 怖さ(さ/さ), 高み(み/み), (形容詞性: 少なめ(めだ/めだ))
        """
        if "体言" in tag.features and ("サ変" in tag.features or tag.repname.endswith("v")):
            return True
        return False

    @staticmethod
    def extract_tense(tag: Tag) -> Set[str]:
        tenses = set()
        if "時制-過去" in tag.features:
            tenses.add("過去")
        if "時制-未来" in tag.features:
            tenses.add("未来")
        if "時制-現在" in tag.features:
            tenses.add("現在")
        if "時制-無時制" in tag.features:
            tenses.add("無時性")
        return tenses

    @staticmethod
    def extract_voice(tag):
        return tag.features.get("態", "能動")

    @staticmethod
    def extract_negation(tag: Tag) -> bool:
        """
        n>1重否定は注目する述語に対して否定表現のmod(2)をとる必要がある
        """
        if (
            "準否定表現" in tag.features or "否定表現" in tag.features
        ) and "二重否定" not in tag.features:
            return True
        return False

    @staticmethod
    def is_ability(tag: Tag) -> bool:
        return "可能表現" in tag.features or tag.features.get("態", "") == "可能"

    @staticmethod
    def parse_keigo(tag: Tag) -> str:
        if "敬語" not in tag.features:
            return ""
        sym = ""
        keigo = tag.features["敬語"]
        if keigo == "尊敬表現":
            for m in tag.mrph_list():
                for r in re.findall("<([^<>]+)>", m.fstring):
                    if r.startswith("尊敬動詞:"):
                        sym = r.split(":")[1]
        elif keigo == "謙譲表現":
            for m in tag.mrph_list():
                for r in re.findall("<([^<>]+)>", m.fstring):
                    if r.startswith("謙譲動詞"):
                        sym = r.split(":")[1]
        return f"{keigo}:{sym}"
