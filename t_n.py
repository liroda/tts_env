# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
from typing import List

from .lang_deal import rm_empty_char
from .address_convert import address_replace
from .fullTohalf import stringpartQ2B

from .char_convert import tranditional_to_simplified
from .chronology import RE_DATE
from .chronology import RE_DATE2
from .chronology import RE_TIME
from .chronology import RE_TIME_RANGE
from .chronology import replace_date
from .chronology import replace_date2
from .chronology import replace_time
from .constants import F2H_ASCII_LETTERS
from .constants import F2H_DIGITS
from .constants import F2H_SPACE
from .num import RE_DECIMAL_NUM
from .num import RE_DEFAULT_NUM
from .num import RE_FRAC
from .num import RE_INTEGER
from .num import RE_NUMBER
from .num import RE_PERCENTAGE
from .num import RE_POSITIVE_QUANTIFIERS
from .num import RE_RANGE
from .num import replace_default_num
from .num import replace_frac
from .num import replace_negative_num
from .num import replace_number
from .num import replace_percentage
from .num import replace_positive_quantifier
from .num import replace_range
from .num import replace_express_bill_num
from .num import RE_EXPRESS_BILL_NUM
from .phonecode import RE_MOBILE_PHONE
from .phonecode import RE_NATIONAL_UNIFORM_NUMBER
from .phonecode import RE_TELEPHONE
from .phonecode import replace_mobile
from .phonecode import replace_phone
from .quantifier import RE_TEMPERATURE
from .quantifier import replace_temperature
import traceback


class TextNormalizer():
    def __init__(self):
        self.SENTENCE_SPLITOR = re.compile(r'([：、，；。？！,;?!$][”’]?)')

    def _split(self, text: str, lang="zh") -> List[str]:
        """Split long text into sentences with sentence-splitting punctuations.
        Args:
            text (str): The input text.
        Returns:
            List[str]: Sentences.
        """
        # Only for  Chinese and mix(chinese and english) here
        if lang == "zh":
            text = rm_empty_char(text)
            # 过滤掉特殊字符
            text = re.sub(r'[+《》【】<=>{}()（）&@“”^_|…\\]', '', text)  # by hsl
            #text = re.sub(r'[《》【】<=>{}()（）#&@“”^_|…\\]', '', text)
            # 场景地址 1-3-701 1-3  FA-1b
            text = address_replace(text)
        text = self.SENTENCE_SPLITOR.sub(r'\1\n', text)
        #print (text)
        # add TAG  # by hsl
        text = re.sub(r'$','',text)
        text = text.strip()
        sentences = [sentence.strip() for sentence in re.split(r'\n+', text)]
        return sentences

    def _post_replace(self, sentence: str) -> str:
        #sentence = sentence.replace('/', '每')
        sentence = sentence.replace('~', '至')

        return sentence

    def normalize_sentence(self, sentence: str) -> str:
        try:
            # basic character conversions
            sentence = tranditional_to_simplified(sentence)
            sentence = sentence.translate(F2H_ASCII_LETTERS).translate(F2H_DIGITS).translate(F2H_SPACE)

            # number related NSW verbalization
            sentence = RE_DATE.sub(replace_date, sentence)
            sentence = RE_DATE2.sub(replace_date2, sentence)

            # range first
            sentence = RE_TIME_RANGE.sub(replace_time, sentence)
            sentence = RE_TIME.sub(replace_time, sentence)

            sentence = RE_TEMPERATURE.sub(replace_temperature, sentence)
            sentence = RE_FRAC.sub(replace_frac, sentence)
            sentence = RE_PERCENTAGE.sub(replace_percentage, sentence)

            # add by hsl sf快递单号
            sentence = RE_EXPRESS_BILL_NUM.sub(replace_express_bill_num, sentence)
            sentence = RE_MOBILE_PHONE.sub(replace_mobile, sentence)
            sentence = RE_TELEPHONE.sub(replace_phone, sentence)
            sentence = RE_NATIONAL_UNIFORM_NUMBER.sub(replace_phone, sentence)

            sentence = RE_RANGE.sub(replace_range, sentence)
            sentence = RE_INTEGER.sub(replace_negative_num, sentence)
            sentence = RE_DECIMAL_NUM.sub(replace_number, sentence)
            sentence = RE_POSITIVE_QUANTIFIERS.sub(replace_positive_quantifier,
                                                   sentence)
            sentence = RE_DEFAULT_NUM.sub(replace_default_num, sentence)
            sentence = RE_NUMBER.sub(replace_number, sentence)
            sentence = self._post_replace(sentence)
        except Exception as e:
            print ("text normalization warning {}".format(traceback.format_exc()))
        return sentence

    def normalize(self, text: str) -> List[str]:
        text = stringpartQ2B(text)
        sentences = self._split(text)

        #sentences = [self.normalize_sentence(sent) for sent in sentences]
        sentences = [self.normalize_sentence(re.sub(r'[^\u4e00-\u9fa5A-Za-z0-9:：、，；。？！,;?!%％#\.]',' ',sent)) for sent in sentences]

        return sentences
