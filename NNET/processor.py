# -*- coding: utf-8 -*
from flyai.processor.base import Base


# import bert.tokenization as tokenization
# from bert.run_classifier import convert_single_example_simple


class Processor(Base):
    def __init__(self):
        self.token = None

    def input_x(self, TARGET, TEXT):
        """
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        """
        """
        if self.token is None:
            bert_vocab_file = os.path.join(DATA_PATH, "model", "multi_cased_L-12_H-768_A-12", 'vocab.txt')
            self.token = tokenization.CharTokenizer(vocab_file=bert_vocab_file)

        word_ids, word_mask, word_segment_ids = \
            convert_single_example_simple(max_seq_length=256, tokenizer=self.token, text_a=TARGET, text_b=TEXT)

        return word_ids, word_mask, word_segment_ids
        """


def input_y(self, STANCE):
    """
    参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
    """
    if STANCE == 'NONE':
        return 0
    elif STANCE == 'FAVOR':
        return 1
    elif STANCE == 'AGAINST':
        return 2


def output_y(self, data):
    """
    验证时使用，把模型输出的y转为对应的结果
    """
    return data[0]
