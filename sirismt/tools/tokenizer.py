BV_THEORY = [
    'bvadd', 'bvsub', 'bvneg', 'bvmul', 'bvurem', 'bvsrem', 'bvsmod',
    'bvshl', 'bvlshr', 'bvashr', 'bvor', 'bvand', 'bvnand', 'bvnor',
    'bvxnor', 'bvule', 'bvult', 'bvugt', 'bvuge', 'bvsle', 'bvslt',
    'bvsge', 'bvsgt', 'bvudiv', 'extract', 'bvudiv_i', 'bvnot',
]

ST_TOKENS = [
    '=', '<', '>', '==', '>=', '<=', '=>', '+', '-', '*', '/',
    'true', 'false', 'not', 'and', 'or', 'xor',
    'zero_extend', 'sign_extend', 'concat', 'let', '_', 'ite',
    'exists', 'forall', 'assert', 'declare-fun',
    'int', 'bool', 'bitVec',
]

ALL_TOKENS = ["UNK"] + ST_TOKENS + BV_THEORY


import abc

class Tokenizer(abc.ABC):
    def __init__(self):
        self.cache = {}

    def tokenize(self, instance: str) -> list:
        if instance in self.cache:
            return self.cache.get(instance)
        self.cache[instance] = self.re_tokenize(instance)
        return self.cache.get(instance)

    @abc.abstractmethod
    def re_tokenize(self, instance: str) -> list:
        raise NotImplementedError

    @abc.abstractmethod
    def token_length(self) -> int:
        raise NotImplementedError


class BowTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()
        self.token_idx = {}

        for token_i, token in enumerate(ALL_TOKENS):
            self.token_idx[token] = token_i

    def bow(self, txt):
        if type(txt) is not str:
            txt = str(txt)

        except_tokens = ['[', ']', '(', ')', '\n', ',']
        for tok in except_tokens:
            txt.replace(tok, ' ')

        tokens = txt.split(' ')

        ret = [0 for _ in ALL_TOKENS]
        for token in tokens:
            token = token.lower()
            if token not in self.token_idx:
                ret[0] += 1
            else:
                ret[self.token_idx[token]] += 1
        return ret

    def token_length(self) -> int:
        return len(ALL_TOKENS)

    def re_tokenize(self, instance: str) -> list:
        with open(instance, 'r') as f:
            txt = ' '.join(f.readlines())
        return self.bow(txt)
