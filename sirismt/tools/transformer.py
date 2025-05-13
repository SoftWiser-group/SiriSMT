from sirismt.tools.types import *
from sirismt.tools import objects


__all__ = [
    'make_strategy',
    'strategy2list',
    'parse_smt',
    'parse_strategy',
    'combine_strategy'
]

def combine_strategy(prefix: list, tac: objects.Tactic) -> objects.Tactic:
    if len(prefix) > 0:
        if tac.to_smt2() == 'skip':
            return objects.AndThen(*prefix) if len(prefix) > 1 else prefix[0]
        return objects.AndThen(*prefix, tac)
    return tac


def make_strategy(*tactic: str | objects.Tactic) -> objects.Tactic:
    if len(tactic) == 0:
        return objects.Tactic('skip')
    if len(tactic) > 1:
        return objects.AndThen(*tactic)
    if isinstance(tactic[0], objects.Tactic):
        return tactic[0]
    return objects.Tactic(tactic[0])

def strategy2list(strategy: Strategy) -> list[objects.Tactic]:
    if strategy is None:
        return []
    if isinstance(strategy, objects.AndThen):
        return strategy.v
    return [strategy]


def parse_probe(line: str) -> objects.Probe:
    if line.startswith('Probe'):
        pos1 = line.find('(')
        pos2 = line.find(')')
        return objects.Probe(line[pos1 + 1: pos2])


def parse_strategy(line: str) -> objects.Tactic:
    """
    Parse string which expressing tactic in PyZ3 format (e.g. 'AndThen(Tactic(simplify), Tactic(smt))') to Z3Object.

    Args:
        line: String expressing tactic.

    Returns:
        Z3Object: The Z3Object corresponding to given string.

    Raises:
        AssertionError
    """

    if line.startswith('Tactic'):
        pos1 = line.find('(')
        pos2 = line.find(')', pos1)
        return objects.Tactic(line[pos1 + 1: pos2])

    if line.startswith('With'):
        pos1 = line.find(';')
        tac1 = line[5:pos1]
        line = line[pos1 + 1:]
        dic = {}
        while line.find(';') > 0:
            pos1 = line.find(';')
            pos2 = line.find('=')
            dic[line[:pos2]] = eval(line[pos2 + 1:pos1])
            line = line[pos1 + 1:]
        pos1 = line.find('=')
        dic[line[:pos1]] = eval(line[pos1 + 1:line.find(')')])
        return objects.With(tac1, dic)

    if line.startswith('Cond'):
        balance = 0
        pos1 = line.find(',')
        pos2 = line.find('>')
        pb1 = parse_probe(line[5: pos2 - 1])
        assert isinstance(pb1, objects.Probe), 'unexpected syntax'
        num1 = int(line[pos2 + 1: pos1])
        pb1 = objects.ProbeCond(pb1, num1)

        line = line[pos1:]

        tac_v = []

        for i in range(len(line)):
            if line[i] == ' ' and balance == 0:
                pos1 = i + 1
            if line[i] == '(':
                balance += 1
            if line[i] == ')':
                balance -= 1
                if balance == 0:
                    tac = parse_strategy(line[pos1: i + 1])
                    assert isinstance(tac, objects.Tactic), 'unexpected syntax'
                    tac_v.append(tac)
        return objects.Cond(pb1, tac_v[0], tac_v[1])

    if line.startswith('AndThen'):
        balance = 0
        line = line[8:]
        tac_v = []
        pos1 = 0
        for i in range(len(line)):
            if line[i] == ',' and balance == 0:
                pos1 = i + 1
            if line[i] == '(':
                balance += 1
            if line[i] == ')':
                balance -= 1
                if balance == 0:
                    tac_v.append(parse_strategy(line[pos1: i + 1]))

        return objects.AndThen(*tac_v)
    if line.startswith('['):
        pos1, pos2 = 0, 0
        tac_v = []
        while True:
            pos1 = line.find('\'', pos2 + 1)
            if pos1 == -1:
                break
            pos2 = line.find('\'', pos1 + 1)
            tac_v.append(parse_strategy(line[pos1 + 1:pos2]))
        return objects.AndThen(*tac_v) if len(tac_v) > 1 else tac_v[0]

    raise AssertionError('unexpected syntax')


def parse_smt(source: str) -> objects.Tactic | objects.Probe:
    """
    Parse string which expressing tactic in SMT-LIB format (e.g. '(then simplify smt)') to Z3Object.

    Args:
        source: String expressing tactic.

    Returns:
        Z3Object: The Z3Object corresponding to given string.

    Raises:
        AssertionError
    """
    assert isinstance(source, str), 'unexpected type'
    source = source.strip()
    if source.startswith('('):
        source = __elim_brackets(source)

    if source.startswith('>'):
        word, pos = __find_next_space(source)
        word1, pos = __find_next_space(source, pos)
        word2, pos = __find_next_space(source, pos)
        return objects.ProbeCond(objects.Probe(word1), eval(word2))

    if source.startswith('if'):
        cond, pos = __find_next_brackets(source)
        cond = parse_smt(cond)
        assert isinstance(cond, objects.ProbeCond), 'unexpected syntax'
        if source[pos] == '(' or source[pos + 1] == '(':
            st_a, pos = __find_next_brackets(source, pos)
        else:
            st_a, pos = __find_next_space(source, pos + 1)
        if source[pos] == '(' or source[pos + 1] == '(':
            st_b, _ = __find_next_brackets(source, pos)
        else:
            st_b, _ = __find_next_space(pos + 1)
        st_a = parse_smt(st_a)
        st_b = parse_smt(st_b)
        assert isinstance(st_a, objects.Tactic), 'unexpected syntax'
        assert isinstance(st_b, objects.Tactic), 'unexpected syntax'
        return objects.Cond(cond, st_a, st_b)

    if source.startswith('using-params'):
        word, pos = __find_next_space(source)
        word, pos = __find_next_space(source, pos)
        params = {}
        while pos > 0:
            word1, pos = __find_next_space(source, pos)
            word2, pos = __find_next_space(source, pos)
            if word2 == 'false':
                word2 = 'False'
            if word2 == 'true':
                word2 = 'True'
            params[word1[1:]] = eval(word2)

        return objects.With(word, params)

    if source.startswith('then'):
        word, pos = __find_next_space(source)
        wt = []
        while pos > 0:
            if source[pos] == ' ':
                pos += 1
            if source[pos] == '(':
                word, pos = __find_next_brackets(source, pos)
            else:
                word, pos = __find_next_space(source, pos)
            tac = parse_smt(word)
            if isinstance(tac, objects.AndThen):
                wt += tac.v
            else:
                wt.append(tac)
        return objects.AndThen(*wt) if len(wt) > 1 else wt[0]

    if source.startswith('or-else'):
        word, pos = __find_next_space(source)
        word, pos = __find_next_brackets(source, pos)
        tac1 = parse_smt(word)
        word, pos = __find_next_brackets(source, pos)
        tac2 = parse_smt(word)

        assert isinstance(tac1, objects.TryFor)
        return objects.OrElse(tac1, tac2)

    if source.startswith('try-for'):
        word, pos = __find_next_space(source)
        if source[pos] == '(':
            word, pos = __find_next_brackets(source, pos)
        else:
            word, pos = __find_next_space(source, pos)
        tac = parse_smt(word)
        t = eval(source[pos:])
        return objects.TryFor(t, tac)

    return objects.Tactic(source)


def __elim_brackets(source: str) -> str:
    if source[0] != '(':
        return ''
    balance = 0
    for i in range(len(source)):
        if source[i] == '(':
            balance += 1
        elif source[i] == ')':
            balance -= 1
            if balance == 0:
                return source[1:i]
    return ''


def __find_next_brackets(source: str, pos: int = 0) -> tuple[str, int]:
    st = source.find('(', pos)
    if st < 0:
        return '', -1
    balance = 0
    for i in range(st, len(source)):
        if source[i] == '(':
            balance += 1
        elif source[i] == ')':
            balance -= 1
            if balance == 0:
                return source[st:i + 1], i + 1 if i + 1 < len(source) else -1
    return '', -1


def __find_next_space(source: str, pos: int = 0) -> tuple:
    st = source.find(' ', pos)
    if st < 0:
        return source[pos:], -1
    return source[pos:st], st + 1
