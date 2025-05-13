import abc
import z3

__all__ = [
    'Probe',
    'ProbeCond',
    'Tactic',
    'AndThen',
    'OrElse',
    'TryFor',
    'With',
    'Cond',
]


class Z3Object(abc.ABC):
    """Abstract wrapper class around Z3 object"""

    def __init__(self):
        pass

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def to_smt2(self) -> str:
        raise NotImplementedError


class Probe(Z3Object):
    """ Wrapper class around Z3 Probe object. """

    def __init__(self, s: str):
        super().__init__()
        assert isinstance(s, str)

        self.s = s
        self.probe = z3.Probe(s)

    def __call__(self, g: z3.Goal):
        return self.probe(g)

    def __str__(self):
        return f'Probe({self.s})'

    def to_smt2(self) -> str:
        return self.s


class ProbeCond(Probe):
    """ Wrapper class around Z3 Probe object. """

    def __init__(self, probe: Probe, cond: int):
        super().__init__(probe.s)

        self.probe = probe.probe > cond
        self.cond = cond

    def __call__(self, g: z3.Goal):
        assert isinstance(self.probe, z3.Probe)
        return self.probe(g) > self.cond

    def __str__(self) -> str:
        return f'{str(self.s)} > {self.cond}'

    def to_smt2(self) -> str:
        return f'> {self.s} {str(int(self.cond + 0.5))}'


class Tactic(Z3Object):
    """ Wrapper class around Z3 Tactic object. """

    def __init__(self, s: str):
        super().__init__()
        assert isinstance(s, str)
        self.s = s
        self.tactic = z3.Tactic(s)

    def __str__(self):
        return f'Tactic({self.s})'

    def to_smt2(self) -> str:
        return self.s


class AndThen(Tactic):
    """ Wrapper class around Z3 AndThen object. """

    def __init__(self, *args: Tactic | str):
        super().__init__('skip')
        self.v = [Tactic(x) if isinstance(x, str) else x for x in args]
        self.tactic = z3.AndThen(*[x.tactic for x in self.v])
        self.s = self.to_smt2()

    def __str__(self):
        return f'AndThen({",".join(map(str, self.v))})'

    def to_smt2(self) -> str:
        return f'(then {" ".join([t.to_smt2() for t in self.v])})'


class With(Tactic):
    """ Wrapper class around Z3 With object. """

    def __init__(self, s: str, params: dict):
        super().__init__(s)
        assert isinstance(s, str)

        self.params = params
        self.tactic = z3.With(s, **params)

    def __str__(self):
        param_str = ';'.join(sorted(['{}={}'.format(x, self.params[x]) for x in self.params]))
        return f'With({self.s};{param_str})'

    def to_smt2(self) -> str:
        param_str = [f':{p_name} {str(p_value).lower()}' for p_name, p_value in self.params]

        return f'(using-params {self.s} {" ".join(param_str)})'


class Cond(Tactic):
    """ Wrapper class around Z3 Cond object. """

    def __init__(self, p: ProbeCond, t1: Tactic, t2: Tactic):
        super().__init__('skip')
        self.p = p
        self.t1 = t1
        self.t2 = t2
        self.tactic = z3.Cond(p.probe, t1.tactic, t2.tactic)

    def __str__(self):
        return f'Cond({str(self.p)},{str(self.t1)},{str(self.t2)})'

    def to_smt2(self) -> str:
        return f'(if ({self.p.to_smt2()}) (then {self.t1.to_smt2()}) (then {self.t2.to_smt2()}))'


class TryFor(Tactic):
    def __init__(self, timeout, t: Tactic):
        super().__init__('skip')
        self.timeout = timeout
        self.tactic = z3.TryFor(t.tactic, timeout)
        self.t = t

    def __str__(self):
        return f'TryFor({str(self.t)},{self.timeout})'

    def to_smt2(self) -> str:
        return f'(try-for {self.t.to_smt2()} {self.timeout})'


class OrElse(Tactic):
    def __init__(self, t1: TryFor, t2: Tactic):
        super().__init__('skip')
        self.t2 = t2
        self.t1 = t1
        self.tactic = z3.OrElse(t1.tactic, t2.tactic)

    def __str__(self):
        return f'OrElse({str(self.t1)},{str(self.t2)})'

    def to_smt2(self) -> str:
        return f'(or-else {self.t1.to_smt2()} {self.t2.to_smt2()})'
