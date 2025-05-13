import random
import abc


def __res2list(sample_tuples: tuple) -> list:
    samples = []
    sample_tuples = list(zip(*sample_tuples))
    for i in range(len(sample_tuples)):
        samples.append(list(sample_tuples[i]))
    return samples


def _compact_res(*args):
    samples = []
    for sample_tuples in args:
        lists = __res2list(sample_tuples)
        for lst in lists:
            if not samples:
                samples = [[] for _ in range(len(lst))]
            for i in range(len(lst)):
                samples[i].append(lst[i])
    return samples


class SampleBuffer(abc.ABC):
    def __init__(self, memory_size, batch_size):
        self.MEM_SIZE = memory_size
        self.BATCH_SIZE = batch_size
        self.item_length = 0

    @abc.abstractmethod
    def add_sample(self, item: list, **kwargs) -> tuple:
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self) -> tuple:
        raise NotImplementedError

    @abc.abstractmethod
    def get_batch(self, size: int):
        raise NotImplementedError

    def update_length(self, item_length):
        self.item_length = item_length

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError


class SampleBufferFactory:
    @staticmethod
    def create_buffer(name: str, *args) -> SampleBuffer:
        if name == 'list':
            return ListSampleBuffer(*args)
        if name == 'delay':
            return DelaySampleBuffer(*args)

    @staticmethod
    def check_available(name: str) -> bool:
        legal = ['list', 'delay']
        return name in legal


class ListSampleBuffer(SampleBuffer):
    def __init__(self, memory_size: int, batch_size: int):
        super().__init__(memory_size, batch_size)
        self.buf = []
        self.param_num = 0

    def add_sample(self, item: list, **kwargs):
        self.buf.append(item)
        if self.param_num <= 0:
            self.update_length(len(item))
        if len(self.buf) > self.MEM_SIZE:
            return self.buf.pop(0)
        else:
            return None

    def sample(self):
        if len(self.buf) <= 1:
            return tuple([None for _ in range(self.param_num)])
        samples = random.sample(self.buf, min(self.BATCH_SIZE, len(self.buf)))
        return tuple(zip(*samples))

    def get_batch(self, size: int):
        return random.sample(self.buf, min(len(self.buf), size))

    def __len__(self):
        return len(self.buf)


class DelaySampleBuffer(SampleBuffer):
    def __init__(self, memory_size: int, batch_size: int, level_num: int = 3, level_bias: list = None,
                 buf_type: str = 'list'):
        super().__init__(memory_size, batch_size)
        if level_bias is None:
            level_bias = [5, 3, 2]
        assert level_num == len(level_bias)
        assert SampleBufferFactory.check_available(buf_type)

        self.level_num = level_num
        self.level_bias = level_bias
        self.item_length = 0

        self.buf = []
        self.full = False
        for i in range(len(self.level_bias)):
            self.buf.append(
                SampleBufferFactory.create_buffer(
                    buf_type,
                    memory_size * self.level_bias[-i-1] / sum(self.level_bias),
                    batch_size)
            )

    def add_sample(self, item: list, **kwargs):
        if self.item_length <= 0:
            self.update_length(len(item))

        if not self.full:
            for i in range(self.level_num):
                tmp = self.buf[i].add_sample(item)
                if tmp is not None:
                    self.full = True
                    break
            return

        for i in range(self.level_num):
            item = self.buf[i].add_sample(item)
            if item is None:
                break

    def sample(self):
        samples = []
        for i in range(self.level_num):
            samples.append(self.buf[i].sample())
        if len(samples[0]) == 0 or samples[0] is None or samples[0][0] is None:
            return tuple([None for _ in range(self.item_length)])
        samples = _compact_res(*samples)
        return tuple(samples)

    def get_batch(self, size: int):
        res = []
        for buf_ in self.buf:
            res.extend(buf_.get_batch(size))
        return random.sample(res, min(size, len(res)))

    def update_length(self, item_length):
        for sb in self.buf:
            sb.update_length(item_length)
        self.item_length = item_length

    def __len__(self):
        length = 0
        for buf in self.buf:
            length += len(buf)
        return length


class PriorSampleBuffer(SampleBuffer):
    def __init__(self, memory_size, batch_size, prior_level=None, buf_type='delay'):
        super().__init__(memory_size, batch_size)
        if prior_level is None:
            prior_level = [0.25, 0.75]
        self.buf = []
        self.item_length = 0
        for i in range(2):
            self.buf.append(SampleBufferFactory.create_buffer(
                buf_type, memory_size, int(batch_size*prior_level[i])))

    def add_sample(self, item: list, *args, done=None, **kwargs):
        if self.item_length <= 0:
            self.update_length(len(item))
        if not kwargs['done']:
            self.buf[0].add_sample(item)
        else:
            self.buf[1].add_sample(item)

    def update_length(self, item_length):
        self.item_length = item_length
        for sb in self.buf:
            sb.update_length(item_length)

    def sample(self):
        done_samples = self.buf[0].sample()
        mid_samples = self.buf[1].sample()
        if done_samples[0] is not None and mid_samples[0] is None:
            return done_samples
        if done_samples[0] is None and mid_samples[0] is not None:
            return mid_samples
        if done_samples[0] is None and mid_samples[0] is None:
            return tuple([None for _ in range(self.item_length)])
        res = _compact_res(done_samples, mid_samples)
        return tuple(res)

    def get_batch(self, size: int) -> list:
        res = []
        for sb in self.buf:
            res.extend(sb.get_batch(size))
        return random.sample(res, min(len(res), size))

    def __len__(self):
        length = 0
        for buf in self.buf:
            length += len(buf)
        return length
