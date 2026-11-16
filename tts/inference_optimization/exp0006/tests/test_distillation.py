"""蒸留で入力を共有し、教師へ勾配を流さないことを検査。"""

import copy
import unittest

import torch

from ..common.distillation import loss_for
from ..common.paths import benchmark_module


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 8)
        self.head = torch.nn.Linear(8, 3)
        self.n_codes = 3
        self.text_drop = 0.1
        self.device = torch.device("cpu")
        self.inputs = None

    def encode(self, texts):
        return torch.zeros(len(texts), 1, 8), None

    def align(self, encoded, durations):
        return None

    def hidden(self, texts, masked, time, drop, condition):
        self.inputs = ([x.clone() for x in masked], time.clone(), drop.clone())
        padded = torch.nn.utils.rnn.pad_sequence(masked, batch_first=True)
        return self.embedding(padded), None


class DistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        benchmark_module().import_models()

    def test_same_inputs_zero_kl_and_no_teacher_gradient(self):
        torch.manual_seed(10)
        student = TinyModel()
        teacher = copy.deepcopy(student).requires_grad_(False)
        data = {
            "texts": [torch.tensor([1, 2]), torch.tensor([2])],
            "codes": [torch.tensor([0, 1, 2]), torch.tensor([1])],
            "durations": [torch.tensor([1, 2]), torch.tensor([1])],
        }
        loss, parts = loss_for(student, teacher, data, [0, 1], 1.0)
        self.assertAlmostEqual(parts["kl"], 0.0, places=6)
        self.assertTrue(torch.isfinite(loss))
        for a, b in zip(student.inputs[0], teacher.inputs[0]):
            self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(student.inputs[1], teacher.inputs[1]))
        self.assertTrue(torch.equal(student.inputs[2], teacher.inputs[2]))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(any(p.grad is not None for p in student.parameters()))
        self.assertGreaterEqual(parts["masked_positions"], 2)


if __name__ == "__main__":
    unittest.main()
