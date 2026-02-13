"""
Tests for calculator tool-use (use_calculator) and tool-use integration during generation.

python -m pytest tests/test_calculator.py -v
"""

from dataclasses import dataclass

import pytest
import torch

from nanochat.engine import Engine, use_calculator


# -----------------------------------------------------------------------------
# Unit tests for use_calculator
# -----------------------------------------------------------------------------


class TestUseCalculator:
    def test_basic_addition(self):
        assert use_calculator("2+3") == 5

    def test_basic_multiplication(self):
        assert use_calculator("10*20") == 200

    def test_basic_subtraction(self):
        assert use_calculator("100-37") == 63

    def test_division(self):
        assert use_calculator("10/4") == 2.5

    def test_parentheses(self):
        assert use_calculator("(2+3)*4") == 20

    def test_nested_parentheses(self):
        assert use_calculator("((1+2)*(3+4))") == 21

    def test_decimal(self):
        assert use_calculator("3.14*2") == pytest.approx(6.28)

    def test_spaces(self):
        assert use_calculator("2 + 3") == 5

    def test_commas_stripped(self):
        assert use_calculator("1,000 + 2,000") == 3000

    def test_power_blocked(self):
        assert use_calculator("2**10") is None

    def test_division_by_zero(self):
        assert use_calculator("1/0") is None

    def test_string_count(self):
        assert use_calculator("'hello'.count('l')") == 2

    def test_string_count_double_quotes(self):
        assert use_calculator('"mississippi".count("ss")') == 2

    def test_string_count_zero(self):
        assert use_calculator("'hello'.count('z')") == 0

    def test_non_count_string_op_blocked(self):
        assert use_calculator("'hello'.upper()") is None

    def test_dangerous_import_blocked(self):
        assert use_calculator("__import__('os')") is None

    def test_dangerous_exec_blocked(self):
        assert use_calculator("exec('print(1)')") is None

    def test_dangerous_getattr_blocked(self):
        assert use_calculator("getattr('', '__class__')") is None

    def test_dangerous_dunder_blocked(self):
        assert use_calculator("''.__class__") is None

    def test_invalid_expression(self):
        assert use_calculator("hello world") is None

    def test_empty_expression(self):
        assert use_calculator("") is None


# -----------------------------------------------------------------------------
# Tool-use integration tests (calculator during generation)
# -----------------------------------------------------------------------------


@dataclass
class MockConfig:
    n_kv_head: int = 4
    n_head: int = 4
    n_embd: int = 64
    n_layer: int = 2
    sequence_len: int = 128
    n_prelude: int = 1
    n_recur_block: int = 1
    n_coda: int = 1
    train_recur_mean: float = 1.0


class ByteTokenizer:
    """Tokens 0-255 are raw bytes, 256+ are special tokens."""

    def __init__(self):
        self._special_tokens = {
            "<|python_start|>": 256,
            "<|python_end|>": 257,
            "<|output_start|>": 258,
            "<|output_end|>": 259,
            "<|assistant_end|>": 260,
            "<|bos|>": 261,
        }
        self._bos = 261

    def encode_special(self, s):
        return self._special_tokens[s]

    def get_bos_token_id(self):
        return self._bos

    def encode(self, s, prepend=None):
        tokens = list(s.encode("utf-8"))
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

    def decode(self, tokens):
        byte_tokens = [t for t in tokens if t < 256]
        return bytes(byte_tokens).decode("utf-8", errors="replace")


class SequenceMockModel:
    """
    Mock model that returns logits predicting a fixed sequence of tokens.
    At each forward call, predicts the next entry in the sequence (at the last position).
    During forced-token decode steps the model's logits are ignored, so those
    entries can be arbitrary (0 by convention).
    """

    def __init__(self, token_sequence: list[int], vocab_size: int = 262):
        self.vocab_size = vocab_size
        self.config = MockConfig()
        self._device = torch.device("cpu")
        self._sequence = token_sequence
        self._step = 0

    def get_device(self):
        return self._device

    def forward(self, ids, kv_cache=None, num_recur=None, warm_start_state=None, **kwargs):
        B, T = ids.shape
        if kv_cache is not None:
            kv_cache.advance(T)
        logits = torch.full((B, T, self.vocab_size), -100.0)
        next_tok = self._sequence[self._step] if self._step < len(self._sequence) else 260
        logits[:, -1, next_tok] = 100.0
        self._step += 1
        if warm_start_state is None:
            warm_start_state = torch.zeros(B, T, self.config.n_embd)
        return logits, warm_start_state


class TestToolUseIntegration:
    """Test that calculator tool-use works correctly during generation."""

    def test_calculator_injects_result(self):
        """Model generates '2+3', calculator injects output_start + '5' + output_end."""
        # ord('2')=50, ord('+')=43, ord('3')=51, ord('5')=53
        model_sequence = [
            256,          # python_start
            50, 43, 51,  # '2', '+', '3'
            257,          # python_end → calc("2+3")=5 → forced: [258, 53, 259]
            0, 0, 0,      # ignored (forced: output_start, '5', output_end)
            260,           # assistant_end
        ]
        prompt = [261, 10, 20]
        engine = Engine(SequenceMockModel(model_sequence), ByteTokenizer())
        results, _ = engine.generate_batch(prompt, num_samples=1, max_tokens=50, temperature=0.0)

        generated = results[0][len(prompt):]
        assert generated == [256, 50, 43, 51, 257, 258, 53, 259]

    def test_calculator_masks(self):
        """Forced (calculator-injected) tokens get mask=0, sampled tokens get mask=1."""
        model_sequence = [256, 50, 43, 51, 257, 0, 0, 0, 260]
        prompt = [261, 10, 20]
        engine = Engine(SequenceMockModel(model_sequence), ByteTokenizer())
        _, masks = engine.generate_batch(prompt, num_samples=1, max_tokens=50, temperature=0.0)

        prompt_masks = masks[0][:len(prompt)]
        generated_masks = masks[0][len(prompt):]
        assert prompt_masks == [0, 0, 0]
        # python_start(1), '2'(1), '+'(1), '3'(1), python_end(1), output_start(0), '5'(0), output_end(0)
        assert generated_masks == [1, 1, 1, 1, 1, 0, 0, 0]

    def test_calculator_multidigit_result(self):
        """Calculator result '100' (3 bytes) is injected correctly."""
        # "99+1" = 100 → "100" → bytes [49, 48, 48]
        # ord('9')=57, ord('+')=43, ord('1')=49
        model_sequence = [
            256,              # python_start
            57, 57, 43, 49,  # '9', '9', '+', '1'
            257,              # python_end → calc("99+1")=100 → forced: [258, 49, 48, 48, 259]
            0, 0, 0, 0, 0,   # ignored (5 forced tokens)
            260,              # assistant_end
        ]
        prompt = [261]
        engine = Engine(SequenceMockModel(model_sequence), ByteTokenizer())
        results, _ = engine.generate_batch(prompt, num_samples=1, max_tokens=50, temperature=0.0)

        generated = results[0][len(prompt):]
        # python_start, 9, 9, +, 1, python_end, output_start, 1, 0, 0, output_end
        assert generated == [256, 57, 57, 43, 49, 257, 258, 49, 48, 48, 259]

    def test_invalid_expression_no_injection(self):
        """Invalid calculator expression → no output tokens injected."""
        # "abc" is not a valid expression
        # ord('a')=97, ord('b')=98, ord('c')=99
        model_sequence = [
            256,           # python_start
            97, 98, 99,   # 'a', 'b', 'c'
            257,           # python_end → calc("abc")=None → no forced tokens
            260,           # assistant_end
        ]
        prompt = [261]
        engine = Engine(SequenceMockModel(model_sequence), ByteTokenizer())
        results, _ = engine.generate_batch(prompt, num_samples=1, max_tokens=50, temperature=0.0)

        generated = results[0][len(prompt):]
        assert generated == [256, 97, 98, 99, 257]

    def test_empty_python_block_no_injection(self):
        """python_start immediately followed by python_end → no calculator call."""
        model_sequence = [
            256,  # python_start
            257,  # python_end (empty expression → no calc)
            260,  # assistant_end
        ]
        prompt = [261]
        engine = Engine(SequenceMockModel(model_sequence), ByteTokenizer())
        results, _ = engine.generate_batch(prompt, num_samples=1, max_tokens=50, temperature=0.0)

        generated = results[0][len(prompt):]
        assert generated == [256, 257]
