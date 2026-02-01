"""
Test window pattern application for looped transformer.

Example run:
    uv run pytest tests/test_window_pattern.py -v
"""

from nanochat.gpt import GPT, GPTConfig


def test_window_pattern_2_4_2_setup():
    """
    Test that window pattern 'LLSSSLLL' is correctly applied to 2:4:2 looped transformer.

    Config: n_prelude=2, n_recur_block=4, n_coda=2 (8 layers total)
    Pattern: 'LLSSSLLL' (8 chars, maps 1:1 to layers)

    Expected mapping:
        Layer 0 (prelude):   L -> (seq_len, 0)
        Layer 1 (prelude):   L -> (seq_len, 0)
        Layer 2 (recur):     S -> (seq_len//2, 0)
        Layer 3 (recur):     S -> (seq_len//2, 0)
        Layer 4 (recur):     S -> (seq_len//2, 0)
        Layer 5 (recur):     L -> (seq_len, 0)
        Layer 6 (coda):      L -> (seq_len, 0)
        Layer 7 (coda):      L -> (seq_len, 0) [final layer always L]
    """
    config = GPTConfig(
        sequence_len=2048,
        window_pattern="LLSSSLLL",
        n_prelude=2,
        n_recur_block=4,
        n_coda=2,
    )

    # Instantiate model on meta device to avoid allocating real memory
    import torch

    with torch.device("meta"):
        model = GPT(config)

    window_sizes = model.window_sizes
    seq_len = config.sequence_len
    long_window = (seq_len, 0)
    short_window = (seq_len // 2, 0)

    # Total layers should match
    expected_num_layers = config.n_prelude + config.n_recur_block + config.n_coda
    assert len(window_sizes) == expected_num_layers, f"Expected {expected_num_layers} window sizes, got {len(window_sizes)}"

    # Verify each layer's window size matches the pattern
    expected_windows = [
        long_window,  # Layer 0: L (prelude)
        long_window,  # Layer 1: L (prelude)
        short_window,  # Layer 2: S (recur)
        short_window,  # Layer 3: S (recur)
        short_window,  # Layer 4: S (recur)
        long_window,  # Layer 5: L (recur)
        long_window,  # Layer 6: L (coda)
        long_window,  # Layer 7: L (coda, final always L)
    ]

    for i, (actual, expected) in enumerate(zip(window_sizes, expected_windows)):
        assert actual == expected, f"Layer {i}: expected window {expected}, got {actual}"


def test_window_pattern_tiling():
    """
    Test that shorter patterns are correctly tiled across layers.

    Pattern 'SL' with 8 layers should produce: S, L, S, L, S, L, S, L
    But final layer always gets L.
    """
    config = GPTConfig(
        sequence_len=1024,
        window_pattern="SL",
        n_prelude=2,
        n_recur_block=4,
        n_coda=2,
    )

    import torch

    with torch.device("meta"):
        model = GPT(config)

    window_sizes = model.window_sizes
    seq_len = config.sequence_len
    long_window = (seq_len, 0)
    short_window = (seq_len // 2, 0)

    # Pattern 'SL' tiled over 8 layers: S, L, S, L, S, L, S, L
    # Final layer forced to L (it's already L in this case)
    expected_windows = [
        short_window,  # Layer 0: S
        long_window,  # Layer 1: L
        short_window,  # Layer 2: S
        long_window,  # Layer 3: L
        short_window,  # Layer 4: S
        long_window,  # Layer 5: L
        short_window,  # Layer 6: S
        long_window,  # Layer 7: L (final, forced L)
    ]

    for i, (actual, expected) in enumerate(zip(window_sizes, expected_windows)):
        assert actual == expected, f"Layer {i}: expected window {expected}, got {actual}"
