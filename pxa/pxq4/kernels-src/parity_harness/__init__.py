"""PXQ4-in-vLLM parity harness (plan §9, agent D).

Gates G1-G4 run here with numpy only, no GPU, no vLLM, no container.
Gates G6-G8 run the same oracles against torch.ops.pxq4.* when a GPU is present.
"""
__all__ = ["oracle", "gguf_raw", "fixtures", "cref_bridge"]
