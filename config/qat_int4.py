# Quantization-aware training: the transformer blocks train against int4 weights with one
# scale per group of 64 input channels. Embeddings and lm_head stay in full precision.
quant = 'int4_g64'
