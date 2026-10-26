"""
Evaluate trained checkpoints. Not written yet; the plan:

$ python eval.py out/baseline out/rope              # compare experiments
$ python eval.py out/baseline --quant=int4_g64      # post-training quantization, via quant.quantize_
$ python eval.py roneneldan/TinyStories-28M         # the published model, as a reference point

Each checkpoint stores its model_args, so any variant rebuilds without extra flags.
For each one, write out/<run>/eval.json and out/<run>/samples.txt with:
1) loss over the entire val split, in order (train.py's estimate_loss uses random batches)
2) loss at each token position, including past the training block_size (rope vs learned)
3) with --quant: change in loss and KL divergence from the unquantized model
4) samples from fixed prompts with a fixed seed, to compare side by side
5) params, weight bits, tokens/sec
"""
