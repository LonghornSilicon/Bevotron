# Bevotron: precision and vocabulary studies

## Precision (ranked by quantized perplexity; lower is better)

Baseline 8L/512d/16H bevo, 50k TinyStories BPE, 504M tokens, 16K-batch laptop config.
Memory = non-embedding weights + KV cache at the scheme's precision, 512-token context.

| rank | variant | run | perplexity fp / quant | BPB | judge_mean | memory_mb | d-perplexity vs baseline |
|---|---|---|---|---|---|---|---|
| 1 | baseline | 4c89c1af04 | 4.18 / 4.19 | 0.5000 | - | 29.4 | +0.00 |
| 2 | scheme=w4a8kv4,mode=ptq | 0972284b02 | 4.18 / 5.25 | 0.5786 | - | 14.7 | +1.06 |
| 3 | scheme=w4a4kv4,mode=qat | 72ff842a7c | 5.36 / 5.69 | 0.6065 | - | 14.7 | +1.50 |
| 4 | scheme=w4a4kv4,mode=ptq | 97c1a6468e | 4.18 / 245.42 | 1.9192 | - | 14.7 | +241.22 |

## Vocabulary size (ranked by bits per byte; lower is better)

Each run trains from scratch on the 131K-batch baseline config with the amount of text held
fixed (see configs/ablations/vocab_size.yaml). Rows marked 'not run' are waiting on the
data-center runs.

| rank | variant | run | perplexity fp / quant | BPB | judge_mean | memory_mb | d-bpb vs baseline |
|---|---|---|---|---|---|---|---|
| - | vocab_size=12000,tokens=506075168 | 1b069e964b | not run | - | - | 29.4 | - |
| - | vocab_size=25000,tokens=504351627 | 25f071f726 | not run | - | - | 29.4 | - |
| - | baseline | 27cb45751c | not run | - | - | 29.4 | - |
| - | vocab_size=16000,tokens=505069953 | a17e1945dd | not run | - | - | 29.4 | - |
| - | vocab_size=8000,tokens=509287831 | acefd89fea | not run | - | - | 29.4 | - |
| - | vocab_size=4000,tokens=531004867 | db30129050 | not run | - | - | 29.4 | - |

## Samples (`bevotron sample <id> [--quant] -n 1 --max-new-tokens 150 --seed 0`, prompt "Once upon a time")

### bevotron sample 4c89c1af04

```text
Once upon a time, there was a little girl named Lily. She loved to draw and color pictures of animals. One day, she drew a picture of a big, gray elephant. She was so proud of her picture that she couldn't stop drawing it.

Lily's mom came in and asked her what she was drawing. Lily proudly showed her the picture and her mom said, "Wow, Lily! That's a great picture. You are very talented!" Lily smiled and felt very happy.

Later that day, Lily's friends came over to play. They all wanted to draw something too. Lily decided to lead them all some ideas on how to draw a picture of a elephant. They all had so much fun and Lily felt very proud of herself
```

### bevotron sample 4c89c1af04 --quant

```text
Once upon a time, there was a little girl named Lily. She loved to draw and color pictures of animals. One day, she drew a picture of a big, gray elephant. She was so proud of her picture that she couldn't stop smiling.

But then, her little brother came and asked to play with her. He wanted to play with the elephant too. Lily didn't want to share her picture to make her brother sad, so she decided to take it off the shelf and hide it.

Later that day, Lily showed her brother the picture she had drawn. He thought it was really cool and asked if he could draw with it too. But Lily said no because she wanted to keep it for herself.

The next day,
```

### bevotron sample 97c1a6468e --quant

```text
Once upon a time, there and one, a, the big, two of a deep, of the one, a wide, it.






 in the deep, it, the one in the way.

... luckier the.
, like in the park.
```

### bevotron sample 72ff842a7c

```text
Once upon a time, there was a little girl named Lily. She loved to play outside and explore the world around her. One day, she saw a big maze made of sticks. She wanted to explore it and see what was inside.

Lily asked her mom if she could go in the maze. Her mom said yes, but warned her to be careful because it was dark outside. Lily walked through the maze and saw many beautiful things. She had so much fun exploring and discovering new things at the maze.
After a while, Lily got tired of exploring the maze. She went back to her mom and said, "Mommy, I wanted to go back to the maze again. It was so much fun! Can we go back tomorrow?" Her mom smiled
```

### bevotron sample 72ff842a7c --quant

```text
Once upon a time, there was a little girl named Lily. She was very happy because she loved to sing and dance. One day, she saw a big maze in the park. She wanted to go inside and see what was there.

When she got to the maze, she found a lot of different ways to show her friends. They were all very excited to see what she was doing. Lily started to sing and dance with her friends. They all thought she was very clever and funny at the maze.

As they played in the maze, they saw other kids playing. They were all singing and dancing like they were in a fun way to dance. But then, one of the kids did something disgusting. He stepped into the maze and made it
```

### bevotron sample 0972284b02 --quant

```text
Once upon a time, there was a little girl. She was only three years old and she loved to pretend. She would pretend she was a princess, a fairy, or a tent.

One day, she decided to pretend she was a fairy. She danced around her house and made up a story. She pretended she was a fairy, a fairy, and she made up lots of silly moves.

But then, her mom saw what she was pretending and she became angry. She told the little girl she was not allowed to act like that. She told the little girl to stop playing and to take care of her toys.

The little girl was sad and she knew she was not allowed to do it anymore. She knew her mom was right
```
