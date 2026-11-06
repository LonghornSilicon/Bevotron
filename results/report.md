# Bevotron: precision and vocabulary studies

## Precision (ranked by quantized perplexity; lower is better)

Baseline 8L/512d/16H bevo, 50k TinyStories BPE, 504M tokens, 16K-batch laptop config.
memory_mb = non-embedding weights + KV cache at the scheme's precision, 512-token context.
Memory with embedding adds the tied token table and positions at the weight precision,
which a chip that runs the output layer has to hold or stream for every token.

| rank | variant | run | perplexity fp / quant | BPB | judge_mean | memory_mb | memory with embedding (MB) | d-perplexity vs baseline |
|---|---|---|---|---|---|---|---|---|
| 1 | baseline | 4c89c1af04 | 4.18 / 4.19 | 0.5000 | - | 29.4 | 55.2 | +0.00 |
| 2 | scheme=w4a8kv4,mode=ptq | 0972284b02 | 4.18 / 5.25 | 0.5786 | - | 14.7 | 27.6 | +1.06 |
| 3 | scheme=w4a4kv4,mode=qat | 72ff842a7c | 5.36 / 5.69 | 0.6065 | - | 14.7 | 27.6 | +1.50 |
| 4 | scheme=w4a4kv4,mode=ptq | 97c1a6468e | 4.18 / 245.42 | 1.9192 | - | 14.7 | 27.6 | +241.22 |

## Vocabulary size (ranked by bits per byte; lower is better)

Each run trains from scratch on the 131K-batch baseline config (B300) with the amount of
text held fixed (see configs/ablations/vocab_size.yaml). BPB and perplexity are under the
int8 scheme. Perplexity is per token and not comparable across rows. memory_mb counts
non-embedding weights and KV cache only, so it is the same for every vocabulary. The
vocabulary only changes the embedding params: the tied token table (vocab x 512) plus
512 x 512 learned positions.

| rank | vocab | run | perplexity fp / quant | BPB | d-BPB vs 50k | judge_mean | non-embedding params (M) | embedding params (M) | total params (M) | memory_mb |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16k | a17e1945dd | 3.97 / 3.97 | 0.4822 | -0.0006 | - | 25.2 | 8.5 | 33.6 | 29.4 |
| 2 | 50k | 27cb45751c | 3.98 / 3.99 | 0.4828 | +0.0000 | - | 25.2 | 25.9 | 51.0 | 29.4 |
| 3 | 25k | 25f071f726 | 3.99 / 3.99 | 0.4831 | +0.0004 | - | 25.2 | 13.1 | 38.2 | 29.4 |
| 4 | 12k | 1b069e964b | 3.98 / 3.99 | 0.4841 | +0.0014 | - | 25.2 | 6.4 | 31.6 | 29.4 |
| 5 | 8k | acefd89fea | 3.96 / 3.97 | 0.4858 | +0.0030 | - | 25.2 | 4.4 | 29.5 | 29.4 |
| 6 | 4k | db30129050 | 3.80 / 3.81 | 0.4918 | +0.0090 | - | 25.2 | 2.3 | 27.5 | 29.4 |

### Memory saved by a smaller vocabulary

Memory with embedding = non-embedding weights + KV cache + embeddings, all at the
scheme's precision. The output layer shares the token table and reads all of it for
every generated token, so this is the size a chip that runs the output layer needs.
int8 is the scheme these runs were scored under (PTQ quantizes the tied table too);
int4 is the same model stored as w4a4kv4. Every part halves together, so the
percentage saved is the same at both precisions.

| vocab | embedding params (M) | int8: memory with embedding (MB) | int8: saved vs 50k | int4: memory with embedding (MB) | int4: saved vs 50k | saved % |
|---|---|---|---|---|---|---|
| 50k | 25.9 | 55.2 | - | 27.6 | - | - |
| 25k | 13.1 | 42.4 | 12.8 MB | 21.2 | 6.4 MB | 23.2% |
| 16k | 8.5 | 37.8 | 17.4 MB | 18.9 | 8.7 MB | 31.5% |
| 12k | 6.4 | 35.8 | 19.5 MB | 17.9 | 9.7 MB | 35.2% |
| 8k | 4.4 | 33.7 | 21.5 MB | 16.9 | 10.8 MB | 38.9% |
| 4k | 2.3 | 31.7 | 23.6 MB | 15.8 | 11.8 MB | 42.6% |

## Samples

Every sample uses the prompt "Once upon a time", seed 0, temperature 0.7, top-k 50 and
150 new tokens (`bevotron sample <id> [--quant] -n 1 --max-new-tokens 150 --seed 0`).
`--quant` applies the run's PTQ scheme; without it the checkpoint runs in full precision.
Vocab samples are full precision. Smaller vocabularies spend more tokens per word, so
their 150 tokens cover slightly less text.

### Precision

#### bevotron sample 4c89c1af04

```text
Once upon a time, there was a little girl named Lily. She loved to draw and color pictures of animals. One day, she drew a picture of a big, gray elephant. She was so proud of her picture that she couldn't stop drawing it.

Lily's mom came in and asked her what she was drawing. Lily proudly showed her the picture and her mom said, "Wow, Lily! That's a great picture. You are very talented!" Lily smiled and felt very happy.

Later that day, Lily's friends came over to play. They all wanted to draw something too. Lily decided to lead them all some ideas on how to draw a picture of a elephant. They all had so much fun and Lily felt very proud of herself
```

#### bevotron sample 4c89c1af04 --quant

```text
Once upon a time, there was a little girl named Lily. She loved to draw and color pictures of animals. One day, she drew a picture of a big, gray elephant. She was so proud of her picture that she couldn't stop smiling.

But then, her little brother came and asked to play with her. He wanted to play with the elephant too. Lily didn't want to share her picture to make her brother sad, so she decided to take it off the shelf and hide it.

Later that day, Lily showed her brother the picture she had drawn. He thought it was really cool and asked if he could draw with it too. But Lily said no because she wanted to keep it for herself.

The next day,
```

#### bevotron sample 97c1a6468e --quant

```text
Once upon a time, there and one, a, the big, two of a deep, of the one, a wide, it.






 in the deep, it, the one in the way.

... luckier the.
, like in the park.
```

#### bevotron sample 72ff842a7c

```text
Once upon a time, there was a little girl named Lily. She loved to play outside and explore the world around her. One day, she saw a big maze made of sticks. She wanted to explore it and see what was inside.

Lily asked her mom if she could go in the maze. Her mom said yes, but warned her to be careful because it was dark outside. Lily walked through the maze and saw many beautiful things. She had so much fun exploring and discovering new things at the maze.
After a while, Lily got tired of exploring the maze. She went back to her mom and said, "Mommy, I wanted to go back to the maze again. It was so much fun! Can we go back tomorrow?" Her mom smiled
```

#### bevotron sample 72ff842a7c --quant

```text
Once upon a time, there was a little girl named Lily. She was very happy because she loved to sing and dance. One day, she saw a big maze in the park. She wanted to go inside and see what was there.

When she got to the maze, she found a lot of different ways to show her friends. They were all very excited to see what she was doing. Lily started to sing and dance with her friends. They all thought she was very clever and funny at the maze.

As they played in the maze, they saw other kids playing. They were all singing and dancing like they were in a fun way to dance. But then, one of the kids did something disgusting. He stepped into the maze and made it
```

#### bevotron sample 0972284b02 --quant

```text
Once upon a time, there was a little girl. She was only three years old and she loved to pretend. She would pretend she was a princess, a fairy, or a tent.

One day, she decided to pretend she was a fairy. She danced around her house and made up a story. She pretended she was a fairy, a fairy, and she made up lots of silly moves.

But then, her mom saw what she was pretending and she became angry. She told the little girl she was not allowed to act like that. She told the little girl to stop playing and to take care of her toys.

The little girl was sad and she knew she was not allowed to do it anymore. She knew her mom was right
```

### Vocabulary size

#### bevotron sample 27cb45751c (vocab 50k)

```text
Once upon a time, there was a little girl named Lily. She loved to draw and color pictures. One day, she drew a pretty picture of a flower. She was so happy with it that she started to spin around and around.

Suddenly, her little brother came in and saw the picture. He wanted to have it too, but Lily didn't want to share. They started to argue, and Lily got very angry. She decided to take the picture away from her brother and keep it for herself.

Her brother got very upset and started to cry. Lily felt bad and realized that it wasn't nice to not share her things with her brother. She apologized and shared the picture with him. They both smiled and continued to draw together, happy
```

#### bevotron sample 25f071f726 (vocab 25k)

```text
Once upon a time, there was a little girl. She was very sleepy and wanted to take a nap.

So she went to her bedroom and lay down on her bed. She was so sleepy. She closed her eyes and fell asleep.

In a few minutes later, her mommy came into her bedroom. She said, "Let's play!"

The little girl was so excited, she ran to her room and jumped into her bed. She was so happy to be sleeping at the end of the night.

She felt so safe and content in her bed. She was so happy that she had taken a nap while she was sleeping!
```

#### bevotron sample a17e1945dd (vocab 16k)

```text
Once upon a time, there was a little girl named Lily. She loved to draw and color pictures. One day, she drew a picture of a butterfly and showed it to her mommy. "Look, mommy! I drew a butterfly!" Lily said.

Her mommy smiled and said, "That's great, Lily! You're so gifted at drawing!"

Lily felt proud of herself and happy that she could make something beautiful. She decided to draw another picture of a butterfly and show it to her mommy. "Look, mommy! I drew a butterfly!" she said.

Her mommy hugged her and said, "Wow, Lily! You're such a talented artist. I'm so proud of you."

Lily smiled and felt happy inside
```

#### bevotron sample 1b069e964b (vocab 12k)

```text
Once upon a time, there was a little girl named Lily. She loved to draw and color with her crayons. One day, Lily's mom asked her to wash her hands before dinner. Lily went to the sink and put her hands in the water. 

After she finished washing her hands, Lily went to play with her toys. She played with her dolls and her blocks. Suddenly, she heard a loud noise coming from the living room. She went to investigate and saw that her mom had left a cup of juice on the table. 

Lily's mom came into the room and saw that Lily had used her crayons to draw a picture. "Lily, did you see the picture of the cup of juice?" her mom asked. Lily smiled and said,
```

#### bevotron sample acefd89fea (vocab 8k)

```text
Once upon a time, there was a little girl. She was very sleepy and she wanted to take a nap. She found a nice spot in the garden and curled up. She closed her eyes and soon fell asleep.

When she woke up, she felt a big yawn in her eye. It was time to go home. She quickly got up and walked to her house. On her way, she saw a big dog. The dog was barking very loudly and the little girl was scared.

But then she remembered she had a nap and it made her yawn again. She opened her eyes and saw that the dog was not scary. He was happy and wagged his tail. The little girl gave the dog some food and he ate it all up.
```

#### bevotron sample db30129050 (vocab 4k)

```text
Once upon a time, there was a little girl named Lily. She loved to draw and color with her crayons. One day, Lily's mom gave her a colorful eraser to draw with. Lily was so happy and started drawing right away.

Suddenly, Lily's little brother came into the room and accidentally knocked over the eraser. Lily got upset and started to cry. But then, her mom came in and gave her a hug. She told Lily that it's okay and that she can draw a new eraser. Lily felt better and they drew a beautiful rainbow together.

From that day on, Lily was not afraid to draw and draw with her new eraser. She even showed it to her friends and family. They
```
