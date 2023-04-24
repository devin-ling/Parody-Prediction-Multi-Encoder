# Parody-Prediction-Multi-Encoder

Hello, this is a reproduction of the 2022 accepted NAACL paper, [Combining Humor and Sarcasm for Improving Political Parody Detection] (https://aclanthology.org/2022.naacl-main.131/). This is done as a final project for a graduate course, Natural Language Processing. It's our first dive into NLP, so we'd thought to take an attempt at this project.

The main part of the paper to reproduce is the three encoders (parody, humor, and sarcasm) and the several ways to combine them into one complete multiencoder. We test the final multiencoder on the person split as mentioned in the source paper. We compare to the strongest single-encoder model, which was BERTweet. We also reproduce the ablation study to some extent.
