# Building a Transformer from Scratch

Welcome to this project! This isn't just another machine translation model; it's a journey into the heart of the Transformer architecture, built from the ground up with PyTorch. If you've ever wanted to understand what goes on inside models like GPT and BERT, you're in the right place.

This project was born out of a desire to demystify the magic behind Transformers. We'll walk through everything from creating word embeddings to stacking encoder and decoder layers, all in the service of teaching a machine to translate languages.

## What's Inside?

- **A Pure PyTorch Transformer**: We've built the entire Transformer architecture using PyTorch, so you can see how every piece fits together.
- **Train on Multiple GPUs**: Got more than one GPU? Great! We've set up distributed training to speed things up.
- **Track Your Experiments**: We're using Weights & Biases (`wandb`) to keep an eye on our model's progress. It's like a lab notebook for your experiments.
- **See How Well It's Doing**: We're using standard metrics like BLEU and ROUGE to see how good our model is at translating.

## Getting Started

Ready to dive in? Here's how to get set up.

1.  **Clone this repository:**
    ```bash
    https://github.com/wothmag07/pytorch-transformers.git
    cd pytorch-transformers
    ```

2.  **Install the necessary packages:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Use It

### Tweaking the Settings

You can change the settings for the model and training in the `config.py` file. Feel free to play around with things like the languages you're translating between, the size of the model, or how long you want to train it.

### Let's Train!

To start training the model, just run this command:

```bash
python train.py
```

The script will take care of everything else, from loading the data to saving the trained model in the `weights` folder.

## Data

This project uses the `opus_books` dataset from `Helsinki-NLP`, which contains a collection of translated books. This dataset is ideal for training machine translation models. The `train.py` script automatically downloads the dataset for the languages specified in `config.py`.

## Tokenizers

To understand language, our model first needs to break it down into smaller pieces called tokens. We're using custom tokenizers for this, and we've already trained some for you.

You can download the pre-trained tokenizers directly from this repository:

- **English**: [tokenizer_en.json](./tokenizers/tokenizer_en.json)
- **Spanish**: [tokenizer_es.json](./tokenizers/tokenizer_es.json)
- **French**: [tokenizer_fr.json](./tokenizers/tokenizer_fr.json)
- **Italian**: [tokenizer_it.json](./tokenizers/tokenizer_it.json)

## Under the Hood: Model

The `model.py` file is where all the magic happens. It's where we've defined all the parts of our Transformer model. It's a bit like building with LEGOs—we have smaller blocks that we put together to create something amazing.

## How Do We Know It's Working?

We use a couple of standard tests to see how well our model is doing:

- **BLEU**: This score tells us how close our model's translations are to a human's.
- **ROUGE**: This one looks at how many of the same words are in our model's translation and a human's.

We log all of this information in `output.txt` and on Weights & Biases, so you can see the progress for yourself.

Happy coding!