from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from nltk.translate.meteor_score import meteor_score

# Source, Target, and Predicted sentences
target_sentences = [
    "Ti prego, vieni. E soprattutto, parla con lei.",
    "— Sì, sì — disse Levin — è proprio vero.",
    "In quell'istante Cinque che guardava attorno pieno d'ansia, gridò: — La Regina! la Regina! — e i tre giardinieri si gettarono immediatamente a faccia a terra.",
    "Tornato, trovò Kitty sulla stessa poltrona.",
    "Ne' nostri tempi noi non abbiamo veduto fare gran cose se non a quelli che sono stati tenuti miseri; li altri essere spenti."
]

pred_sentences = [
    "E non si alzò , e non si mise .",
    "— E non è un ’ è un ’ è un ’ è .",
    "E non è un ’ è un ’ è un ’ è un ’ altra .",
    "Il suo Ivanovic si era un ' era un ' era un ' era .",
    "Il suo Ivanovic si era un ' era un ' era un ' era un ' era ."
]

smooth = SmoothingFunction().method1
# Compute BLEU and METEOR scores
results = []
for target, pred in zip(target_sentences, pred_sentences):
    bleu = sentence_bleu([target.split()], pred.split(), smoothing_function=smooth)
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True).score(target, pred)

    results.append((bleu, rouge))
    print(rouge['rougeL'].fmeasure)

# Print results
for i, (bleu, rouge) in enumerate(results):
    print(f"Sentence {i+1}: BLEU = {bleu:.4f}, Rouge = {rouge['rougeL'].fmeasure:.4f}")
