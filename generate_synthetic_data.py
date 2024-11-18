import pandas as pd
import random
from nltk.corpus import wordnet
from tqdm import tqdm

# Load raw data
def load_data(file_path):
    return pd.read_csv(file_path)

# Paraphrasing function using WordNet
def paraphrase(sentence):
    words = sentence.split()
    new_sentence = []

    for word in words:
        synsets = wordnet.synsets(word)
        if synsets:
            lemma = random.choice(synsets).lemmas()[0].name()
            new_sentence.append(lemma if lemma != word else word)
        else:
            new_sentence.append(word)

    return ' '.join(new_sentence)

# Generate synthetic data with augmentation
def generate_synthetic_data(data, num_samples=500):
    synthetic_data = []
    
    for _ in tqdm(range(num_samples)):
        text = random.choice(data['text'].tolist())
        paraphrased_text = paraphrase(text)
        synthetic_data.append(paraphrased_text)
    
    return pd.DataFrame(synthetic_data, columns=['text'])

# Save generated data
def save_synthetic_data(data, output_path):
    data.to_csv(output_path, index=False)

# Main function to generate and save synthetic data
if __name__ == "__main__":
    # Load original data
    data = load_data('data/raw_data.csv')
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(data)
    
    # Save the generated data
    save_synthetic_data(synthetic_data, 'data/synthetic_data.csv')
