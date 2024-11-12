import nltk
from nltk.corpus import wordnet
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
import re

class TextPreprocessor:
    def __init__(self):
        # Download required NLTK data
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<start>': 2,
            '<end>': 3
        }
    
    def tokenize_text(self, text):
        """Convert text to token IDs"""
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        return torch.tensor(tokens)
    
    def collate_fn(self, batch):
        """Collate function for batching texts"""
        if isinstance(batch[0], tuple):
            texts, labels = zip(*batch)
            texts = [self.tokenize_text(text) for text in texts]
            texts_padded = pad_sequence(texts, batch_first=True, padding_value=self.vocab['<pad>'])
            labels = torch.tensor(labels)
            return texts_padded, labels
        else:
            texts = [self.tokenize_text(text) for text in batch]
            texts_padded = pad_sequence(texts, batch_first=True, padding_value=self.vocab['<pad>'])
            return texts_padded

    def clean_text(self, text):
        """Basic text cleaning"""
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text

class TextAugmenter:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
    
    def synonym_replacement(self, sentence, n=1):
        """
        Replace n words in the sentence with their synonyms
        """
        words = sentence.split()
        new_words = words.copy()
        
        # Keep track of words we've tried to replace
        tried_words = set()
        
        replacements_made = 0
        while replacements_made < n and len(tried_words) < len(words):
            # Choose a word we haven't tried yet
            available_words = [w for w in words if w not in tried_words]
            if not available_words:
                break
                
            word_to_replace = random.choice(available_words)
            tried_words.add(word_to_replace)
            
            synonyms = wordnet.synsets(word_to_replace)
            if synonyms:
                # Get all lemmas from all synsets
                all_lemmas = []
                for syn in synonyms:
                    all_lemmas.extend([lemma.name() for lemma in syn.lemmas()])
                
                # Filter out the original word and multi-word expressions
                valid_synonyms = [lemma for lemma in all_lemmas 
                                if lemma != word_to_replace and '_' not in lemma]
                
                if valid_synonyms:
                    # Choose a random synonym
                    synonym = random.choice(valid_synonyms)
                    # Replace all instances of the word
                    new_words = [synonym if word == word_to_replace else word 
                               for word in new_words]
                    replacements_made += 1
        
        return ' '.join(new_words)
    
    def generate_augmentations(self, text, num_aug=4):
        """Generate multiple augmented versions of the text"""
        augmentations = []
        for _ in range(num_aug):
            # Randomly choose number of words to replace (1 to 3)
            n_words = random.randint(1, 3)
            aug_text = self.synonym_replacement(text, n_words)
            if aug_text != text:  # Only add if different from original
                augmentations.append(aug_text)
        return augmentations

    def preprocess_batch(self, texts, with_labels=False):
        """Preprocess a batch of texts"""
        if with_labels:
            cleaned_texts = [(self.preprocessor.clean_text(text), label) 
                           for text, label in texts]
            padded_texts, labels = self.preprocessor.collate_fn(cleaned_texts)
            return {
                'padded_tensor': padded_texts.tolist(),
                'labels': labels.tolist()
            }
        else:
            cleaned_texts = [self.preprocessor.clean_text(text) for text in texts]
            padded_texts = self.preprocessor.collate_fn(cleaned_texts)
            return {
                'padded_tensor': padded_texts.tolist()
            }

# Add this at the end of the file
__all__ = ['TextPreprocessor', 'TextAugmenter']