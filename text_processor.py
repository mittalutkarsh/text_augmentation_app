import nltk
from nltk.corpus import wordnet
import random
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
import re
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib
matplotlib.use('Agg')  # Required for saving plots without display

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

class ImagePreprocessor:
    def __init__(self):
        self.target_size = (224, 224)  # Standard size for many models
    
    def preprocess_image(self, image_data):
        """Preprocess image data"""
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_data))
        original_image = image.copy()
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        resized_img = image.resize(self.target_size)
        
        # Convert to array
        img_array = np.array(resized_img)
        
        # Normalize
        normalized_img = img_array / 255.0
        
        # Convert normalized array back to image for display
        normalized_image = Image.fromarray((normalized_img * 255).astype('uint8'))
        
        # Save images to bytes for display
        original_bytes = io.BytesIO()
        original_image.save(original_bytes, format='JPEG')
        original_bytes = original_bytes.getvalue()
        
        processed_bytes = io.BytesIO()
        normalized_image.save(processed_bytes, format='JPEG')
        processed_bytes = processed_bytes.getvalue()
        
        return {
            'original_size': image.size,
            'original_image': original_bytes,
            'processed_image': processed_bytes,
            'normalized_array': normalized_img.tolist()
        }
    
    def augment_image(self, image_data, num_aug=2):
        """Generate augmented versions of the image"""
        image = Image.open(io.BytesIO(image_data))
        augmented_images = []
        
        for _ in range(num_aug):
            # Apply random augmentations
            aug_image = image.copy()
            
            # Random rotation (-30 to 30 degrees)
            angle = random.uniform(-30, 30)
            aug_image = aug_image.rotate(angle, expand=True)
            
            # Random horizontal flip
            if random.random() > 0.5:
                aug_image = aug_image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Convert to bytes for display
            aug_bytes = io.BytesIO()
            aug_image.save(aug_bytes, format='JPEG')
            augmented_images.append(aug_bytes.getvalue())
        
        return augmented_images

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

class Object3DProcessor:
    def __init__(self):
        self.vertices = None
        self.faces = None
        self.original_data = None
    
    def read_off_file(self, file_data):
        """Read OFF file and extract vertices and faces"""
        lines = file_data.decode('utf-8').split('\n')
        
        # Check OFF header
        if lines[0].strip() != 'OFF':
            raise ValueError('Not a valid OFF file')
        
        # Get counts
        vertices_count, faces_count, _ = map(int, lines[1].split())
        
        # Read vertices
        self.vertices = []
        for i in range(vertices_count):
            x, y, z = map(float, lines[i + 2].split())
            self.vertices.append([x, y, z])
        
        # Read faces
        self.faces = []
        current_line = vertices_count + 2
        for i in range(faces_count):
            face = list(map(int, lines[current_line + i].split()))[1:]
            self.faces.append(face)
        
        # Store original data
        self.original_data = {
            'vertices': np.array(self.vertices),
            'faces': np.array(self.faces)
        }
        
        return {
            'vertices': self.vertices,
            'faces': self.faces
        }
    
    def preprocess_3d(self):
        """Preprocess 3D object by normalizing and centering"""
        if self.vertices is None or self.faces is None:
            raise ValueError("No 3D object loaded")
        
        vertices = np.array(self.vertices)
        
        # Center the object
        center = vertices.mean(axis=0)
        vertices = vertices - center
        
        # Normalize to unit cube
        max_range = np.max(vertices.max(axis=0) - vertices.min(axis=0))
        vertices = vertices / max_range
        
        return {
            'vertices': vertices,
            'faces': np.array(self.faces)
        }
    
    def augment_3d(self, num_aug=2):
        """Generate augmented versions of the 3D object"""
        if self.original_data is None:
            raise ValueError("No 3D object loaded")
        
        augmented_objects = []
        vertices = self.original_data['vertices']
        faces = self.original_data['faces']
        
        for _ in range(num_aug):
            # Random rotation angles
            theta_x = np.random.uniform(0, 360)
            theta_y = np.random.uniform(0, 360)
            theta_z = np.random.uniform(0, 360)
            
            # Rotation matrices
            Rx = np.array([[1, 0, 0],
                          [0, np.cos(np.radians(theta_x)), -np.sin(np.radians(theta_x))],
                          [0, np.sin(np.radians(theta_x)), np.cos(np.radians(theta_x))]])
            
            Ry = np.array([[np.cos(np.radians(theta_y)), 0, np.sin(np.radians(theta_y))],
                          [0, 1, 0],
                          [-np.sin(np.radians(theta_y)), 0, np.cos(np.radians(theta_y))]])
            
            Rz = np.array([[np.cos(np.radians(theta_z)), -np.sin(np.radians(theta_z)), 0],
                          [np.sin(np.radians(theta_z)), np.cos(np.radians(theta_z)), 0],
                          [0, 0, 1]])
            
            # Apply rotations
            rotated_vertices = vertices.copy()
            rotated_vertices = np.dot(rotated_vertices, Rx)
            rotated_vertices = np.dot(rotated_vertices, Ry)
            rotated_vertices = np.dot(rotated_vertices, Rz)
            
            augmented_objects.append({
                'vertices': rotated_vertices,
                'faces': faces
            })
        
        return augmented_objects
    
    def render_3d_view(self, vertices, faces, elevation=30, azimuth=45):
        """Render a single 3D view"""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create polygon collection
        verts = [vertices[face] for face in faces]
        collection = Poly3DCollection(verts, alpha=0.5)
        collection.set_facecolor('cyan')
        collection.set_edgecolor('black')
        
        # Add collection to axes
        ax.add_collection3d(collection)
        
        # Set viewing angle
        ax.view_init(elev=elevation, azim=azimuth)
        
        # Set axes labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Auto-scale axes
        max_range = np.array([
            vertices[:, 0].max() - vertices[:, 0].min(),
            vertices[:, 1].max() - vertices[:, 1].min(),
            vertices[:, 2].max() - vertices[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
        mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
        mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        
        view_data = buffer.getvalue()
        buffer.close()
        
        return view_data
    
    def generate_all_views(self, num_aug=2):
        """Generate original, preprocessed, and augmented views"""
        if self.vertices is None or self.faces is None:
            raise ValueError("No 3D object loaded")
        
        # Original view
        original_view = self.render_3d_view(
            np.array(self.vertices), 
            np.array(self.faces)
        )
        
        # Preprocessed view
        preprocessed_data = self.preprocess_3d()
        preprocessed_view = self.render_3d_view(
            preprocessed_data['vertices'],
            preprocessed_data['faces']
        )
        
        # Augmented views
        augmented_data = self.augment_3d(num_aug)
        augmented_views = []
        for aug_obj in augmented_data:
            aug_view = self.render_3d_view(
                aug_obj['vertices'],
                aug_obj['faces']
            )
            augmented_views.append(aug_view)
        
        return {
            'original_view': original_view,
            'preprocessed_view': preprocessed_view,
            'augmented_views': augmented_views
        }

# Add this at the end of the file
__all__ = ['TextPreprocessor', 'TextAugmenter', 'ImagePreprocessor', 'Object3DProcessor']