from flask import Flask, request, render_template, jsonify
from text_processor import TextAugmenter

app = Flask(__name__)
augmenter = TextAugmenter()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/augment', methods=['POST'])
def augment():
    """Handle single text augmentation"""
    text = request.form.get('text', '')
    num_aug = int(request.form.get('num_augmentations', '4'))
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Generate augmentations
    augmentations = augmenter.generate_augmentations(text, num_aug)
    
    # Preprocess original and augmented texts
    preprocessed = augmenter.preprocess_batch([text])
    
    results = {
        'original': text,
        'augmentations': augmentations,
        'preprocessed': preprocessed
    }
    
    return jsonify(results)

@app.route('/batch_augment', methods=['POST'])
def batch_augment():
    """Handle batch text augmentation"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        num_aug = data.get('num_augmentations', 4)
        
        if not texts:
            return jsonify({'error': 'No texts provided'}), 400
        
        results = []
        for text in texts:
            augmentations = augmenter.generate_augmentations(text, num_aug)
            results.append({
                'original': text,
                'augmentations': augmentations
            })
        
        # Add batch preprocessing results
        preprocessed = augmenter.preprocess_batch(texts)
        results.append({'preprocessed': preprocessed})
            
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)