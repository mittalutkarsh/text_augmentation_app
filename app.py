from flask import Flask, render_template, request
from text_processor import TextAugmenter, TextPreprocessor
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1MB max file size
ALLOWED_EXTENSIONS = {'txt'}

def allowed_file(filename):
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')
        
        if not allowed_file(file.filename):
            return render_template('index.html', 
                                error='Only .txt files are allowed')
        
        try:
            # Read the file content
            text = file.read().decode('utf-8')
            
            # Get number of augmentations
            num_aug = int(request.form.get('num_aug', 4))
            
            # Process the text
            augmenter = TextAugmenter()
            preprocessor = TextPreprocessor()
            
            # Preprocess text
            cleaned_text = preprocessor.clean_text(text)
            tokenized_text = preprocessor.tokenize_text(text).tolist()
            
            # Generate augmentations
            augmented_texts = augmenter.generate_augmentations(text, num_aug)
            
            return render_template(
                'index.html',
                original_text=text,
                cleaned_text=cleaned_text,
                tokenized_text=tokenized_text,
                augmented_texts=augmented_texts
            )
            
        except Exception as e:
            return render_template('index.html', 
                                error=f'Error processing file: {str(e)}')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)