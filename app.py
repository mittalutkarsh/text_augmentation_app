from flask import Flask, render_template, request
from text_processor import TextAugmenter, TextPreprocessor, ImagePreprocessor, Object3DProcessor
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size
ALLOWED_EXTENSIONS = {'txt', 'jpg', 'jpeg', 'png', 'off'}

def allowed_file(filename):
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

def is_image_file(filename):
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'})

def is_3d_file(filename):
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in {'off'})

@app.template_filter('b64encode')
def b64encode_filter(data):
    return base64.b64encode(data).decode('utf-8')

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
                                error='Only txt/jpg/jpeg/png/off files allowed')
        
        try:
            if is_image_file(file.filename):
                # Process image
                image_processor = ImagePreprocessor()
                image_data = file.read()
                processed_data = image_processor.preprocess_image(image_data)
                
                # Generate augmentations
                num_aug = int(request.form.get('num_aug', 4))
                augmented_images = image_processor.augment_image(
                    processed_data['original_image'], 
                    num_aug
                )
                
                return render_template(
                    'index.html',
                    is_image=True,
                    original_size=processed_data['original_size'],
                    original_image=processed_data['original_image'],
                    processed_image=processed_data['processed_image'],
                    augmented_images=augmented_images
                )
            elif is_3d_file(file.filename):
                # Process 3D object
                processor = Object3DProcessor()
                file_data = file.read()
                
                # Read the 3D object
                object_data = processor.read_off_file(file_data)
                
                # Generate all views
                num_aug = int(request.form.get('num_aug', 2))
                views_data = processor.generate_all_views(num_aug)
                
                return render_template(
                    'index.html',
                    is_3d=True,
                    original_view=views_data['original_view'],
                    preprocessed_view=views_data['preprocessed_view'],
                    augmented_views=views_data['augmented_views'],
                    vertices_count=len(object_data['vertices']),
                    faces_count=len(object_data['faces'])
                )
            else:
                # Process text
                text = file.read().decode('utf-8')
                num_aug = int(request.form.get('num_aug', 4))
                
                augmenter = TextAugmenter()
                preprocessor = TextPreprocessor()
                
                cleaned_text = preprocessor.clean_text(text)
                tokenized_text = preprocessor.tokenize_text(text).tolist()
                augmented_texts = augmenter.generate_augmentations(text, num_aug)
                
                return render_template(
                    'index.html',
                    is_image=False,
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