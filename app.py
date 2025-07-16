# backend/app.py
from flask import Flask, request, send_file, jsonify
import os, subprocess, uuid, shutil

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def health():
    return 'ECG Backend Running ✅'

@app.route('/upload', methods=['POST'])
def upload_files():
    uploaded = request.files
    record_id = str(uuid.uuid4())[:8]
    
    # Save uploaded files
    for ext in ['hea', 'atr', 'dat']:
        file = uploaded.get(f'file_{ext}')
        if file:
            file.save(os.path.join(UPLOAD_FOLDER, f'{record_id}.{ext}'))

    # Run processing script
    try:
        subprocess.run(['python3', 'process_ecg.py', record_id], cwd='backend', check=True)
    except subprocess.CalledProcessError as e:
        return jsonify({'error': 'Processing failed', 'details': str(e)}), 500

    # Send output files
    try:
        plot_path = os.path.join(OUTPUT_FOLDER, f'ecg_plot{record_id}.json')
        phases_path = os.path.join(OUTPUT_FOLDER, f'ecg_phases{record_id}.json')
        
        # Read both JSONs to return them directly
        with open(plot_path) as f1, open(phases_path) as f2:
            result = {
                "plot": f1.read(),
                "phases": f2.read()
            }

        # Cleanup
        for ext in ['hea', 'atr', 'dat']:
            try:
                os.remove(os.path.join(UPLOAD_FOLDER, f'{record_id}.{ext}'))
            except FileNotFoundError:
                pass

        os.remove(plot_path)
        os.remove(phases_path)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': 'File handling failed', 'details': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Railway provides PORT env var
    app.run(debug=True, host='0.0.0.0', port=port)
