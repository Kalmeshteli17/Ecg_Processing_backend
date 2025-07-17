from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, subprocess, uuid
import json # Import json module

app = Flask(__name__)
CORS(app)

# Define UPLOAD_FOLDER and OUTPUT_FOLDER relative to the current script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return '✅ ECG Backend is running!'

@app.route('/upload', methods=['POST'])
def upload_files():
    record_id = str(uuid.uuid4())[:8]
    expected_extensions = ['hea', 'dat', 'atr']
    saved_files = []

    print(f"[{record_id}] Starting file upload process.")

    # Validate and save uploaded files
    for ext in expected_extensions:
        file = request.files.get(f'file_{ext}')
        if not file:
            print(f"[{record_id}] ❌ Missing file: file_{ext}")
            # Clean up any files saved so far if one is missing
            for path_to_clean in saved_files:
                if os.path.exists(path_to_clean):
                    os.remove(path_to_clean)
            return jsonify({'error': f'Missing file: file_{ext}'}), 400

        path = os.path.join(UPLOAD_FOLDER, f"{record_id}.{ext}")
        try:
            file.save(path)
            saved_files.append(path)
            print(f"[{record_id}] ✅ Saved: {path}")
        except Exception as e:
            print(f"[{record_id}] ❌ Failed to save file {path}: {e}")
            # Clean up if save fails
            for path_to_clean in saved_files:
                if os.path.exists(path_to_clean):
                    os.remove(path_to_clean)
            return jsonify({'error': f'Failed to save {ext} file', 'details': str(e)}), 500

    print(f"[{record_id}] 📡 Running process_ecg.py for: {record_id}")
    try:
        # Pass the record_id to process_ecg.py
        # Ensure process_ecg.py is in the same directory or accessible via PATH
        command = ['python3', os.path.join(BASE_DIR, 'process_ecg.py'), record_id]
        print(f"[{record_id}] Executing command: {' '.join(command)}")
        
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"[{record_id}] process_ecg.py stdout:\n{result.stdout}")
        if result.stderr:
            print(f"[{record_id}] process_ecg.py stderr:\n{result.stderr}")

        plot_path = os.path.join(OUTPUT_FOLDER, f'ecg_plot{record_id}.json')
        phases_path = os.path.join(OUTPUT_FOLDER, f'ecg_phases{record_id}.json')

        # Check if output files were created
        if not os.path.exists(plot_path) or not os.path.exists(phases_path):
            print(f"[{record_id}] ❌ Output files not found after processing. Plot: {os.path.exists(plot_path)}, Phases: {os.path.exists(phases_path)}")
            return jsonify({'error': 'ECG processing failed: Output files not generated'}), 500

        # Read the generated JSON files
        with open(plot_path, 'r') as f1, open(phases_path, 'r') as f2:
            response = {
                'plot': f1.read(),
                'phases': f2.read()
            }
        
        print(f"[{record_id}] ✅ Successfully processed and read JSONs.")

        return jsonify(response)

    except subprocess.CalledProcessError as e:
        print(f"[{record_id}] ❌ Subprocess error: {e}")
        print(f"[{record_id}] stdout: {e.stdout}")
        print(f"[{record_id}] stderr: {e.stderr}")
        return jsonify({'error': 'ECG processing failed', 'details': e.stderr or str(e)}), 500
    except FileNotFoundError:
        print(f"[{record_id}] ❌ process_ecg.py not found or Python interpreter issue.")
        return jsonify({'error': 'Server configuration error: processing script not found'}), 500
    except json.JSONDecodeError as e:
        print(f"[{record_id}] ❌ JSON decoding error from output files: {e}")
        return jsonify({'error': 'Failed to read generated JSON files', 'details': str(e)}), 500
    except Exception as e:
        print(f"[{record_id}] ❌ Unexpected error: {e}")
        return jsonify({'error': 'Unexpected server error', 'details': str(e)}), 500
    finally:
        # Cleanup: Delete all temporary files (uploaded and generated outputs)
        print(f"[{record_id}] Initiating cleanup of temporary files.")
        files_to_clean = list(saved_files) # Copy list to avoid modification during iteration
        
        # Add generated output files to cleanup list, if they exist
        output_plot_path = os.path.join(OUTPUT_FOLDER, f'ecg_plot{record_id}.json')
        output_phases_path = os.path.join(OUTPUT_FOLDER, f'ecg_phases{record_id}.json')
        if os.path.exists(output_plot_path):
            files_to_clean.append(output_plot_path)
        if os.path.exists(output_phases_path):
            files_to_clean.append(output_phases_path)

        for file_path in files_to_clean:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"[{record_id}] Deleted: {file_path}")
            except Exception as e:
                print(f"[{record_id}] ⚠️ Could not delete {file_path}: {e}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)