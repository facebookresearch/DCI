# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from flask import Flask, send_from_directory, jsonify, redirect
from PIL import Image
import json
import base64
import io
import os
import sys
from densely_captioned_images.dataset.config import DATASET_PHOTO_PATH, DATASET_ANNOTATIONS_PATH

ENTRIES = os.listdir(DATASET_ANNOTATIONS_PATH)
ENTRIES_MAP = {str(i): e for i, e in enumerate(ENTRIES)}
ENTRIES_REVERSE_MAP = {str(e): i for i, e in enumerate(ENTRIES)}

app = Flask(__name__, static_folder='view/src/static')


def extract_data(entry_key):
    print(entry_key)
    if entry_key in ENTRIES_MAP:
        entry_key = ENTRIES_MAP[entry_key]

    assert entry_key in ENTRIES_REVERSE_MAP
    total_entries = len(ENTRIES_REVERSE_MAP)
    next = (ENTRIES_REVERSE_MAP[entry_key] + 1) % total_entries
    next = ENTRIES_MAP[str(next)]
    prev = (ENTRIES_REVERSE_MAP[entry_key] - 1) % total_entries
    prev = ENTRIES_MAP[str(prev)]

    with open(os.path.join(DATASET_ANNOTATIONS_PATH, entry_key)) as entry_file:
        base_data = json.load(entry_file)
    
    img = Image.open(os.path.join(DATASET_PHOTO_PATH, base_data['image']))
    width, height = img.size
    base_data['width'] = width
    base_data['height'] = height

    # Convert the image to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    base_data['image_data'] = "data:image/png;base64," + img_str

    # Return a JSON object with the image's base64 encoding, height and width
    return jsonify({
        'item': base_data, 
        'next': next, 
        'prev': prev,
        'curr': entry_key,
    })


@app.route('/getdata/<path:path>')
def get_data_path(path):
    try:
        return extract_data(path)
    except Exception as e:
        raise e

@app.route('/getdata/', defaults={'path': None})
def get_data(path):
    # Return your pre-prepared JSON object
    return extract_data(0)

@app.route('/bundle.js')
def serve_js():
    return send_from_directory('view/build/', 'bundle.js')

@app.route('/')
def serve_html():
    return redirect("/0")

# If path doesn't match any of the above routes, serve HTML
@app.route('/<path:path>')
def catch_all(path):
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    if len(sys.argv) > 2:
        print(f'Usage: python {sys.argv[0]} <PortNumber>')
        sys.exit()
    
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        port = 3000

    app.run(debug=True, port=port)
