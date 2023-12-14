# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.tools.data_browser import DataBrowser
from tqdm import tqdm
import os
import shutil
import json
from densely_captioned_images.dataset.config import DATASET_PHOTO_PATH, DATASET_ANNOTATIONS_PATH

"""
Script to allow exporting data from the Mephisto data collection to the checkpoint
dataset location.
"""

TARGET_TASKS = [
    'TODO - fill with task names' # TODO
]

ASSET_PATH = os.path.join(os.path.dirname(__file__), 'assets/images')

def extract_final_data(m_data):
    """Convert the Mephisto-formatted data to a final annotation"""
    inputs = m_data['inputs']
    outputs = m_data['outputs']
    reconstructed_masks = inputs['mask_data']
    for mask_key in reconstructed_masks.keys():
        reconstructed_masks[mask_key].update(outputs['data'][mask_key])

    return {
        'short_caption': outputs['baseCaption'],
        'extra_caption': outputs['finalCaption'],
        'image': inputs['mask'].split('-mask.json')[0],
        'height': inputs['image_data']['height'],
        'width': inputs['image_data']['width'],
        'mask_data': reconstructed_masks,
        'mask_keys': list(outputs['data'].keys()),
    }


def main():
    db = LocalMephistoDB()
    browser = DataBrowser(db)
    approved_units = []
    print("Querying for tasks...")
    for target in TARGET_TASKS:
        print(f"q: {target} - ", end="", flush=True)
        units = browser.get_units_for_task_name(target)
        print(f"{len(units)} total - ", end="", flush=True)
        units = [u for u in units if u.get_assigned_agent() is not None and u.get_assigned_agent().get_status()=='approved']
        print(f"{len(units)} approved", flush=True)
        approved_units += units
    
    print(f"Found {len(approved_units)} approved units, preparing for export")
    for u in tqdm(approved_units):
        try:
            data = u.get_assigned_agent().state.get_data()
            formatted = extract_final_data(data)
            source_image_path = os.path.join(ASSET_PATH, formatted['image'])
            target_image_path = os.path.join(DATASET_PHOTO_PATH, formatted['image'])
            output_data_path = os.path.join(DATASET_ANNOTATIONS_PATH, f"{u.db_id}-data.json")
            shutil.copy2(source_image_path, target_image_path)
            with open(output_data_path, 'w+') as output_file:
                json.dump(formatted, output_file)
        except OSError:
            print(f"Skipping {u}")
            continue 


if __name__ == '__main__':
    main()
