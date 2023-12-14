#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

# This script and the data is from iCAN: Instance-Centric Attention Network for Human-Object Interaction Detection
# The homepage for iCAN is http://chengao.vision/iCAN/ 
# we use the detection result from iCAN 

# Download files from Google Drive with terminal
# Credit: https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive
# Usage: python google_drive.py FILE_ID DESTINATION_FILENAME
# How to get FILE_ID? Click "get sharable link", then you can find it in the end.

import requests
from tqdm import tqdm


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        if 'confirm=t&amp;' in response.text:
            key = response.text.split('confirm=t&amp;uuid=')[1].split('"')[0]
            return key

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768 * 256

        with open(destination, "wb") as f:
            for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : 't', 'uuid': token }
        response = session.get(URL, params = params, stream = True)
    save_response_content(response, destination)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python google_drive.py drive_file_id destination_file_path")
    else:
        # TAKE ID FROM SHAREABLE LINK
        file_id = sys.argv[1]
        # DESTINATION FILE ON YOUR DISK
        destination = sys.argv[2]
        download_file_from_google_drive(file_id, destination)
