#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
import requests
import tarfile
from tqdm import tqdm
import logging
import time
from densely_captioned_images.dataset.config import DATASET_BASE, MODEL_BASE
import hashlib

RESOURCES = {
    'densely_captioned_images': {
        'url': 'https://dl.fbaipublicfiles.com/densely_captioned_images/dci.tar.gz',
        'check': 'd865c244150168d3f25daaad0bf5b70b2123cf3e83ca7b0e207d5b26943c5fc7'
    },
    'dci_pick1': {
        'url': 'https://dl.fbaipublicfiles.com/densely_captioned_images/dci_pick1.tar.gz',
        'check': '225b9e1c88a00a2dd62cd5b0e23d4133efbbffe8d28b6de796ba894db1a2aa6a'
    },
    'dci_pick1_nl0': {
        'url': 'https://dl.fbaipublicfiles.com/densely_captioned_images/dci_pick1_nl0.tar.gz',
        'check': '2943daaba2489492c274284631223e02933cc9436db201c23abe5939ed01d446'
    },
}

def check_checksum(file_download_location, target_checksum):
    sha256_hash = hashlib.sha256()
    with open(file_download_location, 'rb') as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    
    assert sha256_hash.hexdigest() == target_checksum, (
        f"Checksums don't match, unzip {file_download_location} at your own risk!!"
    )
    return


def download_file(target_dir, file_meta, num_retries=5):
    """
    Download file using `requests`.

    If ``redownload`` is set to false, then will not download tar file again if it is
    present (default ``False``).
    """
    url = file_meta['url']
    outfile = file_meta['url'].split('/')[-1]

    download = True
    logging.info(f"Downloading {url} to {outfile}")
    retry = num_retries
    exp_backoff = [2**r for r in reversed(range(retry))]

    pbar = tqdm(unit='B', unit_scale=True, desc='Downloading {}'.format(outfile))

    while download and retry > 0:
        response = None

        with requests.Session() as session:
            try:
                response = session.get(url, stream=True, timeout=5)

                # negative reply could be 'none' or just missing
                CHUNK_SIZE = 32768
                total_size = int(response.headers.get('Content-Length', -1))
                # server returns remaining size if resuming, so adjust total
                pbar.total = total_size
                done = 0

                with open(outfile, 'wb') as f:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                        if total_size > 0:
                            done += len(chunk)
                            if total_size < done:
                                # don't freak out if content-length was too small
                                total_size = done
                                pbar.total = total_size
                            pbar.update(len(chunk))
                    break
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
            ):
                retry -= 1
                pbar.clear()
                if retry > 0:
                    pl = 'y' if retry == 1 else 'ies'
                    logging.debug(
                        f'Connection error, retrying. ({retry} retr{pl} left)'
                    )
                    time.sleep(exp_backoff[retry])
                else:
                    logging.error('Retried too many times, stopped retrying.')
            finally:
                if response:
                    response.close()
    if retry <= 0:
        raise RuntimeError('Connection broken too many times. Stopped retrying.')

    if download and retry > 0:
        pbar.update(done - pbar.n)
        if done < total_size:
            raise RuntimeError(
                f'Received less data than specified in Content-Length header for '
                f'{url}. There may be a download problem.'
            )

    pbar.close()

    check_checksum(outfile, file_meta['check'])
    logging.info(f"Checksum passed! Extracting...")

    os.makedirs(target_dir, exist_ok=True)

    with tarfile.open(outfile, 'r') as tar:
        # Go over each member
        for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
            # Extract member
            tar.extract(path=target_dir, member=member)
    os.unlink(outfile)


def run_downloads():
    for dsname in ['densely_captioned_images']:
        download_file(os.path.dirname(DATASET_BASE), RESOURCES[dsname])
    for modelname in ['dci_pick1', 'dci_pick1_nl0']:
        download_file(MODEL_BASE, RESOURCES[modelname])


if __name__ == '__main__':
    run_downloads()
