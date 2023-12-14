# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.tools.data_browser import DataBrowser
from tqdm import tqdm
import os
import json

from mephisto.data_model.unit import Unit
from typing import Dict, List

"""
Script to allow bonusing workers for high quality work that extended beyond the 
intended task duration.
"""

TARGET_TASKS = [
    'TODO - fill with task names' # TODO
]

CUTOFF_TIME = 0
PAY_AMOUNT_PER_TASK = -1 # TODO set
TARGET_PAY = -1 # TODO set

if os.path.exists('assets/last_cutoff_time.txt'):
    with open('assets/last_cutoff_time.txt') as time_f:
        CUTOFF_TIME = float(time_f.read())

def get_all_words(short_caption, extra_caption, mask_data):
    caption = short_caption.strip()
    caption += " " + extra_caption.strip()
    for entry in mask_data.values():
        if len(entry['label'].strip()) > 0:
            caption += " " + entry['label'].strip()
        if len(entry['caption'].strip()) > 0:
            caption += " " + entry['caption'].strip()
    caption = caption.replace(".", "").replace(",", "").replace('?', '')
    return caption

def extract_stats_data(m_data, duration):
    """Convert the Mephisto-formatted data to a final annotation"""
    inputs = m_data['inputs']
    outputs = m_data['outputs']
    reconstructed_masks = inputs['mask_data']
    for mask_key in reconstructed_masks.keys():
        reconstructed_masks[mask_key].update(outputs['data'][mask_key])

    words = get_all_words(outputs['baseCaption'], outputs['finalCaption'], reconstructed_masks)
    words = " ".join([w for w in words.split() if len(w) > 0])
    unique_words = set(words.lower().split())

    return {
        'mask_count': len([r for r in reconstructed_masks.values() if len(r['label'])]),
        'words': len(words.split()),
        'unique_words': len(unique_words),
        'duration': duration,
    }


def main():
    last_time = 0
    worker_to_units_map: Dict[str, List[Unit]] = {}
    if os.path.exists('assets/details.json'):
        with open('assets/details.json', 'r') as jsonf:
            worker_to_stats_map = json.load(jsonf)
    else:
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
        
        print(f"Found {len(approved_units)} approved units, sorting into workers")
        worker_to_stats_map = {}
        worker_to_units_map = {}
        for u in tqdm(approved_units):
            try:
                agent = u.get_assigned_agent()
                w_id = agent.get_worker().worker_name
                data = agent.state.get_data()
                task_start = agent.state.get_task_start()
                if task_start is not None and task_start <= CUTOFF_TIME:
                    continue
                last_time = max(task_start, last_time)
                duration = 0
                if agent.state.get_task_end() is not None and agent.state.get_task_start() is not None:
                    duration = agent.state.get_task_end() - agent.state.get_task_start()
                stats = extract_stats_data(data, duration)
                if w_id not in worker_to_stats_map:
                    worker_to_stats_map[w_id] = []
                    worker_to_units_map[w_id] = []
                worker_to_stats_map[w_id].append(stats)
                worker_to_units_map[w_id].append(u)
            except OSError:
                print(f"Skipping {u}")
                continue 

    total_deficit = 0
    expected_pay = 0
    expected_pay_per_worker = {w_id: 0.0 for w_id in worker_to_stats_map.keys()}
    print(f"Total workers: {len(worker_to_stats_map)}")
    for w_id, stats in sorted(list(worker_to_stats_map.items()), key=lambda x: len(x[1]), reverse=True):
        total_work = len(stats)
        total_words = sum([s['words'] for s in stats])
        total_unique_words = sum([s['unique_words'] for s in stats])
        total_duration = sum([s['duration'] for s in stats])
        total_masks = sum([s['mask_count'] for s in stats])
        total_pay = total_work * PAY_AMOUNT_PER_TASK
        total_hours = total_duration / 60 / 60

        target_pay = TARGET_PAY * total_hours
        deficit_pay = max(target_pay - total_pay, 0)
        target_cpuw = 0.08
        cpuw_target_pay = target_cpuw * total_unique_words
        cpuw_bonus = max(cpuw_target_pay-total_pay, 0)
        bonus_pay = min(deficit_pay, cpuw_bonus)
        expected_pay_per_worker[w_id] = bonus_pay

        print(
            f"Worker {w_id} - count: {total_work}:\n"
            f"  Time - Tot: {total_hours:.2f} hours. Avg: {total_duration / 60 / total_work:.2f} minutes\n"
            f"  Pay - Tot: {total_pay}. Avg Hourly: {total_pay / (total_hours):.2f}. Cost per word {total_pay / total_words:.6f}. Cost per unique word {total_pay / total_unique_words:.6f}.\n"
            f"  Words - Tot: {total_words}. Avg: {total_words/total_work:.2f}. Per $ {total_words / total_pay:.2f}. Per hour {total_words / total_hours:.2f}.\n"
            f"  Unique Words - Tot: {total_unique_words}. Avg: {total_unique_words/total_work:.2f}. Per $ {total_unique_words / total_pay:.2f}. Per hour {total_unique_words / total_hours:.2f}.\n"
            f"  Unique % - Tot: {total_unique_words / total_words * 100:.2f}%.\n"
            f"  Masks - Tot: {total_masks}. Avg: {total_masks/total_work:.2f}. Per $ {total_masks / total_pay:.2f}. Per hour {total_masks / total_hours:.2f}.\n"
            f"  Deficit: {deficit_pay:.2f}. Expected pay {bonus_pay:.2f}"
            "\n"
        )

        total_deficit += max(target_pay - total_pay, 0)
        expected_pay += bonus_pay

    print(f"Total deficit: {total_deficit}, expected pay: {expected_pay}")

    with open('assets/details.json', 'w+') as jsonf:
        json.dump(worker_to_stats_map, jsonf, indent=4)

    if len(worker_to_units_map) == 0:
        return
    
    do_bonus = input("Do bonusing? y/(n)")
    if not do_bonus.startswith('y'):
        return
    
    for w_id, pay_amount in expected_pay_per_worker.items():
        if pay_amount == 0:
            continue
        pay_per_unit = int(pay_amount / len(worker_to_stats_map[w_id]) * 100) / 100.0
        doit = input(f"Bonus {w_id} ${pay_per_unit} for {len(worker_to_units_map[w_id])} units (total ${pay_per_unit * len(worker_to_units_map[w_id])})")
        if doit.startswith('n'):
            continue
        for u in worker_to_units_map[w_id]:
            try:
                u.get_assigned_agent().get_worker().bonus_worker(
                    pay_per_unit, 
                    "Long Captions Bonus: Awarded for having a high average quality score "
                    "across your submitted tasks, as measured by how much of the total visual "
                    "information in the images are captured in your descriptions.",
                    unit=u,
                )
            except Exception as e:
                print(f"Had error on one bonus: {e}")

    with open('assets/last_cutoff_time.txt', 'w') as cutoff_f:
        cutoff_f.write(str(last_time))

if __name__ == '__main__':
    main()
