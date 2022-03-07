import json
import os
import threading
import typing as t
from collections import OrderedDict

import requests
import sh
from PIL import Image
from tqdm import tqdm

from trainer.utils import rank_zero_only


@rank_zero_only
def instantiate_crawler(mode, args):
    # run the images crawler
    print(f"Running a subprocess for images crawler .. {mode}")
    t = threading.Thread(name=f"{mode}_crawler", target=vaw_images_crawler, args=args)
    t.start()


def vaw_images_crawler(
    urls: t.Dict[int, str],
    store_path: str,
    max_trials: int = 10,
    s3_prefix: t.Optional[str] = "s3://scale-ml/home/kareemmetwaly/datasets/vaw_images/",
):
    """
    This function is used to crawl images that is used with VAW dataset
    Args:
        urls (t.Dict[int, str]): dictionary with image_id as key and url as value
        store_path (str): Where the data would be stored
        max_trials (int): the maximum number of trials per image id
        s3_prefix (str): location to check the existence of images first before fetching from url. Set to None to read from url.
    """
    urls = OrderedDict(urls)
    completed_paths = []
    os.makedirs(store_path, exist_ok=True)
    trials_count = {}
    print(f"[Images Crawler] .. Started for {len(urls)}")
    iter_count = 0
    with tqdm(desc="Images Crawler", total=len(urls)) as progress_bar:
        while len(urls) > 0:
            iter_count += 1
            if not iter_count % 1000:
                # print(f"[Images Crawler] .. Stats: remaining={len(urls)}, completed={len(completed_paths)}")
                progress_bar.n = len(completed_paths)
                progress_bar.refresh()
            idx, url = next(iter(urls.items()))
            save_path = os.path.join(store_path, f"{idx}.png")
            if not os.path.exists(save_path):
                try:
                    sh.aws.s3.cp(os.path.join(s3_prefix, f"{idx}.png"), save_path)
                    completed_paths.append(save_path)
                except:
                    try:
                        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
                        image.save(save_path)
                        completed_paths.append(save_path)
                    except:
                        urls.pop(idx)
                        if idx in trials_count.keys():
                            trials_count.update({idx: trials_count[idx] + 1})
                        else:
                            trials_count[idx] = 1
                        if trials_count[idx] < max_trials:
                            urls.update({idx: url})  # leave it to the end
                            continue
            elif (
                os.path.exists(save_path) and save_path not in completed_paths
            ):  # the file might be corrupted if it exists
                try:
                    Image.open(save_path).verify()
                    completed_paths.append(save_path)
                except Exception:
                    print(f"[Images Crawler] .. Removing {save_path}")
                    os.remove(save_path)
                    continue
            urls.pop(idx)
    print("[Images Crawler] .. Finished")


if __name__ == "__main__":
    store_path = "/home/krm/vaw_images/"
    vg_json = os.path.join("/home/krm/datasets/VAW_complete/VAW/", "image_data.json")
    with open(vg_json, "r") as f:
        vg_data = json.load(f)

    vg_data = {
        int(vg_sample["image_id"]): vg_sample["url"]
        for vg_sample in vg_data
        if "url" in vg_sample.keys()
    }
    vaw_images_crawler(urls=vg_data, store_path=store_path)
