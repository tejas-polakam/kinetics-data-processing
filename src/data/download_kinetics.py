import asyncio
import aiohttp
import tarfile
import io
import os
import logging
import tempfile
import numpy as np
import torch

import ffmpeg
import ray
import pyarrow as pa
import lance

from PIL import Image
from transformers import pipeline


spill_dir = os.path.expanduser("~/ray_spill_data")
os.makedirs(spill_dir, exist_ok=True)
ray.init(num_cpus=2, _temp_dir=spill_dir)


# @ray.remote
# class DepthEstimator:
#     def __init__(self):
#         self.pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-base-hf")
    
#     def process_frame(self, frame):
#         # Process a single frame
#         output = self.pipe(frame)
#         if isinstance(output, list) and len(output) > 0:
#             output = output[0]
#         predicted = output["predicted_depth"]
#         if hasattr(predicted, "cpu"):
#             predicted = predicted.cpu().numpy()
#         predicted = (predicted - predicted.min()) / (predicted.max() - predicted.min() + 1e-8) * 255
#         predicted = predicted.astype("uint8")
#         depth_img = Image.fromarray(predicted).convert("L")
#         buf = io.BytesIO()
#         depth_img.save(buf, format="PNG")
#         return buf.getvalue()

#      #Batch function
#     def process_frames(self, frames):
#         results = []
#         for idx, frame in enumerate(frames):
#             output = self.pipe(frame)
#             if isinstance(output, list) and len(output) > 0:
#                 output = output[0]
#             predicted = output["predicted_depth"]
#             if hasattr(predicted, "cpu"):
#                 predicted = predicted.cpu().numpy()
#             predicted = (predicted - predicted.min()) / (predicted.max() - predicted.min() + 1e-8) * 255
#             predicted = predicted.astype("uint8")
#             depth_img = Image.fromarray(predicted).convert("L")
#             buf = io.BytesIO()
#             depth_img.save(buf, format="PNG")
#             results.append((idx, buf.getvalue()))
#         return results


# depth_estimator = DepthEstimator.remote()

# @ray.remote
# def extract_depth_maps(filename: str, transcoded_bytes: bytes):
#     try:
#         # Decode all frames from the transcoded video.
#         frames = decode_video_to_frames(transcoded_bytes)
#     except Exception as e:
#         raise Exception(f"Error decoding video for {filename}: {str(e)}") from e

#     try:
#         # Batch process all frames as one batch.
#         # 'frames' is a list of all frames from one video.
#         batched_results = ray.get(depth_estimator.process_frames.remote(frames))
#         # Sort results by frame index 
#         batched_results.sort(key=lambda x: x[0])
#         # Extract only the depth map PNG bytes.
#         depth_pngs = [depth for idx, depth in batched_results]
#     except Exception as e:
#         raise Exception(f"Error processing depth maps for {filename}: {str(e)}") from e

#     return filename, transcoded_bytes, depth_pngs


def decode_video_to_frames(video_data, width=640, height=360):
    temp_filename = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_data)
            temp_filename = temp_file.name
        process = (
            ffmpeg
            .input(temp_filename)
            .output("pipe:1", format="rawvideo", vcodec="rawvideo", pix_fmt="rgb24", map="0:v")
            .global_args("-nostdin")
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        out, err = process.communicate()
        if process.returncode != 0:
            err_msg = err.decode("utf-8") if err else "No error message"
            raise RuntimeError("ffmpeg decoding failed")
        frames = []
        pixel_count = width * height * 3
        idx = 0
        while idx + pixel_count <= len(out):
            chunk = out[idx: idx + pixel_count]
            idx += pixel_count
            frame = Image.frombytes("RGB", (width, height), chunk)
            frames.append(frame)
        return frames
    finally:
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)


@ray.remote
def transcode_to_360p(filename: str, video_data: bytes) -> (str, bytes):
    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_temp_name = None
    try:
        input_temp.write(video_data)
        input_temp.close()
        output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_temp_name = output_temp.name
        output_temp.close()
        
        cmd = (
            ffmpeg
            .input(input_temp.name, analyzeduration="5000000", probesize="5000000")
            .filter("scale", 640, 360)
            .output(output_temp_name, vcodec="libx264", pix_fmt="yuv420p", f="mp4")
            .overwrite_output()
            .global_args("-nostdin", "-loglevel", "quiet", "-preset", "ultrafast")
        )
        out, err = cmd.run(capture_stdout=True, capture_stderr=True)
        if err:
            err_msg = err.decode("utf-8") if isinstance(err, bytes) else str(err)
        with open(output_temp_name, "rb") as f:
            transcoded_bytes = f.read()
        return filename, transcoded_bytes
    except ffmpeg.Error as e:
        err_msg = e.stderr.decode("utf-8") if e.stderr else "No error message"
        raise Exception(f"Transcoding failed for {filename}: {err_msg}") from e
    except Exception as e:
        raise Exception(f"Transcoding failed for {filename}: {err_msg}") from e
    finally:
        if os.path.exists(input_temp.name):
            os.remove(input_temp.name)
        if output_temp_name and os.path.exists(output_temp_name):
            os.remove(output_temp_name)


async def process_one_tar(url, concurrency_limit=8):
    sem = asyncio.Semaphore(concurrency_limit)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                tar_data = await resp.read()
                print("Downloaded URL:", url)
    except Exception as e:
        logger.error("Error downloading %s: %s", url, e)
        raise

    tar_stream = io.BytesIO(tar_data)

    # We'll collect asyncio Tasks for transcoding.
    transcode_tasks = []

    def submit_transcode(member, video_bytes):
        # Submits a transcoding task using ray.get in a separate thread.
        async def transcode_task():
            async with sem:
                result = await asyncio.to_thread(ray.get, transcode_to_360p.remote(member.name, video_bytes))
                return result
        return transcode_task()

    try:
        with tarfile.open(fileobj=tar_stream, mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".mp4"):
                    try:
                        video_bytes = tar.extractfile(member).read()
                        # Submit the transcoding task and add it to task list.
                        task = asyncio.create_task(submit_transcode(member, video_bytes))
                        transcode_tasks.append(task)
                    except Exception as e:
                        logger.error("Error preparing file %s: %s", member.name, e)
                        continue

        results = await asyncio.gather(*transcode_tasks, return_exceptions=False)

        # Yield each result (filename, transcoded video bytes).
        for fn, transcoded in results:
            yield (fn, transcoded)
            print("Yielded one video record")
    except Exception as e:
        raise

async def main(urls):
    out_rows = []
    for url in urls:
        async for filename, video_bytes in process_one_tar(url):
            out_rows.append({
                "filename": filename,
                "video": video_bytes,
                "depth_maps": None
            })
    try:
        table = pa.Table.from_pylist(out_rows)
        out_path = "kinetics_val_360p_depth_png.lance"
        lance.write_dataset(table, out_path, mode="create")
    except Exception as e:
        print("Error writing dataset: %s", e)


if __name__ == "__main__":
    TAR_URLS = [
        "https://s3.amazonaws.com/kinetics/400/val/part_0.tar.gz"
    ]
    try:
        asyncio.run(main(TAR_URLS))
    except Exception as e:
        print(e)
