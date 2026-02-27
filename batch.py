import os, time, torch
import pandas as pd
from config import *
from model_utils import load_model
from audio_utils import get_audio_transcript
from inference_batch import (
    create_messages, run_inference, run_inference_tensor_mode, save_results_to_csv, save_stats_to_json
)
from evaluation_utils import evaluate_with_stats
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader
from preprocessing import preprocess_dataset

def process_single_audio(vid_file):
    # Helps to get the audio transcript for a single video
    video_id = vid_file.split(".mp4")[0]
    video_path = os.path.join(VIDEO_FOLDER, vid_file)
    
    transcript = get_audio_transcript(video_id, video_path, AUDIO_FOLDER, AUDIO_TRANSCRIPT_FOLDER)
    
    return video_id, transcript

# CODE for simple batch processing.
# class VideoInferenceDataset(Dataset):
#     def __init__(self, video_data_map, video_folder):
#         self.video_ids = list(video_data_map.keys())
#         self.video_data_map = video_data_map
#         self.video_folder = video_folder

#     def __len__(self):
#         return len(self.video_ids)

#     def __getitem__(self, idx):
#         vid_id = self.video_ids[idx]
#         data = self.video_data_map[vid_id]
        
#         video_path = os.path.join(self.video_folder, data["file_name"])
#         transcript = data["transcript"]

#         summary_msg = create_messages(video_path, transcript, mode="summary")
#         category_msg = create_messages(video_path, transcript, mode="category")

#         return vid_id, summary_msg, category_msg

# def custom_collate_fn(batch):
#     vid_ids, sum_msgs, cat_msgs = zip(*batch)
#     return list(vid_ids), list(sum_msgs), list(cat_msgs)

# CODE for simple batch processing.
# def main():
#     video_ids = [v for v in os.listdir(VIDEO_FOLDER) if v.endswith(('.mp4'))]
#     if SAMPLE_VIDEO:
#         video_ids = video_ids[:SAMPLE_SIZE]
#     print(f"Processing {len(video_ids)} videos...")

#     video_data_map = {}

#     with ThreadPoolExecutor(max_workers=4) as executor:
#         results = executor.map(process_single_audio, video_ids)
        
#         for vid_file, (vid_id, transcript) in zip(video_ids, results):
#             video_data_map[vid_id] = {
#                 "transcript": transcript,
#                 "file_name": vid_file
#             }

#     start_time = time.time()

#     processor, model, model_load_time = load_model(MODEL_NAME, DEVICE)

#     dataset = VideoInferenceDataset(video_data_map, VIDEO_FOLDER)
    
#     BATCH_SIZE = 16
#     dataloader = DataLoader(
#         dataset, 
#         batch_size=BATCH_SIZE, 
#         shuffle=False, 
#         num_workers=2, 
#         collate_fn=custom_collate_fn
#     )

#     res_dict, summary_times, category_times = {}, [], []

#     for i, (batch_vid_ids, batch_sum_msgs, batch_cat_msgs) in enumerate(dataloader):
        
#         summaries, t1 = run_inference(processor, model, batch_sum_msgs, mode="summary")
#         summary_times.append(t1)

#         categories, t2 = run_inference(processor, model, batch_cat_msgs, mode="category")
#         category_times.append(t2)

#         for j, vid_id in enumerate(batch_vid_ids):
#             res_dict[vid_id] = {"summary": summaries[j], "category": categories[j]}
        
#         print(f"Processed batch {i+1}/{len(dataloader)}")

#     total_time = time.time() - start_time

#     response_df = save_results_to_csv(res_dict, CSV_PATH)

#     ground_truth = pd.read_csv(GROUND_TRUTH_FILE)
#     # response_df = pd.DataFrame.from_dict(res_dict, orient="index").reset_index()
#     # response_df.rename(columns={"index": "video_id"}, inplace=True)

#     ground_truth_video_ids = set(ground_truth['video_id'].astype(str).tolist())
#     response_video_ids = set(response_df['video_id'].astype(str).tolist())
#     common_video_ids = ground_truth_video_ids.intersection(response_video_ids)
#     ground_truth = ground_truth[ground_truth['video_id'].astype(str).isin(common_video_ids)]
#     response_df = response_df[response_df['video_id'].astype(str).isin(common_video_ids)]

#     # Sort the dataframes by video_id to ensure alignment
#     ground_truth = ground_truth.sort_values(by='video_id').reset_index(drop=True)
#     response_df = response_df.sort_values(by='video_id').reset_index(drop=True)
#     # assert video ids are aligned
#     # print("Asserting video ID alignment between ground truth and response dataframes...")
#     # assert all(ground_truth['video_id'].astype(str) == response_df['video_id'].astype(str))
#     # ground_truth = ground_truth[ground_truth['video_id'].astype(str).isin(response_df['video_id'].astype(str))]
#     print(f"Ground truth size: {len(ground_truth)}, Response size: {len(response_df)}")

#     evaluation_results = evaluate_with_stats(ground_truth, response_df)

#     save_stats_to_json(MODEL_NAME, model_load_time, total_time, summary_times, category_times, evaluation_results, STAT_JSON_PATH)

#     print("✅ Inference completed.")

# if __name__ == "__main__":
#     main()

# Comment out from here while running the simple batch processing
class VideoInferenceDataset(Dataset):
    def __init__(self, video_ids, processed_dir):
        self.video_ids = video_ids
        self.processed_dir = processed_dir

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid_id = self.video_ids[idx]
        path = os.path.join(self.processed_dir, f"{vid_id}.pt")
        data = torch.load(path)
        return vid_id, data["summary"], data["category"]

# Padding to max length in that particular batch
def pad_sequence_batch(batch_items, processor):
    max_frames = max(item["pixel_values"].shape[0] for item in batch_items)
    max_seq_len = max(item["input_ids"].shape[0] for item in batch_items)
    
    pixel_values = []
    input_ids = []
    attention_mask = []

    for item in batch_items:
        pv = item["pixel_values"]
        curr_frames = pv.shape[0]
        if curr_frames < max_frames:
            padding = torch.zeros((max_frames - curr_frames, *pv.shape[1:]), dtype=pv.dtype)
            pv = torch.cat([pv, padding], dim=0)
        pixel_values.append(pv)

        ids = item["input_ids"]
        mask = item["attention_mask"]
        curr_len = ids.shape[0]
        diff = max_seq_len - curr_len

        if diff > 0:
            pad_id = processor.tokenizer.pad_token_id
            ids_pad = torch.full((diff,), pad_id, dtype=ids.dtype)
            mask_pad = torch.zeros((diff,), dtype=mask.dtype) 
            
            ids = torch.cat([ids_pad, ids], dim=0)
            mask = torch.cat([mask_pad, mask], dim=0)
        
        input_ids.append(ids)
        attention_mask.append(mask)

    return {
        "pixel_values": torch.stack(pixel_values),
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask)
    }

def custom_collate_fn(batch):
    vid_ids, sum_data, cat_data = zip(*batch)
    batch_sum = pad_sequence_batch(sum_data, processor_global)
    batch_cat = pad_sequence_batch(cat_data, processor_global)
    
    return list(vid_ids), batch_sum, batch_cat

processor_global = None 

def main():
    global processor_global
    
    video_ids = [v for v in os.listdir(VIDEO_FOLDER) if v.endswith(('.mp4'))]
    if SAMPLE_VIDEO:
        video_ids = video_ids[:SAMPLE_SIZE]
    
    video_data_map = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(process_single_audio, video_ids)
        for vid_file, (vid_id, transcript) in zip(video_ids, results):
            video_data_map[vid_id] = {"transcript": transcript, "file_name": vid_file}

    processor_global, model, model_load_time = load_model(MODEL_NAME, DEVICE)

    # loading the saved tensor directory
    PROCESSED_DIR = "processed_tensors_cache"
    preprocess_dataset(video_data_map, processor_global, PROCESSED_DIR)

    dataset = VideoInferenceDataset(list(video_data_map.keys()), PROCESSED_DIR)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=16, 
        shuffle=False, 
        num_workers=2, 
        collate_fn=custom_collate_fn 
    )

    start_time = time.time()
    res_dict, summary_times, category_times = {}, [], []

    print(f"Starting inference on {len(dataset)} videos...")
    
    for i, (batch_vid_ids, batch_sum_inputs, batch_cat_inputs) in enumerate(dataloader):
        
        summaries, t1 = run_inference_tensor_mode(processor_global, model, batch_sum_inputs, max_new_tokens=140)
        summary_times.append(t1)

        categories, t2 = run_inference_tensor_mode(processor_global, model, batch_cat_inputs, max_new_tokens=64)
        category_times.append(t2)

        for j, vid_id in enumerate(batch_vid_ids):
            res_dict[vid_id] = {"summary": summaries[j], "category": categories[j]}
        
        print(f"Processed batch {i+1}/{len(dataloader)}")

    total_time = time.time() - start_time

    response_df = save_results_to_csv(res_dict, CSV_PATH)

    ground_truth = pd.read_csv(GROUND_TRUTH_FILE)
    # response_df = pd.DataFrame.from_dict(res_dict, orient="index").reset_index()
    # response_df.rename(columns={"index": "video_id"}, inplace=True)

    ground_truth_video_ids = set(ground_truth['video_id'].astype(str).tolist())
    response_video_ids = set(response_df['video_id'].astype(str).tolist())
    common_video_ids = ground_truth_video_ids.intersection(response_video_ids)
    ground_truth = ground_truth[ground_truth['video_id'].astype(str).isin(common_video_ids)]
    response_df = response_df[response_df['video_id'].astype(str).isin(common_video_ids)]

    # Sort the dataframes by video_id to ensure alignment
    ground_truth = ground_truth.sort_values(by='video_id').reset_index(drop=True)
    response_df = response_df.sort_values(by='video_id').reset_index(drop=True)
    # assert video ids are aligned
    # print("Asserting video ID alignment between ground truth and response dataframes...")
    # assert all(ground_truth['video_id'].astype(str) == response_df['video_id'].astype(str))
    # ground_truth = ground_truth[ground_truth['video_id'].astype(str).isin(response_df['video_id'].astype(str))]
    print(f"Ground truth size: {len(ground_truth)}, Response size: {len(response_df)}")

    evaluation_results = evaluate_with_stats(ground_truth, response_df)

    save_stats_to_json(MODEL_NAME, model_load_time, total_time, summary_times, category_times, evaluation_results, STAT_JSON_PATH)

    print("✅ Inference completed.")

if __name__ == "__main__":
    main()