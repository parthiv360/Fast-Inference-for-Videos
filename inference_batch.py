import os, time, numpy as np, pandas as pd, json, torch
from concurrent.futures import ThreadPoolExecutor

def create_messages(video_path: str, transcript: str, mode: str = "summary"):
    """Create system messages for summary or category inference."""
    if mode == "summary":
        text = (
            f"Describe this video in detail. Use the audio transcript to get more context. Audio Transcript: {transcript}"
        )
    else:
        text = (
            "You are a video classification expert. Watch the video carefully and assign it to exactly one of the following categories:\n\n"
            "1. News\n"
            "2. Politics\n"
            "3. Music, Singing, & Dancing\n"
            "4. Comedy\n"
            "5. Sports\n"
            "6. Film & Animation\n"
            "7. Pets & Animals\n"
            "8. Entertainment & Shows\n"
            "9. Gaming\n"
            "10. Science & Technology\n"
            "11. Autos & Vehicles\n"
            "12. Education\n"
            "13. Outfit, Style, & Howto\n"
            "14. Nonprofits & Activism\n"
            "15. Travel & Events\n"
            "16. People & Blogs\n"
            "17. Food\n"
            "18. Relationship\n"
            "19. Family\n"
            "20. Beauty Care\n"
            "21. Daily Life\n"
            "22. Drama\n"
            "23. Lipsync\n"
            "24. Fitness & Health\n"
            "25. Society\n\n"
            "Use both the visuals and the audio transcript to determine the most relevant single category. "
            "Consider what the video is *primarily* about, not minor elements or background music. "
            "Do not include explanations or reasoning — respond with only the category name.\n\n"
            f"Audio Transcript: {transcript}"
        )

    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": video_path},
                {"type": "text", "text": text},
            ],
        }
    ]


def create_messages_old(video_path: str, transcript: str, mode: str = "summary"):
    """Create system messages for summary or category inference."""
    if mode == "summary":
        text = f"Describe this video in detail. Use the audio transcript to get more context. Audio Transcript: {transcript}"
    else:
        text = (
            # "Analyze the video and categorize it into one of the following categories: "
            # "'News, Politics, Music, Comedy, Sports, Film, Pets, Entertainment, Gaming, "
            # "Science, Autos, Education, Style, Nonprofits, Travel, People, Food, "
            # "Relationship, Family, Beauty, Daily Life, Drama, Lipsync, Fitness, Society'. "
            """'News, Politics, `Music, Singing, & Dancing`, Comedy, Sports, Film & Animation, Pets & Animals, Entertainment & Shows, Gaming, Science & Technology, Autos & Vehicles, Education, `Outfit, Style, & Howto`, Nonprofits & Activism, Travel & Events, People & Blogs, Food, Relationship, Family, Beauty Care, Daily Life, Drama, Lipsync, Fitness & Health, Society'`
            Each video should be classified into only one category. Note that Music, Singing, & Dancing is one category and Outfit, Style, & Howto is another category."""
            f"Use the audio transcript to get more context. Audio Transcript: {transcript}. "
            "You must provide only a single category from the provided categories as the response."
        )

    return [{"role": "user", "content": [{"type": "video", "path": video_path}, {"type": "text", "text": text}]}]

# This is used for simple batch processing
def run_inference(processor, model, messages, max_new_tokens=140, mode="category"):
    """Run inference and measure time."""
    start = time.time()
    
    batch_inputs = []
    
    def process_single_message(msg):
        return processor.apply_chat_template(
            [msg], 
            add_generation_prompt=True, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt", 
            padding=False,
            num_frames=64 
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        batch_inputs = list(executor.map(process_single_message, messages))

    #Determine Max Dimensions
    max_frames = max(b["pixel_values"].shape[1] for b in batch_inputs)
    max_seq_len = max(b["input_ids"].shape[1] for b in batch_inputs)

    final_pixel_values = []
    final_input_ids = []
    final_attention_mask = []

    # Padding at the end for pixel values, and at the beginning for Ids and Attention
    for b in batch_inputs:
        pv = b["pixel_values"] 
        curr_frames = pv.shape[1]
        
        if curr_frames < max_frames:
            diff = max_frames - curr_frames
            padding = torch.zeros(
                (1, diff, *pv.shape[2:]), 
                dtype=pv.dtype, 
                device=pv.device 
            )
            pv = torch.cat([pv, padding], dim=1)
        
        final_pixel_values.append(pv.to(model.device, dtype=torch.bfloat16))
        
        ids = b["input_ids"] 
        mask = b["attention_mask"] 
        curr_len = ids.shape[1]
        diff_len = max_seq_len - curr_len

        if diff_len > 0:
            pad_token_id = processor.tokenizer.pad_token_id
            ids_pad = torch.full((1, diff_len), pad_token_id, dtype=ids.dtype, device=ids.device)
            mask_pad = torch.zeros((1, diff_len), dtype=mask.dtype, device=mask.device)
            
            ids = torch.cat([ids_pad, ids], dim=1)
            mask = torch.cat([mask_pad, mask], dim=1)
            
        final_input_ids.append(ids.to(model.device))
        final_attention_mask.append(mask.to(model.device))

    inputs = {
        "input_ids": torch.cat(final_input_ids, dim=0),
        "attention_mask": torch.cat(final_attention_mask, dim=0),
        "pixel_values": torch.cat(final_pixel_values, dim=0)
    }

    if mode == "category":
        max_new_tokens = 64

    ids  = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    texts = processor.batch_decode(ids  , skip_special_tokens=True)
    
    end = time.time()
    results = [text.split("Assistant:")[-1].strip() for text in texts]
    
    return results, end - start

# This is used for preprocessed tensors.
def run_inference_tensor_mode(processor, model, batch_inputs, max_new_tokens=140):
    start = time.time()
    device = model.device
    inputs = {
        "pixel_values": batch_inputs["pixel_values"].to(device, dtype=torch.bfloat16),
        "input_ids": batch_inputs["input_ids"].to(device),
        "attention_mask": batch_inputs["attention_mask"].to(device)
    }
    
    ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    
    end = time.time()
    
    texts = processor.batch_decode(ids, skip_special_tokens=True)
    results = [text.split("Assistant:")[-1].strip() for text in texts]
    
    return results, end - start

def save_results_to_csv(res_dict, csv_path):
    df = pd.DataFrame.from_dict(res_dict, orient="index").reset_index()
    df.rename(columns={"index": "video_id"}, inplace=True)
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved results to {csv_path}")
    return df


def save_stats_to_json(model_name, model_load_time, total_time, summary_times, category_times, evaluation_results, output_path):
    stats = {
        "model_name": model_name,
        "model_load_time": model_load_time,
        "total_time": total_time,
        "summary_inference": {
            "mean": float(np.mean(summary_times)),
            "min": float(np.min(summary_times)),
            "max": float(np.max(summary_times)),
            "std": float(np.std(summary_times)),
        },
        "category_inference": {
            "mean": float(np.mean(category_times)),
            "min": float(np.min(category_times)),
            "max": float(np.max(category_times)),
            "std": float(np.std(category_times)),
        },
        "evaluation_results": evaluation_results
    }
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"📊 Saved statistics to {output_path}")
