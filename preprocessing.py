import os, time
from inference_batch import (
    create_messages, run_inference, save_results_to_csv, save_stats_to_json
)
from concurrent.futures import ThreadPoolExecutor

def preprocess_single_video(args):
    vid_id, data, processor, output_dir = args
    save_path = os.path.join(output_dir, f"{vid_id}.pt")
    
    # Check if the video is already processed or not
    if os.path.exists(save_path):
        return
    
    video_path = os.path.join(VIDEO_FOLDER, data["file_name"])
    transcript = data["transcript"]

    sum_msg = create_messages(video_path, transcript, mode="summary")
    cat_msg = create_messages(video_path, transcript, mode="category")

    try:
        summary_input = processor.apply_chat_template(
            sum_msg, add_generation_prompt=True, tokenize=True,return_dict=True, return_tensors="pt", padding=False, num_frames=64
        )
        
        category_input = processor.apply_chat_template(
            cat_msg, add_generation_prompt=True, tokenize=True,return_dict=True, return_tensors="pt", padding=False, num_frames=64
        )

        processed_data = {
            "summary": {k: v.squeeze(0).to("cpu") for k, v in summary_input.items()},
            "category": {k: v.squeeze(0).to("cpu") for k, v in category_input.items()}
        }
        
        torch.save(processed_data, save_path)
        
    except Exception as e:
        print(f"Failed to preprocess {vid_id}: {e}")

# Preprocess the entire dataset
def preprocess_dataset(video_data_map, processor, output_dir="processed_tensors"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing {len(video_data_map)} videos to '{output_dir}'")
    
    tasks = [(vid, data, processor, output_dir) for vid, data in video_data_map.items()]
    # Parallel preprocessing of data
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(preprocess_single_video, tasks))
    
    print("Preprocessing complete")