import os
import json
import cv2
from pathlib import Path
from validator.evaluation.evaluation import GSRValidator
from validator.challenge.challenge_types import GSRResponse, ValidationResult, GSRChallenge, ChallengeType
from validator.utils.frame_filter import (detect_pitch, batch_clip_verification, init_clip_model)
import random
import time
from datetime import datetime, timezone

async def test_evaluation():
    
    test_video = "result_0.mp4"

    """Test evaluation using sample files from debug_frames directory."""
    # Get OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Initialize validator
    validator = GSRValidator(openai_api_key=openai_api_key, validator_hotkey="")

    # Load test files
    debug_frames_dir = Path(".")
    response_file = debug_frames_dir / "result_0.json"
    video_path = debug_frames_dir / test_video

    # Load response data
    with open(response_file, 'r') as f:
        response_data = json.load(f)

    # Create GSRResponse object
    response = GSRResponse(
        challenge_id=response_data.get('challenge_id', '0'),
        miner_hotkey=response_data.get('miner_hotkey', 'test_validator_hotkey'),
        frames=response_data.get('frames', {}),
        node_id=response_data.get('node_id', 1)
    )

    # Select frames for this challenge
    frame_paths = []
    frame_indices = []
    video_cap = cv2.VideoCapture(str(video_path))
    init_clip_model()
    for idx in range(int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = video_cap.read()
        if not success:
            continue
        tmp_path = f"imgs/_{idx}.jpg"
        cv2.imwrite(str(tmp_path), frame)
        frame_paths.append(str(tmp_path))
        frame_indices.append(idx)
    
    clip_scores = batch_clip_verification(frame_paths)
    
    frames = []
    for i, path in enumerate(frame_paths):
        score = detect_pitch(path, clip_scores=clip_scores)
        if score == 1:
            frames.append(frame_indices[i])
    video_cap.release()

    if len(frames)<75:
        return
    
    selected_frames_id_bbox = random.sample(frames, min(100, len(frames)))

    # Create and queqe tasks for each response
    evaluation_results = []

    result = await validator.evaluate_response(
        response = response,
        challenge=GSRChallenge(
            challenge_id="1",
            type=ChallengeType.GSR,
            created_at=datetime.now(timezone.utc),
            video_url=test_video
        ),
        video_path=test_video,
        frames_to_validate=frames,
        selected_frames_id_bbox=selected_frames_id_bbox
    )

    print(f"validator result >>> {result.score} frame_scores >>> {result.frame_scores}")
    print(f"feedback frame_details >>> {result.feedback}")

    bbox_score = result.score
    keypoints_final_score = 0.0
    feedback = result.feedback
    if isinstance(feedback, dict) and "keypoints_final_score" in feedback:
        keypoints_final_score = feedback["keypoints_final_score"]
    quality_score= (bbox_score*0.90) + (keypoints_final_score*0.10)

    print(f"quality_score >>> {quality_score}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_evaluation())
