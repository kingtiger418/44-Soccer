import os
import json
import time
from typing import Optional, Dict, Any
import supervision as sv
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
import asyncio
from pathlib import Path
from loguru import logger

from fiber.logging_utils import get_logger
from miner.core.models.config import Config
from miner.core.configuration import factory_config
from miner.dependencies import get_config, verify_request, blacklist_low_stake
from sports.configs.soccer import SoccerPitchConfiguration
from miner.utils.device import get_optimal_device
from miner.utils.model_manager import ModelManager
from miner.utils.video_processor import VideoProcessor
from miner.utils.shared import miner_lock
from miner.utils.video_downloader import download_video

from numpy import ndarray, zeros
from transformers import CLIPProcessor, CLIPModel

import torch
import cv2
from torch import no_grad
from PIL import Image
import concurrent.futures

import random

logger = get_logger(__name__)

CONFIG = SoccerPitchConfiguration()

# Global model manager instance
model_manager = None

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
data_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# clip_model = CLIPModel.from_pretrained("openai/clip-rn50").to("cuda")
# data_processor = CLIPProcessor.from_pretrained("openai/clip-rn50")

def get_model_manager(config: Config = Depends(get_config)) -> ModelManager:
    global model_manager
    if model_manager is None:
        model_manager = ModelManager(device=config.device)
        model_manager.load_all_models()
    return model_manager

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute intersection width and height
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute union area
    unionArea = boxAArea + boxBArea - interArea

    if unionArea == 0:
        return 0.0  # avoid division by zero

    # iou = interArea / unionArea
    iou = interArea / boxBArea

    return iou

# original code
async def process_soccer_video(
    video_path: str,
    model_manager: ModelManager,
) -> Dict[str, Any]:
    """Process a soccer video and return tracking data."""
    start_time = time.time()
    
    try:
        video_processor = VideoProcessor(
            device=model_manager.device,
            cuda_timeout=10800.0,
            mps_timeout=10800.0,
            cpu_timeout=10800.0
        )
        
        if not await video_processor.ensure_video_readable(video_path):
            raise HTTPException(
                status_code=400,
                detail="Video file is not readable or corrupted"
            )
        
        player_model = model_manager.get_model("player")
        pitch_model = model_manager.get_model("pitch")
        
        tracker = sv.ByteTrack()
        
        tracking_data = {"frames": []}
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 1")
        
        async for frame_number, frame in video_processor.stream_frames(video_path):
            x = frame_number
            pitch_result = pitch_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
            
            player_result = player_model(frame, imgsz=320, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(player_result)
            detections = tracker.update_with_detections(detections)
            
            # Convert numpy arrays to Python native types
            frame_data = {
                "frame_number": int(frame_number),  # Convert to native int
                "keypoints": keypoints.xy[0].tolist() if keypoints and keypoints.xy is not None else [],
                "objects": [
                    {
                        "id": int(tracker_id),  # Convert numpy.int64 to native int
                        "bbox": [float(x) for x in bbox],  # Convert numpy.float32/64 to native float
                        "class_id": int(class_id)  # Convert numpy.int64 to native int
                    }
                    for tracker_id, bbox, class_id in zip(
                        detections.tracker_id,
                        detections.xyxy,
                        detections.class_id
                    )
                ] if detections and detections.tracker_id is not None else []
            }
            tracking_data["frames"].append(frame_data)
            
            if frame_number % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_number / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {frame_number} frames in {elapsed:.1f}s ({fps:.2f} fps)")
        
        processing_time = time.time() - start_time
        tracking_data["processing_time"] = processing_time
        
        total_frames = len(tracking_data["frames"])
        fps = total_frames / processing_time if processing_time > 0 else 0
        logger.info(
            f"Completed processing {total_frames} frames in {processing_time:.1f}s "
            f"({fps:.2f} fps) on {model_manager.device} device"
        )

        frames = []
        async for frame_number, frame in video_processor.stream_frames(video_path):
            if frame_number > 250:
                break
            frames.append(frame)
        
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 2")
        return tracking_data
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")

def is_touching_scoreboard_zone(bbox, frame_width=1280, frame_height=720):
    # x1, y1, x2, y2 = bbox_dict
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    
    scoreboard_top = 0
    scoreboard_bottom = 150
    scoreboard_left = 0
    scoreboard_right = frame_width

    # If the bbox intersects with the top area, it's out of bounds
    intersects_top = not (x2 < scoreboard_left or x1 > scoreboard_right or y2 < scoreboard_top or y1 > scoreboard_bottom)
    return intersects_top

def extract_regions_of_interest_from_image_frame(frame_number, bboxes, image_array:ndarray) -> list[ndarray]:

    rois = []
    obj_index = 0
    for bbox in bboxes:
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])

        roi = image_array[y1:y2, x1:x2, :].copy() 
        rois.append({'frame_index': frame_number,'object_index': obj_index, 'roi': roi})
        image_array[y1:y2, x1:x2, :] = 0
        obj_index += 1

    return rois

def extract_regions_of_interest_from_image(bboxes, image_array:ndarray) -> list[ndarray]:

    rois = []
    for bbox in bboxes:

        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])

        roi = image_array[y1:y2, x1:x2, :].copy() 
        rois.append(roi)
        image_array[y1:y2, x1:x2, :] = 0  

    return rois

# testing code
async def process_soccer_video_V10(
    video_path: str,
    model_manager: ModelManager,
) -> Dict[str, Any]:

    clip_count = 10

    """Process a soccer video and return tracking data."""
    start_time = time.time()
    
    try:
        video_processor = VideoProcessor(
            device=model_manager.device,
            cuda_timeout=10800.0,
            mps_timeout=10800.0,
            cpu_timeout=10800.0
        )
        
        if not await video_processor.ensure_video_readable(video_path):
            raise HTTPException(
                status_code=400,
                detail="Video file is not readable or corrupted"
            )
        
        player_model = model_manager.get_model("player")
        pitch_model = model_manager.get_model("pitch")
        
        tracker = sv.ByteTrack()
        
        tracking_data = {"frames": []}
        
        # all_rois = []
        keypoints = {}
        async for frame_number, frame in video_processor.stream_frames(video_path):

            if frame_number % 25 == 0:
                pitch_result = pitch_model(frame, verbose=False)[0]
                keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
            
            player_result = player_model(frame, imgsz=320, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(player_result)
            detections = tracker.update_with_detections(detections)

            rois_detects = []
            # ################# openclip model ################
            # step1: 
            rois = extract_regions_of_interest_from_image_frame(
                frame_number=frame_number,
                bboxes=rois_detects,
                image_array=frame[:,:,::-1] # BGR -> RGB
            )
            all_rois.extend(rois)

            # step2:
            # d_conf = 0.7
            # for bbox, conf in zip(
            #     detections.xyxy,
            #     detections.confidence
            # ):  
            #     if conf < d_conf:
            #         rois_detects.append(bbox)
            
            # rois = extract_regions_of_interest_from_image(
            #     bboxes=rois_detects,
            #     image_array=frame[:,:,::-1] # BGR -> RGB
            # )

            # refine_preds = []
            # if len(rois) > 0:
            #     person_labels = [
            #         "goalkeeper",
            #         "football player",
            #         "referee",
            #         "crowd",
            #         "black shape"
            #     ]
            #     refine_inputs = data_processor(
            #         text=person_labels,
            #         images=rois,
            #         return_tensors="pt",
            #         padding=True
            #     ).to("cuda")
            #     with no_grad():
            #         refine_outputs = clip_model(**refine_inputs)
            #         refine_probs = refine_outputs.logits_per_image.softmax(dim=1)
            #         refine_preds = refine_probs.argmax(dim=1)

            # # ##################### object gets ################
            objects = []
            d_index = 0
            for tracker_id, bbox, class_id in zip(
                detections.tracker_id,
                detections.xyxy,
                detections.class_id,
            ):
                x1 = bbox[0]
                y1 = bbox[1]
                x2 = bbox[2]
                y2 = bbox[3]

                dx = x2 - x1
                dy = y2 - y1

                if dx/dy < 0.7:
                    objects.append({
                        "id": int(tracker_id),
                        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        # "class_id": int(class_id) if d_index < len(detections.class_id) - len(rois_detects) else int(refine_preds[d_index + len(rois_detects) - len(detections.class_id)] + 1),
                        "class_id": int(tracker_id)
                    })
                d_index += 1

            # # Convert numpy arrays to Python native types
            frame_data = {
                "frame_number": int(frame_number),  # Convert to native int
                "keypoints": keypoints.xy[0].tolist() if keypoints and keypoints.xy is not None else [],
                "objects": objects
            }
            tracking_data["frames"].append(frame_data)
            
            if frame_number % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_number / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {frame_number} frames in {elapsed:.1f}s ({fps:.2f} fps)")

        ############## clip processing 12S !!!! ################
        clip_time = time.time()
        person_labels = [
            "goalkeeper",
            "football player",
            "referee",
            "crowd",
            "black shape"
        ]
        batch_size = 200
        all_preds = []
        
        for i in range(0, len(all_rois), batch_size):
            batch_rois = all_rois[i:i+batch_size]
            batch_inputs = data_processor(
                text=person_labels,
                images=[item['roi'] for item in batch_rois],
                return_tensors="pt",
                padding=True
            ).to("cuda")
            with no_grad():
                batch_outputs = clip_model(**batch_inputs)
                batch_probs = batch_outputs.logits_per_image.softmax(dim=1)
                batch_preds = batch_probs.argmax(dim=1)
                all_preds.extend(batch_preds.cpu().tolist())
        
        # #################### convert class_id #####################
        for rois, pred in zip(
            all_rois,
            all_preds
        ):
            frame_index = rois["frame_index"]
            object_index = rois["object_index"]
            
            # if pred > 2:
            #     del tracking_data["frames"][frame_index]["objects"][object_index]
            #     continue

            tracking_data["frames"][frame_index]["objects"][object_index]["class_id"] = pred + 1
     
        ####################### calculate #######################
        clip_process_time = time.time() - clip_time
        logger.info(f" ====== >>>>>>> clip process time = {clip_process_time:.1f}s")


        ####################### end #######################
        processing_time = time.time() - start_time
        tracking_data["processing_time"] = processing_time
        
        total_frames = len(tracking_data["frames"])
        fps = total_frames / processing_time if processing_time > 0 else 0
        logger.info(
            f"Completed processing {total_frames} frames in {processing_time:.1f}s "
            f"({fps:.2f} fps) on {model_manager.device} device"
        )
        
        return tracking_data
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")

# testing code
async def process_soccer_video_V11(
    video_path: str,
    model_manager: ModelManager,
) -> Dict[str, Any]:
    """Process a soccer video and return tracking data."""
    start_time = time.time()
    
    try:
        video_processor = VideoProcessor(
            device=model_manager.device,
            cuda_timeout=10800.0,
            mps_timeout=10800.0,
            cpu_timeout=10800.0
        )
        
        if not await video_processor.ensure_video_readable(video_path):
            raise HTTPException(
                status_code=400,
                detail="Video file is not readable or corrupted"
            )
        
        # player_model = model_manager.get_model("player")
        # pitch_model = model_manager.get_model("pitch")
        
        # tracker = sv.ByteTrack()
        
        tracking_data = {"frames": []}


        # detections = [
        #     [0, 151, 30, 220]
        #     for 
        # ]
        # "objects": [
        #     {
        #         "id": int(tracker_id),  # Convert numpy.int64 to native int
        #         "bbox": [float(x) for x in bbox],  # Convert numpy.float32/64 to native float
        #         "class_id": int(class_id)  # Convert numpy.int64 to native int
        #     }
        #     for tracker_id, bbox, class_id in zip(
        #         detections.tracker_id,
        #         detections.xyxy,
        #         detections.class_id
        #     )
        # ] if detections and detections.tracker_id is not None else []
        detections = []
        for i in range(int(570/70)):
            for j in range(int(1280/30)):
                detections.append([0 + j * 30, 150 + i * 70, 30 + j * 30, 220 + i * 70])
        
        # print(detections)

        async for frame_number, frame in video_processor.stream_frames(video_path):

            # pitch_result = pitch_model(frame, verbose=False)[0]
            # keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
            
            # player_result = player_model(frame, imgsz=1280, verbose=False)[0]
            # detections = sv.Detections.from_ultralytics(player_result)
            # detections = tracker.update_with_detections(detections)

            ################# make the object ###############
            print(f">>>>>>>>>>>>>>>>>>>>>>>>> {frame_number}")

            ################# openclip model ################
            rois = extract_regions_of_interest_from_image(
                bboxes=detections,
                image_array=frame[:,:,::-1] # BGR -> RGB
            )

            refine_preds = []
            start_t = time.time()
            if len(rois) > 0:
                person_labels = [
                    "goalkeeper",
                    "football player",
                    "referee",
                    "crowd",
                    "black shape"
                ]
                refine_inputs = data_processor(
                    text=person_labels,
                    images=rois,
                    return_tensors="pt",
                    padding=True
                ).to("cuda")
                with no_grad():
                    refine_outputs = clip_model(**refine_inputs)
                    refine_probs = refine_outputs.logits_per_image.softmax(dim=1)
                    refine_preds = refine_probs.argmax(dim=1)

            ##################### object gets ################
            objects = []
            for bbox, pred in zip(
                detections,
                refine_preds
            ):
                x1 = bbox[0]
                y1 = bbox[1]
                x2 = bbox[2]
                y2 = bbox[3]

                if pred > 2:
                    continue

                objects.append({
                    "id": 0,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "class_id": int(pred) + 1
                })

            # Convert numpy arrays to Python native types
            frame_data = {
                "frame_number": int(frame_number),  # Convert to native int
                "keypoints": [],
                "objects": objects
            }
            tracking_data["frames"].append(frame_data)
            
            if frame_number % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_number / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {frame_number} frames in {elapsed:.1f}s ({fps:.2f} fps)")
        
        processing_time = time.time() - start_time
        tracking_data["processing_time"] = processing_time
        
        total_frames = len(tracking_data["frames"])
        fps = total_frames / processing_time if processing_time > 0 else 0
        logger.info(
            f"Completed processing {total_frames} frames in {processing_time:.1f}s "
            f"({fps:.2f} fps) on {model_manager.device} device"
        )
        
        return tracking_data
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")

# final code
async def process_soccer_video_v3(
    video_path: str,
    model_manager: ModelManager,
    conf: str,
    iou: str,
    imgsz: str,
    step_: str
) -> Dict[str, Any]:
    """Process a soccer video and return tracking data, inferring only every 3rd frame."""
    start_time = time.time()

    conf_ = float(conf)
    iou_ = float(iou)
    imgsz_ = int(float(imgsz))
    step = int(step_)

    try:
        video_processor = VideoProcessor(
            device=model_manager.device,
            cuda_timeout=10800.0,
            mps_timeout=10800.0,
            cpu_timeout=10800.0
        )
        if not await video_processor.ensure_video_readable(video_path):
            raise HTTPException(400, "Video file is not readable or corrupted")

        player_model = model_manager.get_model("player")
        pitch_model = model_manager.get_model("pitch")
        tracker = sv.ByteTrack()

        tracking_data = {"frames": []}
        prev_data = None
        buffered_idxs: List[int] = []

        async for frame_number, frame in video_processor.stream_frames(video_path):
            # Only do real inference on every `step`th frame
            if frame_number % step == 0:
                # 1) Pitch/keypoint inference
                pitch_result = pitch_model(frame, verbose=False)[0]
                keypts = sv.KeyPoints.from_ultralytics(pitch_result)
                kps = keypts.xy[0].tolist() if keypts and keypts.xy is not None else []

                # 2) Player detection + tracking
                player_result = player_model(
                    frame,
                    imgsz=imgsz_,
                    verbose=False
                    # conf=conf_,
                    # iou=iou_
                )[0]
                dets = sv.Detections.from_ultralytics(player_result)
                dets = tracker.update_with_detections(dets)

                # setting the objects xy
                objects = []
                if len(dets) > 3:
                    for tracker_id, bbox, class_id, confidence in zip(
                        dets.tracker_id,
                        dets.xyxy,
                        dets.class_id,
                        dets.confidence
                    ):
                        x1 = bbox[0]
                        y1 = bbox[1]
                        x2 = bbox[2]
                        y2 = bbox[3]

                        dx = x2 - x1
                        dy = y2 - y1

                        if dx/dy >0.7:
                            continue

                        objects.append({
                            "id": int(tracker_id),
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "class_id": int(class_id),
                            "confidence": float(confidence)
                        })

                if len(objects) > 10:
                    del objects[-1]
                
                if len(objects) > 10:
                    del objects[-1]
                
                if len(objects) > 10:
                    del objects[-1]

                # Build current frame’s data
                curr_data = {
                    "frame_number": frame_number,
                    "keypoints": kps,
                    "objects": objects
                }

                # 3) If we have a previous real frame and skipped frames buffered,
                #    interpolate them between prev_data and curr_data
                if prev_data is not None and buffered_idxs:
                    prev_n = prev_data["frame_number"]
                    curr_n = frame_number
                    # Map objects by ID for prev & curr
                    prev_map = {o["id"]: o for o in prev_data["objects"]}
                    curr_map = {o["id"]: o for o in curr_data["objects"]}
                    common_ids = set(prev_map) & set(curr_map)

                    for m in buffered_idxs:
                        alpha = (m - prev_n) / (curr_n - prev_n)
                        # interpolate keypoints if shapes match
                        prev_kps = prev_data["keypoints"]
                        curr_kps = curr_data["keypoints"]
                        if len(prev_kps) == len(curr_kps):
                            new_kps = [
                                [
                                    prev_kps[i][c] + alpha * (curr_kps[i][c] - prev_kps[i][c])
                                    for c in (0, 1)
                                ]
                                for i in range(len(prev_kps))
                            ]
                        else:
                            new_kps = prev_kps

                        # interpolate boxes for each common object
                        new_objs = []
                        for oid in sorted(common_ids):
                            p = prev_map[oid]["bbox"]
                            q = curr_map[oid]["bbox"]
                            interp_box = [
                                p[i] + alpha * (q[i] - p[i]) for i in range(4)
                            ]
                            new_objs.append({
                                "id": oid,
                                "bbox": interp_box,
                                "class_id": prev_map[oid]["class_id"]
                            })

                        tracking_data["frames"].append({
                            "frame_number": m,
                            "keypoints": new_kps,
                            "objects": new_objs
                        })
                        print(f"--> {frame_number} frame => {len(new_objs)}")
                    buffered_idxs.clear()

                # 4) Append the actual processed frame
                print(f"{frame_number} frame => {len(curr_data["objects"])}")
                tracking_data["frames"].append(curr_data)
                prev_data = curr_data

                # 5) Log progress on real inferences
                if frame_number %  (step * 10) == 0:
                    elapsed = time.time() - start_time
                    fps = frame_number / elapsed if elapsed > 0 else 0
                    logger.info(f"Inferred {frame_number} frames in {elapsed:.1f}s ({fps:.2f} fps)")

            else:
                # Buffer this index so we can interpolate it later
                buffered_idxs.append(frame_number)

        # 6) After loop, any leftover buffered frames get filled with the last real frame
        if prev_data and buffered_idxs:
            for m in buffered_idxs:
                tracking_data["frames"].append({
                    "frame_number": m,
                    "keypoints": prev_data["keypoints"],
                    "objects": prev_data["objects"]
                })

        # Record total time & fps
        total_time = time.time() - start_time
        tracking_data["processing_time"] = total_time
        total_frames = len(tracking_data["frames"])
        final_fps = total_frames / total_time if total_time > 0 else 0
        logger.info(
            f"Completed {total_frames} frames in {total_time:.1f}s ({final_fps:.2f} fps)"
            f" on {model_manager.device}"
        )

        return tracking_data

    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(500, f"Video processing error: {e}")

async def process_challenge(
    request: Request,
    config: Config = Depends(get_config),
    model_manager: ModelManager = Depends(get_model_manager),
):
    logger.info("Attempting to acquire miner lock...")
    async with miner_lock:
        logger.info("Miner lock acquired, processing challenge...")
        try:
            challenge_data = await request.json()
            challenge_id = challenge_data.get("challenge_id")
            video_url = challenge_data.get("video_url")
            
            logger.info(f"Received challenge data: {json.dumps(challenge_data, indent=2)}")
            
            if not video_url:
                raise HTTPException(status_code=400, detail="No video URL provided")
            
            logger.info(f"Processing challenge {challenge_id} with video {video_url}")
            
            video_path = await download_video(video_url)
            
            try:
                tracking_data = await process_soccer_video(
                    video_path,
                    model_manager
                )
                
                response = {
                    "challenge_id": challenge_id,
                    "frames": tracking_data["frames"],
                    "processing_time": tracking_data["processing_time"]
                }
                
                logger.info(f"Completed challenge {challenge_id} in {tracking_data['processing_time']:.2f} seconds")
                return response
                
            finally:
                try:
                    os.unlink(video_path)
                except:
                    pass
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing soccer challenge: {str(e)}")
            logger.exception("Full error traceback:")
            raise HTTPException(status_code=500, detail=f"Challenge processing error: {str(e)}")
        finally:
            logger.info("Releasing miner lock...")

# Create router with dependencies
router = APIRouter()
router.add_api_route(
    "/challenge",
    process_challenge,
    tags=["soccer"],
    dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    methods=["POST"],
)
