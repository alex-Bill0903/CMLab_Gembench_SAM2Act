import json
import os
from datetime import datetime
from typing import List
import cv2

from plan_agent import *

def read_json(json_path):
    with open(json_path) as f:
        instructions = json.load(f)
    return instructions

def read_demo_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def test_planning(agent, instructions):
    planning_results = {}
    for task, instruction_list in tqdm(instructions.items(), total=len(instructions), ncols=100):
        plan = agent.get_plan(instruction_list)
        planning_results[task] = {"instruction": instruction_list, "result": plan}
    return planning_results

def run_test_planning(
    agent: PlanAgent,
    json_path = 'tasks_instruction.json',
    output_path = 'planning_results.json'
):
    instructions = read_json(json_path)
    planning_results = test_planning(agent, instructions)
    with open(output_path, 'w') as f:
        json.dump(planning_results, f, indent=4)
    
def test_subgoal_validation(agent: PlanAgent, instruction_list: List[str], video_frames, frames_freq=5):
    subgoals = agent.get_plan(instruction_list)
    subgoal_idx = 0
    _current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    output_folder = f'test/{_current_time}'
    os.makedirs(output_folder, exist_ok=True)
    for i in range(0, len(video_frames), frames_freq):
        frame = video_frames[i]
        frame_path = os.path.join(output_folder, f'frame_{i}.png')
        cv2.imwrite(frame_path, frame)
        finished, response = agent.verify_subgoal(subgoals[subgoal_idx], frame_path)
        with open(os.path.join(output_folder, f'frame_{i}_response.txt'), 'w') as f:
            f.write(f"{subgoals[subgoal_idx]}\n{response}")
        if finished:
            print(f"Subgoal {subgoal_idx} is achievable at frame {i}")
            subgoal_idx += 1
            if subgoal_idx >= len(subgoals):
                break
    print("All subgoals verified or no more frames available.")

def run_test_subgoal_validation(
    agent: PlanAgent,
    json_path = 'tasks_instruction.json',
    video_path = 'demo_video.mp4',
):
    instruction_list = read_json(json_path)
    video_frames = read_demo_video(video_path)
    test_subgoal_validation(agent, instruction_list, video_frames)
        
if __name__ == "__main__":
    print("Load Gemma Plan Agent")
    agent = QwenPlanAgent()
    print("Load Gemma Plan Agent Done")
    run_test_planning(agent, json_path='tasks_instruction.json', output_path='planning_results_qwen14b_bf16_subtask_list_v2.json')
    # task_root = 'push_buttons4+2/0_SR1.0'
    # instruction_path = os.path.join(task_root, 'instruction.json')
    # video_path = os.path.join(task_root, 'global.mp4')
    # print("Run test subgoal validation")
    # run_test_subgoal_validation(agent, instruction_path, video_path)
    
