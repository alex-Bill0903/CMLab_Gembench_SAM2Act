import os
import json

def save_instruction(taskvar: str, instruction: str):
    """
    讀取或建立 JSON，然後更新指定 taskvar 的 instruction。
    """
    # 確保資料夾存在
    BASE_DIR = 'data/experiments/gembench/3dlotus/v1'
    JSON_FILE = os.path.join(BASE_DIR, 'tasks_instruction.json')
    os.makedirs(BASE_DIR, exist_ok=True)

    # 載入已存在的 JSON，如果不存在就用空 dict
    if os.path.isfile(JSON_FILE):
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    # 比對並更新
    old = data.get(taskvar)
    if old is not None and old == instruction:
        return  # 相同就不必重新寫檔

    data[taskvar] = instruction

    # 寫回 JSON
    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved to {JSON_FILE}")