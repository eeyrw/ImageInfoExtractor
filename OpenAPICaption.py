import asyncio
import base64
import os
import io
import re
import json
import random
from PIL import Image
from openai import AsyncOpenAI, AuthenticationError
from pillow_heif import register_heif_opener
import polars as pl

# -------------------------------
# 初始化
# -------------------------------
register_heif_opener()  # 支持 HEIC 文件

# -------------------------------
# 配置区
# -------------------------------
API_KEY = os.getenv("OPENAI_API_KEY") or "sk-qjf95hrFHzQ79UydxLz1AIMmQg3G5bC9g2FnyJpsl2oNDk7F"  # 建议用环境变量
MODEL = "gemini-2.5-pro"
TARGET_SHORT = 768
IMG_DIR = "/mnt/yuansnas/Backup/big_server/ds/DiffusionDataset/artman/photo/twitter_nsfw_data/raw_media3"
PROMPT_FILE = "prompt3.txt"
MAX_CONCURRENCY = 5
MAX_RETRIES = 3
BASE_BACKOFF = 1.0

# 初始化 AsyncOpenAI 客户端
client = AsyncOpenAI(api_key=API_KEY, base_url="https://www.chataiapi.com/v1")

# 全局停止标志：用于 No more token 场景
STOP_ALL_TASKS = False

# -------------------------------
# 工具函数
# -------------------------------

def load_prompt(path=PROMPT_FILE) -> str:
    """从外部文件读取 prompt 文本"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_image_as_data_uri(path: str) -> str:
    """
    打开本地图片 → 缩放最短边到 TARGET_SHORT → 转成 WebP → 返回 data URI
    """
    with Image.open(path) as img:
        img = img.convert("RGB")  # 转 RGB 避免透明或 P 模式问题
        w, h = img.size
        scale = TARGET_SHORT / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/webp;base64,{b64}"


def extract_json(text: str):
    """
    尝试从模型输出中提取 JSON
    支持：
      1. 纯 JSON
      2. Markdown ```json ... ```
      3. 文字 + JSON
    返回 dict 或 None
    """
    text = text.strip()
    # 1. 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. 去掉 Markdown 包裹
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. 提取最外层 { ... }
    brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(1))
        except json.JSONDecodeError:
            pass

    return None

# -------------------------------
# 核心函数
# -------------------------------

async def caption_image(path: str, custom_id: str, prompt: str):
    """
    单张图片生成 caption，带重试 + 指数退避
    遇到 No more token 会设置 STOP_ALL_TASKS=True
    """
    global STOP_ALL_TASKS

    if STOP_ALL_TASKS:
        print(f"⚠️ Skipping {custom_id} because token exhausted")
        return {"id": custom_id, "filename": os.path.basename(path), "caption": None}

    img_data_uri = load_image_as_data_uri(path)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": [{"type": "image_url", "image_url": img_data_uri}]},
                ],
            )

            caption_text = resp.choices[0].message.content
            caption_json = extract_json(caption_text)
            caption = caption_json  # 如果你的 JSON 有具体字段可以改成 caption_json.get("caption")

            print({"id": custom_id, "filename": os.path.basename(path), "caption": caption})
            return {"id": custom_id, "filename": os.path.basename(path), "caption": caption}

        except AuthenticationError as e:
            print(f"❌ No more token detected: {e}")
            STOP_ALL_TASKS = True
            return {"id": custom_id, "filename": os.path.basename(path), "caption": None}

        except Exception as e:
            print(f"❌ Attempt {attempt} failed for {custom_id} ({path}): {e}")
            if attempt < MAX_RETRIES:
                backoff = BASE_BACKOFF * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                print(f"⏳ Retrying after {backoff:.1f}s...")
                await asyncio.sleep(backoff)
            else:
                print(f"❌ Failed after {MAX_RETRIES} attempts for {custom_id}")
                return {"id": custom_id, "filename": os.path.basename(path), "caption": None}

async def process_images(image_paths, prompt: str, max_concurrency=5):
    """
    并发处理多张图片
    如果 STOP_ALL_TASKS 被触发，会提前取消剩余任务
    """
    sem = asyncio.Semaphore(max_concurrency)
    global STOP_ALL_TASKS

    async def sem_task(i, path):
        async with sem:
            return await caption_image(path, f"img_{i}", prompt)

    tasks = [asyncio.create_task(sem_task(i, path)) for i, path in enumerate(image_paths)]
    results = []

    for task in asyncio.as_completed(tasks):
        if STOP_ALL_TASKS:
            # 取消剩余未完成任务
            for t in tasks:
                if not t.done():
                    t.cancel()
            print("⚠️ Stopping all tasks due to token exhaustion")
            break
        try:
            res = await task
            results.append(res)
        except asyncio.CancelledError:
            results.append({"id": None, "filename": None, "caption": None})
    return results

# -------------------------------
# 主函数
# -------------------------------

def main():
    prompt = load_prompt(PROMPT_FILE)

    # 过滤支持的图片文件
    image_paths = [
        os.path.join(IMG_DIR, f)
        for f in os.listdir(IMG_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".heic"))
    ]

    # 并发生成 caption
    results = asyncio.run(process_images(image_paths, prompt, max_concurrency=MAX_CONCURRENCY))

    # 使用 Polars 保存 Parquet
    df = pl.DataFrame(results)
    df.write_parquet("captions.parquet")

    print("✅ Captions saved to captions.parquet")


if __name__ == "__main__":
    main()
