"""HyperReality: blind video action detection with human-reviewed Excel approvals.

Run one video at a time:
  python HyperReality_Final_Action_Review.py analyze --video "videos/clip.MOV"
  # Inspect/edit its Review_*.xlsx: Actions and Review sheets; mark YES for good rows.
  python HyperReality_Final_Action_Review.py approve --video "videos/clip.MOV"
  python HyperReality_Final_Action_Review.py status

Important: Qwen detections/timestamps are proposals, NOT verified annotations.
The input video name, case classification and manuals are never sent to the model.
Only rows explicitly approved by a person enter the master workbook.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

VERSION = "10.0-review-first"
MODEL = "Qwen/Qwen3-VL-2B-Instruct"
WINDOW = 5
SAMPLE_FPS = 2
FRAMES_PER_WINDOW = WINDOW * SAMPLE_FPS
PREVIEW_EDGE = 768
MODEL_EDGE = 384
MAX_NEW_TOKENS = 168
OUTPUT_ROOT = Path("outputs") / "HyperReality_Final"
MASTER_FILE = OUTPUT_ROOT / "HyperReality_Master_Approved.xlsx"
MASTER_DB = OUTPUT_ROOT / "approved_master.json"

# Controlled vocabulary. Use no specialized repair checklist or expected steps.
VERBS = {
    "remove", "install", "place", "lift", "lower", "loosen", "tighten",
    "rotate", "open", "close", "adjust", "position", "insert", "pull",
    "push", "pick", "inspect", "handle", "operate", "move", "hold",
    "clean", "attach", "detach", "connect", "disconnect", "release",
    "align", "support", "carry", "turn", "press", "slide", "check",
}
SENSITIVE = {"remove", "install", "loosen", "tighten", "lift", "lower", "connect", "disconnect", "attach", "detach", "release"}
FIXTURE_WORDS = (
    "jack", "lift", "hoist", "workbench", "bench", "table", "stand",
    "vise", "vice", "clamp", "wheel chock", "support block", "fixture", "tool trolley",
)
HEADER = [
    "Timestamp", "Action Type", "Action", "Part Name", "Tool", "Tool Setting",
    "Fixture", "Instructions", "Context / Evidence", "Before", "After", "Action"
]
COL_WIDTHS = [21, 16, 30, 23, 23, 20, 23, 32, 62, 32, 32, 32]

# Short response: one line per 5-second window, one action maximum.
# No filenames, expected outcomes, procedures, or manuals enter this prompt.
PROMPT = """You are reviewing 10 visual observations in TIME ORDER, numbered 0 to 9.
Observe the physical interaction in the images, not what the technician intends.
Report ONLY the clearest distinct visible action, if any. Do NOT guess a
finished removal, installation, lifting, tightening, loosening, or locking
unless a visible physical change supports it. Hand/tool contact alone is not
proof of any hidden mechanical result. Ignore camera movement and narration.
If no meaningful action is clearly observable, output exactly NONE.
Otherwise return ONE line with exactly SEVEN fields separated by | :
verb|object|tool|fixture|first_index|end_index|visual_evidence
verb: ONE simple base-form keyword (remove, place, lift, open, adjust, move,
operate, inspect, handle, etc.). Do not use 'removing' or 'removed'.
object: short name of the directly visible item involved.
tool: name only if clearly identifiable, otherwise empty.
fixture: stationary SUPPORT or HOLDER (such as jack, vehicle lift, bench),
ONLY when plainly visible and supporting the object; otherwise empty.
first_index: first observed evidence index, an integer 0-9.
end_index: one PAST the last observed evidence index, integer 1-10.
visual_evidence: one short factual sentence stating what actually moves or
changes in relation to something fixed; no imagined results.
Use 0 <= first_index < end_index <= 10. No extra explanation, headings,
examples, markdown, labels, additional lines, or guesses. NEVER output the
video filename or a textbook procedure.
"""


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._")[:72] or "video"


def video_identity(video):
    st = video.stat()
    s = f"{video.resolve()}|{st.st_size}|{st.st_mtime_ns}|{VERSION}|{MODEL}|{WINDOW}|{SAMPLE_FPS}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def folder_for(video, root=OUTPUT_ROOT):
    return Path(root) / "reviews" / f"{safe_filename(video.stem)}_{video_identity(video)}"


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def read_json(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def mmss(sec):
    n = max(0, int(round(sec)))
    h, rem = divmod(n, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def parse_clock(v):
    m = re.fullmatch(r"\s*(?:(\d+):)?(\d+):(\d{1,2})\s*", str(v))
    if m:
        if m[1] is not None:
            return int(m[1])*3600 + int(m[2])*60 + int(m[3])
        return int(m[2])*60 + int(m[3])
    raise ValueError(f"Invalid timestamp value: {v!r}")


def parse_timeframe(value):
    bits = re.split(r"\s*[–—-]\s*", str(value).strip())
    if len(bits) != 2:
        raise ValueError(f"Timestamp should be 0:07 - 0:12, got: {value!r}")
    a, b = (parse_clock(x) for x in bits)
    if b <= a:
        raise ValueError(f"End must be after start: {value!r}")
    return a, b


def interval(a, b):
    # Show conservative whole-second boundaries (floor start / ceil end),
    # never round a 0.5-second action into a zero-duration timestamp.
    return f"{mmss(math.floor(a+1e-9))} - {mmss(math.ceil(b-1e-9))}"


def check_environment(needs_model=True):
    if importlib.util.find_spec("xlsxwriter") is None:
        raise RuntimeError("Missing Excel writer. Run once: python -m pip install XlsxWriter")
    if importlib.util.find_spec("PIL") is None:
        raise RuntimeError("Missing image library. Run: python -m pip install Pillow")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is not available on PATH. Verify with: ffmpeg -version")
    if needs_model and (importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None):
        raise RuntimeError("The existing Qwen Python packages aren't available in this .venv")


def video_meta(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or count <= 0:
            raise RuntimeError("Video FPS/duration not readable")
        return {"duration": count/fps, "fps":fps,
                "width":int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height":int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        cap.release()


def windows(duration):
    result = []
    t = 0.0
    while t < duration - .001:
        end = min(duration, t+WINDOW)
        result.append((round(t,3), round(end,3)))
        t = end
    return result


def prepare_images(video, info, folder):
    """Decode once sequentially. 2 fps image cache supports inference and thumbnails."""
    pictures = folder / "frames"
    manifest = pictures / "manifest.json"
    needed = max(1, math.floor(info["duration"] * SAMPLE_FPS)-1)
    if manifest.exists():
        m = read_json(manifest,{})
        if m.get("identity") == video_identity(video) and len(list(pictures.glob("f*.jpg"))) >= needed:
            return pictures
    pictures.mkdir(parents=True, exist_ok=True)
    # Clean stale frame cache only, never review files/approved files.
    for stale in pictures.glob("f*.jpg"):
        stale.unlink()
    vf = f"fps={SAMPLE_FPS},scale={PREVIEW_EDGE}:{PREVIEW_EDGE}:force_original_aspect_ratio=decrease"
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(video),
           "-vf", vf, "-q:v", "4", "-y", str(pictures/"f%06d.jpg")]
    print("Preparing video imagery once:", video.name, flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    n = len(list(pictures.glob("f*.jpg")))
    if p.returncode or n < needed:
        raise RuntimeError(f"FFmpeg failed to sample video ({n}/{needed}): {p.stderr[-700:]}")
    atomic_json(manifest, {"identity":video_identity(video), "frames":n, "sample_fps":SAMPLE_FPS})
    return pictures


def image_at(frames, sec):
    # ffmpeg's first fps=2 output corresponds to approximately 0.25s.
    idx = max(1, int(math.floor(max(0.0, sec)*SAMPLE_FPS)) + 1)
    limit = len(list(frames.glob("f*.jpg")))
    idx = min(idx, limit)
    image = frames / f"f{idx:06d}.jpg"
    if not image.is_file():
        raise RuntimeError(f"Missing extracted image: {image}")
    return image


def load_image_section(frames, start, end):
    from PIL import Image
    contents = []
    last = len(list(frames.glob("f*.jpg")))
    first = int(start*SAMPLE_FPS)
    top = min(last, max(first+1, int(math.ceil(end*SAMPLE_FPS))))
    for i in range(first, top):
        path = frames / f"f{i+1:06d}.jpg"
        if not path.exists():
            continue
        with Image.open(path) as img:
            copy = img.convert("RGB").copy()
            copy.thumbnail((MODEL_EDGE, MODEL_EDGE))
            contents.append((i-first,copy))
    if not contents:
        raise RuntimeError("No frames for a video window")
    return contents


def load_qwen():
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, dtype="auto", device_map="auto", local_files_only=True
    )
    model.eval()
    return model, processor


def infer_once(model, processor, frames):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    from qwen_vl_utils import process_vision_info
    # The temporal indices are given as metadata. These do NOT identify the action.
    content = [{"type":"text", "text":PROMPT}]
    content[0]["text"] = PROMPT.replace("10 visual observations", f"{len(frames)} visual observations").replace("numbered 0 to 9", f"numbered 0 to {len(frames)-1}")
    for idx,img in frames:
        content.extend([{"type":"text", "text":f"Index {idx}."},
                        {"type":"image", "image":img}])
    messages = [{"role":"user", "content":content}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs_img, _ = process_vision_info(messages, image_patch_size=16)
    inputs = processor(text=[chat], images=inputs_img, padding=True, return_tensors="pt").to(model.device)

    class StopAfterRecord(StoppingCriteria):
        def __init__(self, tokenizer, prefix):
            self.tokenizer, self.prefix = tokenizer, prefix
        def __call__(self, input_ids, scores, **kwargs):
            # Short output; stop after one complete line or seven fields.
            out = self.tokenizer.decode(input_ids[0,self.prefix:], skip_special_tokens=True)
            return ("\n" in out.strip() or out.strip().upper() == "NONE"
                    or (out.count("|")>=6 and len(out.split("|",6)[-1].split()) >= 18))
    try:
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                       do_sample=False, repetition_penalty=1.12,
                                       no_repeat_ngram_size=5,
                                       stopping_criteria=StoppingCriteriaList([
                                           StopAfterRecord(processor.tokenizer, inputs.input_ids.shape[1])
                                       ]))
        answer = generated[:,inputs.input_ids.shape[1]:]
        text = processor.batch_decode(answer, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0].strip()
        return text, int(answer.shape[1])
    finally:
        del inputs


def normalize_blank(s):
    t = re.sub(r"\s+", " ", str(s or "")).strip()
    return "" if t.casefold() in {"", "none", "n/a", "na", "unknown", "uncertain", "not visible", "not clear", "-"} else t


def parse_one(raw, count):
    """Return None (NO ACTION), candidate dict, or review placeholder dict.

    Inference errors are never automatically retried, converted to YES, or hidden.
    """
    txt = str(raw or "").strip()
    if txt.upper().rstrip(".") in {"NONE", "NO CLEAR ACTION"}:
        return None, "NO ACTION", ""
    txt = txt.strip("` ")
    lines = [line.strip() for line in txt.splitlines() if line.strip()]
    if len(lines) != 1:
        return None, "REVIEW", "Model did not return exactly one line"
    bits = [x.strip() for x in lines[0].split("|")]
    if len(bits) != 7:
        return None, "REVIEW", f"Expected 7 fields, found {len(bits)}"
    verb, part, tool, fixture, first, last, evidence = bits
    verb=verb.lower().strip()
    if verb not in VERBS or verb in {"verb", "action type"}:
        return None, "REVIEW", f"Invalid action keyword: {verb!r}"
    part=normalize_blank(part).lower()
    if not part or part in {"object", "main object", "part name", "state before", "state after"}:
        return None, "REVIEW", "Missing real object/part name"
    if len(part)>80 or len(evidence)>280:
        return None, "REVIEW", "Excessively long result"
    try:
        first,last=int(first),int(last)
    except ValueError:
        return None, "REVIEW", "Time indices are not integers"
    if not 0 <= first < last <= count:
        return None, "REVIEW", f"Time indices outside 0..{count}"
    evidence=normalize_blank(evidence)
    if not evidence or len(evidence.split())<5 or evidence.lower() in {"visual evidence", "evidence"}:
        return None, "REVIEW", "Missing meaningful visual evidence"
    tool=normalize_blank(tool)
    fixture=normalize_blank(fixture)
    if fixture and not any(word in fixture.lower() for word in FIXTURE_WORDS):
        fixture=""  # Never claim an arbitrary object is a supporting fixture.
    warning = "ACTION OUTCOME NEEDS VIDEO CHECK" if verb in SENSITIVE else "UNVERIFIED — HUMAN REVIEW"
    return {"verb":verb,"part":part,"tool":tool,"fixture":fixture,
            "first":first,"last":last,"evidence":evidence}, "CANDIDATE", warning


def past_tense(verb):
    irregular={"hold":"held","pick":"picked","put":"put","slide":"slid","carry":"carried",
               "remove":"removed","place":"placed","move":"moved","operate":"operated",
               "loosen":"loosened","tighten":"tightened","lift":"lifted","lower":"lowered",
               "rotate":"rotated","handle":"handled","inspect":"inspected","attach":"attached",
               "detach":"detached","press":"pressed","align":"aligned","check":"checked",
               "turn":"turned","open":"opened","close":"closed","clean":"cleaned"}
    if verb in irregular: return irregular[verb]
    if verb.endswith("e"):return verb+"d"
    if verb.endswith("y"):return verb[:-1]+"ied"
    return verb+"ed"


def object_phrase(part):
    """Readable deterministic grammar; never add an object the model didn't identify."""
    if part.startswith(("the ","a ","an ")) or part.endswith("s"):
        return part
    return "the "+part


def candidate_record(video, frame_dir, idx, start, end, pred, status, note, elapsed, raw):
    if pred is None:
        return None
    # Index-to-seconds is a localization estimate; reviewer may correct both boundaries.
    a = min(end, start + pred["first"]/SAMPLE_FPS)
    b = min(end, start + pred["last"]/SAMPLE_FPS)
    if b<=a:
        b=min(end,a+0.5)
    part=pred["part"]
    form=object_phrase(part)
    row = {
        "id": f"e{idx+1:04d}", "start":a, "end":b,
        "timestamp":interval(a,b), "action_type":pred["verb"],
        "action":f"{past_tense(pred['verb']).capitalize()} {form}",
        "part":part, "tool":pred["tool"], "tool_setting":"",
        "fixture":pred["fixture"], "instructions":f"{pred['verb'].capitalize()} {form}",
        "context":pred["evidence"], "status":note,"raw":raw,
        "processing_seconds":elapsed,
        "before":str(image_at(frame_dir,max(0,a-0.5)).resolve()),
        "during":str(image_at(frame_dir,min(end-0.01,(a+b)/2)).resolve()),
        "after":str(image_at(frame_dir,max(0,min(end-0.01,b+0.5))).resolve()),
    }
    return row


def writer_module():
    try:
        import xlsxwriter
    except ImportError as exc:
        raise RuntimeError("Please run: python -m pip install XlsxWriter") from exc
    return xlsxwriter


def xlsx_atomic(path, build):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.parent / ("._"+path.stem+"_writing.xlsx")
    if temp.exists(): temp.unlink()
    try:
        build(temp)
        os.replace(temp,path)
    except PermissionError as exc:
        raise RuntimeError(f"Close {path.name} in Excel and run again; data remains in cache") from exc
    finally:
        if temp.exists():
            try:temp.unlink()
            except OSError:pass


def formats(wb):
    return {
        "title":wb.add_format({"bold":True,"font_size":16,"font_color":"#FFFFFF","bg_color":"#172C45","valign":"vcenter"}),
        "subtitle":wb.add_format({"font_color":"#354E68","bg_color":"#E8F0F8","valign":"vcenter"}),
        "header":wb.add_format({"bold":True,"font_color":"#FFFFFF","bg_color":"#22699C","text_wrap":True,"valign":"vcenter","border":0}),
        "text":wb.add_format({"font_color":"#243E56","text_wrap":True,"valign":"top","bg_color":"#FFFFFF","border":0}),
        "stripe":wb.add_format({"font_color":"#243E56","text_wrap":True,"valign":"top","bg_color":"#F3F8FC","border":0}),
        "highlight":wb.add_format({"bold":True,"font_color":"#7C4F17","bg_color":"#FFF3DB","text_wrap":True,"valign":"vcenter"}),
        "input":wb.add_format({"font_color":"#115487","bg_color":"#EDF8FF","text_wrap":True}),
        "warn":wb.add_format({"font_color":"#A63341","bg_color":"#FDECEE","text_wrap":True}),
    }


def insert_picture(ws, r, c, path):
    from PIL import Image
    p=Path(path)
    if not p.is_file():return
    with Image.open(p) as im:
        w,h=im.size
    maxw,maxh=210,112
    scale=min(maxw/w,maxh/h,1.0)
    ws.insert_image(r,c,str(p), {"x_scale":scale,"y_scale":scale,
                               "x_offset":4,"y_offset":5,"object_position":1})


def main_columns(ws, fmt, header_row, rows, use_id=True):
    for col,w in enumerate(COL_WIDTHS):ws.set_column(col,col,w)
    ws.set_column(12,12,12,None,{"hidden":True})
    ws.set_column(13,13,30,None,{"hidden":True})  # provenance, not a visible column
    ws.set_row(header_row,37)
    for col,name in enumerate(HEADER):ws.write(header_row,col,name,fmt["header"])
    for i,item in enumerate(rows):
        row=header_row+i+1
        ws.set_row(row,95)
        values = [item.get("timestamp",""), item.get("action_type",""), item.get("action",""),
                  item.get("part",""), item.get("tool",""), item.get("tool_setting",""),
                  item.get("fixture",""), item.get("instructions",""), item.get("context","")]
        for c,value in enumerate(values):
            s=str(value or "")
            # Prevent Excel formula injection from any model text.
            if s.lstrip().startswith(("=","+","-","@")):s="'"+s
            ws.write_string(row,c,s,fmt["stripe"] if i%2 else fmt["text"])
        for c,key in ((9,"before"),(10,"after"),(11,"during")):
            insert_picture(ws,row,c,item.get(key,""))
        if use_id:
            ws.write_string(row,12,item.get("id",""))
            ws.write_string(row,13,item.get("video_source",""))
    ws.autofilter(header_row,0,header_row+len(rows),11)
    ws.freeze_panes(header_row+1,1)
    ws.set_landscape()
    ws.fit_to_pages(1,0)


def create_review_book(path, video, info, rows, audit):
    xl=writer_module()
    def build(temp):
        w=xl.Workbook(str(temp))
        try:
            fmt=formats(w)
            s=w.add_worksheet("Actions")
            s.hide_gridlines(2)
            s.merge_range("A1:L2","HYPERREALITY  /  REVIEW VIDEO ACTIONS",fmt["title"])
            s.set_row(0,26);s.set_row(1,22)
            s.merge_range("A3:L3",f"{video.name}  •  {mmss(info['duration'])}  •  Qwen suggestions only — not confirmed actions",fmt["subtitle"])
            s.merge_range("A4:L4","Edit text/timestamps here; mark YES on the Review sheet only for correct actions. Tool setting is blank by default.",fmt["highlight"])
            main_columns(s,fmt,4,rows)
            r=w.add_worksheet("Review")
            r.hide_gridlines(2);r.set_column("A:A",13);r.set_column("B:B",21)
            r.set_column("C:C",31);r.set_column("D:D",37);r.set_column("E:E",15);r.set_column("F:F",58)
            r.merge_range("A1:F2","HUMAN APPROVAL GATE  •  NO AUTO-APPROVAL",fmt["title"])
            r.merge_range("A3:F3","Only rows marked YES are eligible for the master workbook. Review visible frames, action boundaries and evidence.",fmt["subtitle"])
            labels=["Event ID","Candidate time","Proposed action","Model status","Approve?","Reviewer notes"]
            for col,name in enumerate(labels):r.write(4,col,name,fmt["header"])
            for ix,item in enumerate(rows,5):
                vals=[item["id"],item["timestamp"],item["action"],item["status"],"",""]
                r.set_row(ix,37)
                for c,v in enumerate(vals):r.write_string(ix,c,str(v),fmt["input"] if c>=4 else fmt["text"])
            if rows:r.data_validation(5,4,4+len(rows),4,{"validate":"list","source":["YES","NO"]})
            r.freeze_panes(5,2)
            # Optional human-created annotations for events Qwen missed or misparsed.
            manual=w.add_worksheet("Manual Additions")
            manual.hide_gridlines(2)
            manual.merge_range("A1:J2", "OPTIONAL  /  ADD MISSED ACTIONS", fmt["title"])
            manual.merge_range("A3:J3", "Add an action Qwen missed. Fill a timestamp and evidence, then type YES in Approve?; thumbnails are generated automatically.",fmt["subtitle"])
            manual_headers=["Timestamp","Action Type","Action","Part Name","Tool","Tool Setting", "Fixture","Instructions","Context / Evidence","Approve?"]
            manual_widths=[22,16,32,23,23,20,23,34,66,15]
            for col,(name,width) in enumerate(zip(manual_headers,manual_widths)):
                manual.set_column(col,col,width)
                manual.write(4,col,name,fmt["header"])
            for rnum in range(5,30):
                manual.set_row(rnum,36)
                for c in range(10):
                    manual.write_blank(rnum,c,None,fmt["input"])
            manual.data_validation(5,9,29,9,{"validate":"list","source":["YES","NO"]})
            manual.freeze_panes(5,2)
            a=w.add_worksheet("Raw Audit")
            a.set_column("A:A",19);a.set_column("B:B",17);a.set_column("C:C",16)
            a.set_column("D:D",55);a.set_column("E:E",90);a.set_column("F:F",18)
            for j,h in enumerate(["Timeframe","Status","Reason","Parsed suggestion","Exact Qwen response","Seconds"]):
                a.write(0,j,h,fmt["header"])
            for i,ent in enumerate(audit,1):
                for j,k in enumerate(("timeframe","status","reason","summary","raw","seconds")):
                    a.write(i,j,ent.get(k,""),fmt["stripe"] if i%2 else fmt["text"])
                a.set_row(i,48)
            a.freeze_panes(1,0)
            guide=w.add_worksheet("Instructions")
            guide.set_column("A:A",33);guide.set_column("B:B",88)
            instructions=[
                ("1. Inspect video", "Watch the actual video around the proposed timestamp and compare before/action/after pictures."),
                ("2. Correct", "In Actions, fix start/end times, verbs, part/tool/fixture, action, instruction and evidence. Leave unsupported fields blank."),
                ("3. Exclude false predictions", "In Review, leave Approve? blank or set NO when action is wrong or uncertain."),
                ("4. Approve correct rows", "In Review, type YES for each row you personally verified."),
                ("5. Close workbook", "Save changes, then close Excel before running approve command."),
                ("6. Commit to master", "Run python HyperReality_Final_Action_Review.py approve --video <path>. Enter YES at the terminal prompt."),
                ("Timestamp accuracy", "Qwen localization is an estimate from images sampled at 2 FPS, not a measured exact action boundary."),
                ("Tool setting", "Always blank initially. Fill manually only when clearly verified (e.g. No. 8 spanner)."),
                ("Rejected/empty fields", "These do not enter the master. The Raw Audit sheet preserves Qwen's original output."),
                ("Missed actions", "Use the Manual Additions sheet for actions Qwen did not report. Fill timestamp, keyword, object, evidence and YES. Other fields optional."),
                ("Repeated actions", "If the same continuous action appears in adjacent rows, combine/fix their timestamp manually and approve only the correct consolidated row."),
            ]
            for ix,(a_txt,b_txt) in enumerate(instructions):
                guide.write(ix,0,a_txt,fmt["header"]);guide.write(ix,1,b_txt,fmt["text"])
                guide.set_row(ix,44)
        finally:
            w.close()
    xlsx_atomic(path,build)


def xlsx_values(path, sheet_name):
    """Read edited xlsx cells using Python's standard library (no openpyxl)."""
    ns={"x":"http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r":"http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "p":"http://schemas.openxmlformats.org/package/2006/relationships"}
    with zipfile.ZipFile(path) as z:
        wb=ET.fromstring(z.read("xl/workbook.xml"))
        rels=ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        targets={rel.get("Id"):rel.get("Target") for rel in rels}
        sheets=wb.find("x:sheets",ns)
        match=next((s for s in sheets if s.get("name")==sheet_name),None)
        if match is None:raise ValueError(f"No '{sheet_name}' worksheet in {path.name}")
        rid=match.get("{"+ns["r"]+"}id")
        target=targets.get(rid,"")
        src=target.lstrip("/") if target.startswith("/") else "xl/"+target
        src=re.sub(r"(^|/)\.\./", "", src).replace("xl/xl/","xl/")
        root=ET.fromstring(z.read(src))
        strings=[]
        if "xl/sharedStrings.xml" in z.namelist():
            shared=ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in shared.findall("x:si",ns):
                strings.append("".join(x.text or "" for x in si.iter("{"+ns["x"]+"}t")))
        out={}
        for r in root.findall(".//x:sheetData/x:row",ns):
            number=int(r.get("r")); cells={}
            for cell in r.findall("x:c",ns):
                ref=cell.get("r","");col=re.match(r"[A-Z]+",ref)
                if not col:continue
                c=cell.find("x:v",ns)
                inline=cell.find("x:is",ns)
                val=c.text if c is not None and c.text is not None else ""
                if cell.get("t")=="s" and val:
                    try:val=strings[int(val)]
                    except (IndexError, ValueError):val=""
                if inline is not None:
                    val="".join(t.text or "" for t in inline.iter("{"+ns["x"]+"}t"))
                cells[col.group()]=val
            out[number]=cells
    return out


def read_review_entries(review_file, original):
    actions=xlsx_values(review_file,"Actions")
    review=xlsx_values(review_file,"Review")
    # All event IDs come from the original cache, not arbitrary injected row names.
    allowed={r["id"]:r for r in original}
    edits={}
    for num, cells in actions.items():
        if num<=5:continue
        eid=cells.get("M","").strip()
        if not eid or eid not in allowed:continue
        # 12 visible fields; image columns JKL are generated from original caches.
        edits[eid]={
            "timestamp":cells.get("A",""),"action_type":cells.get("B","").strip().lower(),
            "action":cells.get("C",""),"part":cells.get("D",""),
            "tool":cells.get("E",""),"tool_setting":cells.get("F",""),
            "fixture":cells.get("G",""),"instructions":cells.get("H",""),
            "context":cells.get("I",""),"before":allowed[eid]["before"],
            "during":allowed[eid]["during"],"after":allowed[eid]["after"],
            "id":eid,"video":allowed[eid].get("video","")}
    approvals={}
    for num, cells in review.items():
        if num<=5:continue
        eid=cells.get("A","").strip()
        if eid in allowed:
            approvals[eid]=cells.get("E","").strip().upper()
    accepted=[]
    for eid,decision in approvals.items():
        if decision!="YES":continue
        if eid not in edits:raise ValueError(f"Missing edited event {eid} in Actions sheet")
        row=edits[eid]
        if row["action_type"] not in VERBS:
            raise ValueError(f"{eid}: invalid Action Type '{row['action_type']}'. Use a keyword like remove/place/lift")
        if not all(str(row[k]).strip() for k in ("timestamp","action","part","instructions","context")):
            raise ValueError(f"{eid}: approved row has missing required fields. Reject or complete it")
        a,b=parse_timeframe(row["timestamp"])
        if a<0 or b>original[0].get("video_duration",math.inf)+2:
            raise ValueError(f"{eid}: timestamp exceeds video duration")
        if len(row["context"].strip().split())<5:
            raise ValueError(f"{eid}: context needs specific visible evidence")
        row["timestamp"]=interval(a,b)
        row["start"],row["end"]=a,b
        accepted.append(row)
    accepted.sort(key=lambda r:(r["start"],r["end"]))
    return accepted,len(approvals)


def read_manual_additions(review_file, source, folder):
    """Read optional human-added rows. Never fabricate an action missing from Qwen."""
    try:
        sheet=xlsx_values(review_file,"Manual Additions")
    except ValueError:
        return []
    frames=folder/"frames"
    duration=float(source["video_meta"]["duration"])
    manual=[]
    for sheet_row,cell in sheet.items():
        if sheet_row<=5 or cell.get("J","").strip().upper()!="YES":
            continue
        stamp=cell.get("A","")
        a,b=parse_timeframe(stamp)
        if a < 0 or b > duration+2:
            raise ValueError(f"Manual Additions row {sheet_row}: timestamp beyond video duration")
        verb=cell.get("B","").strip().lower()
        part=cell.get("D","").strip()
        evidence=cell.get("I","").strip()
        if verb not in VERBS or not part or len(evidence.split())<5:
            raise ValueError(f"Manual Additions row {sheet_row}: requires valid action keyword, part name, and specific evidence")
        obj=object_phrase(part)
        manual.append({
            "id":f"m{sheet_row:04d}","timestamp":interval(a,b),"start":a,"end":b,
            "action_type":verb,"part":part,"action":cell.get("C","").strip() or f"{past_tense(verb).capitalize()} {obj}",
            "tool":cell.get("E","").strip(),"tool_setting":cell.get("F","").strip(),
            "fixture":cell.get("G","").strip(),"instructions":cell.get("H","").strip() or f"{verb.capitalize()} {obj}",
            "context":evidence,"before":str(image_at(frames,max(0,a-0.5)).resolve()),
            "during":str(image_at(frames,(a+b)/2).resolve()),
            "after":str(image_at(frames,min(duration-0.01,b+0.5)).resolve()),
            "source":"human-added",
        })
    return manual


def save_master_db(db):
    atomic_json(MASTER_DB,db)


def create_master_book(path,db):
    xl=writer_module()
    all_rows=[]
    for v in db.get("videos",{}).values():
        all_rows.extend(v.get("approved",[]))
    all_rows.sort(key=lambda r:(r.get("approved_order",0),r.get("start",0)))
    def build(temp):
        w=xl.Workbook(str(temp))
        try:
            fmt=formats(w)
            s=w.add_worksheet("Master Actions")
            s.hide_gridlines(2)
            s.merge_range("A1:L2","HYPERREALITY  /  APPROVED ACTION ANNOTATIONS",fmt["title"])
            s.merge_range("A3:L3",f"{len(db.get('videos',{}))} reviewed videos  •  {len(all_rows)} approved actions  •  Human reviewed (not automatically model verified)",fmt["subtitle"])
            s.merge_range("A4:L4","Each row was explicitly selected for inclusion during human review. Thumbnails are supporting references; verify action boundaries against source videos.",fmt["highlight"])
            main_columns(s,fmt,4,all_rows)
            m=w.add_worksheet("Video Index")
            m.set_column("A:A",42);m.set_column("B:B",19);m.set_column("C:C",17);m.set_column("D:D",30)
            m.set_column("E:F",17)
            for j,header in enumerate(["Video","Approved actions","Reviewed on","Video file path","First row","Last row"]):
                m.write(0,j,header,fmt["header"])
            next_master_row=6  # First data row in Master Actions, Excel 1-based
            indexed_videos=sorted(db.get("videos",{}).values(),key=lambda x:x.get("order",0))
            for i,v in enumerate(indexed_videos,1):
                count=len(v["approved"])
                start_row=next_master_row
                end_row=next_master_row+count-1
                for j,cell in enumerate([v["name"],count,v.get("reviewed_at",""),v["path"],start_row,end_row]):
                    m.write(i,j,cell,fmt["stripe"] if i%2 else fmt["text"])
                m.set_row(i,33)
                next_master_row+=count
            m.freeze_panes(1,0)
            guide=w.add_worksheet("About")
            guide.set_column("A:A",32);guide.set_column("B:B",100)
            notes=[
                ("Source", "Approved model proposals and human-added actions from per-video review workbooks; the JSON approval database is authoritative."),
                ("Timestamp", "Video-relative approximate action interval, manually verified or corrected before approval."),
                ("Evidence", "Only direct visual observations should be retained; unsupported/ambiguous tool or fixture fields must remain empty."),
                ("Tool Setting", "Never auto-filled. Enter manually only with clear supporting evidence."),
                ("Three thumbnails", "Before, After, and Action (representative moment). These are samples, not automatic proof of a state change."),
                ("Updating a video", "Edit its per-video review workbook, then approve again; approved rows for that video replace its older approved rows without duplicates."),
                ("Do not edit master", "Edit review workbook and reapprove instead. The master workbook is regenerated from approved_master.json."),
            ]
            for ix,(a,b) in enumerate(notes):
                guide.write(ix,0,a,fmt["header"]);guide.write(ix,1,b,fmt["text"]);guide.set_row(ix,44)
        finally:
            w.close()
    xlsx_atomic(path,build)
    return len(all_rows)


def analyze(video,root=OUTPUT_ROOT,model_runner=None,limit_windows=None):
    video=Path(video).expanduser().resolve()
    if not video.is_file():raise FileNotFoundError(f"Video not found: {video}")
    check_environment(needs_model=model_runner is None)
    info=video_meta(video)
    folder=folder_for(video,root)
    folder.mkdir(parents=True,exist_ok=True)
    review_file=folder/("Review_"+safe_filename(video.stem)+".xlsx")
    if review_file.exists():
        print("Review workbook already exists; preserved your edits:",review_file)
        return review_file
    frames=prepare_images(video,info,folder)
    raw_dir=folder/"cache";raw_dir.mkdir(exist_ok=True)
    all_windows=windows(info["duration"])
    if limit_windows is not None:all_windows=all_windows[:limit_windows]
    print(f"{video.name}: {len(all_windows)} 5-second windows (blind analysis)")
    model=processor=None
    rows=[]; audit=[]
    for idx,(start,end) in enumerate(all_windows):
        cache=raw_dir/f"window_{idx+1:04d}.json"
        saved=read_json(cache)
        if saved and saved.get("version")==VERSION and saved.get("video_id")==video_identity(video):
            raw=saved["raw"];secs=float(saved.get("seconds",0));count=int(saved.get("frame_count",10))
            reused=True
        else:
            imgs=load_image_section(frames,start,end)
            count=len(imgs)
            if model_runner is None:
                if model is None:
                    print("Loading locally cached Qwen3-VL-2B...",flush=True)
                    model,processor=load_qwen()
                begin=time.perf_counter()
                try:raw,_=infer_once(model,processor,imgs)
                except Exception as exc:
                    raw=f"[INFERENCE_ERROR] {type(exc).__name__}: {exc}"
                secs=round(time.perf_counter()-begin,2)
            else:
                begin=time.perf_counter();raw=model_runner(idx,start,end,imgs)
                secs=round(time.perf_counter()-begin,2)
            atomic_json(cache,{"version":VERSION,"video_id":video_identity(video),
                               "window":idx,"start":start,"end":end,
                               "raw":raw,"seconds":secs,"frame_count":count})
            reused=False
        parsed,status,note=parse_one(raw,count)
        one=candidate_record(video,frames,idx,start,end,parsed,status,note,secs,raw)
        if one:
            one["video_duration"]=info["duration"]
            rows.append(one)
        audit.append({"timeframe":interval(start,end),"status":status,"reason":note,
                      "summary":one["action"] if one else "", "raw":raw,"seconds":secs})
        print(f"{interval(start,end)} | {status} | {secs:.2f}s" + (" | cached" if reused else ""),flush=True)
    atomic_json(folder/"review_source.json",{
        "video":str(video),"video_id":video_identity(video),"video_meta":info,
        "version":VERSION,"actions":rows,"audit":audit,"created_at":now_iso()
    })
    create_review_book(review_file,video,info,rows,audit)
    print("\nREVIEW WORKBOOK:",review_file)
    print("Candidates:",len(rows),"| Other windows retained in Raw Audit")
    print("No records have been added to the master workbook.")
    return review_file


def approve(video,root=OUTPUT_ROOT,interactive=True):
    global OUTPUT_ROOT,MASTER_DB,MASTER_FILE
    if Path(root)!=OUTPUT_ROOT:
        OUTPUT_ROOT=Path(root); MASTER_DB=OUTPUT_ROOT/"approved_master.json"
        MASTER_FILE=OUTPUT_ROOT/"HyperReality_Master_Approved.xlsx"
    video=Path(video).expanduser().resolve()
    folder=folder_for(video,root)
    source=read_json(folder/"review_source.json")
    if not source:
        raise RuntimeError("No review source found. Run analyze on this video first")
    review=folder/("Review_"+safe_filename(video.stem)+".xlsx")
    if not review.exists():raise RuntimeError("Review workbook not found")
    accepted,reviewed=read_review_entries(review,source["actions"])
    manually_added=read_manual_additions(review,source,folder)
    accepted.extend(manually_added)
    accepted.sort(key=lambda r:(r["start"],r["end"]))
    # Refresh all thumbnails from the FINAL reviewer-edited timestamp, not the
    # model's original 5-second proposal. This is essential if a person extends
    # 0:07–0:10 to 0:07–0:12 after inspecting the actual video.
    duration=float(source["video_meta"]["duration"])
    cached_frames=folder/"frames"
    for row in accepted:
        a,b=row["start"],row["end"]
        row["before"]=str(image_at(cached_frames,max(0,a-0.5)).resolve())
        row["during"]=str(image_at(cached_frames,(a+b)/2).resolve())
        row["after"]=str(image_at(cached_frames,min(duration-0.01,b+0.5)).resolve())
    print(f"Selected {len(accepted)} actions ({len(manually_added)} manually added) from {video.name}")
    if len(accepted)==0:
        print("Nothing will be added. Mark YES on Review or Manual Additions for verified actions, save Excel, and retry.")
        return 0
    if interactive:
        choice=input("Type YES to replace this video's master entries with the selected rows: ").strip()
        if choice!="YES":
            print("Cancelled; master unchanged.")
            return 0
    db=read_json(MASTER_DB,{"version":1,"videos":{}})
    old=db.get("videos",{}).get(video_identity(video),{})
    order=old.get("order",len(db.get("videos",{}))+1)
    for item in accepted:
        item["approved_order"]=order
        item["video_source"]=str(video)
    db["videos"][video_identity(video)]={
        "name":video.name,"path":str(video),"order":order,
        "approved":accepted,"reviewed_at":now_iso()
    }
    # DB saved first, so approval survives even if Excel happens to be open.
    atomic_json(MASTER_DB,db)
    count=create_master_book(MASTER_FILE,db)
    print(f"MASTER UPDATED: {MASTER_FILE} | {count} total approved actions")
    return len(accepted)


def rebuild_master(root=OUTPUT_ROOT):
    db=read_json(Path(root)/"approved_master.json",{"version":1,"videos":{}})
    outfile=Path(root)/"HyperReality_Master_Approved.xlsx"
    return create_master_book(outfile,db)


def status(root=OUTPUT_ROOT):
    db=read_json(Path(root)/"approved_master.json",{"videos":{}})
    if not db.get("videos"):
        print("Master has no approved videos yet.")
        return
    print("APPROVED MASTER")
    for v in db["videos"].values():
        print(f" - {v['name']}: {len(v['approved'])} approved actions")
    print("Master:",Path(root)/"HyperReality_Master_Approved.xlsx")


def self_test(video=None):
    tests=[
        ("NONE",10,"NO ACTION"),
        ("remove|wheel|wrench||2|9|The wheel separates from the visible hub.",10,"CANDIDATE"),
        ("remove|wheel|wrench||7|12|Visible action of a wheel.",10,"REVIEW"),
        ("main object|state before|high||2|7|This is not a real observation.",10,"REVIEW"),
        ("open|tool drawer|||1|8|The drawer moves outward from the trolley.",10,"CANDIDATE"),
        ("remove|wheel|wrench||2|9|high",10,"REVIEW"),
    ]
    for raw,count,expected in tests:
        _,status,_=parse_one(raw,count)
        assert status==expected,(raw,status,expected)
    assert parse_timeframe("0:07 - 0:12")== (7,12)
    assert interval(2,2.5) == "0:02 - 0:03"
    assert past_tense("remove")=="removed"
    assert object_phrase("wheel")=="the wheel"
    assert "tool setting" not in PROMPT.lower() # never ask Qwen to infer size
    print("Parser, timestamp, grammar and no-tool-setting tests PASS")
    if video:
        class FakeRunner:
            def __call__(self,idx,start,end,images):
                if idx==0:
                    return "handle|metal tool|||1|7|The technician moves a metal tool beside the trolley."
                if idx==1:
                    return "open|drawer|||1|9|The drawer visibly moves outward from the cabinet."
                return "NONE"
        root=Path(tempfile.mkdtemp(prefix="hr_test_"))
        try:
            report=analyze(Path(video),root=root,model_runner=FakeRunner())
            assert report.exists()
            s=read_json(report.parent/"review_source.json")
            assert len(s["audit"])==len(windows(video_meta(Path(video))["duration"]))
            assert len(s["actions"])==2
            assert len(list((report.parent/"frames").glob("f*.jpg")))>10
            # Produce and read back fully populated XLSX, then test approval-gate exclusion.
            cells=xlsx_values(report,"Actions")
            assert any(v.get("M")=="e0001" for v in cells.values()),"Event IDs not preserved"
            accepted,_=read_review_entries(report,s["actions"])
            assert len(accepted)==0,"Unapproved rows must never pass"
            print(f"FULL VIDEO DECODE AND XLSX TEST PASS: {len(s['audit'])} intervals, 2 synthetic candidates")
            print("No Qwen inference was run in self-test.")
        finally:
            shutil.rmtree(root,ignore_errors=True)


def main():
    p=argparse.ArgumentParser(description="HyperReality blind Qwen action annotations with human approval")
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("analyze",help="Analyze one new video and create its review workbook")
    a.add_argument("--video",type=Path,required=True)
    a=sub.add_parser("approve",help="Approve selected rows from a reviewed workbook")
    a.add_argument("--video",type=Path,required=True)
    sub.add_parser("status",help="List approved videos/actions")
    sub.add_parser("rebuild-master",help="Regenerate master XLSX from approved database, no AI")
    t=sub.add_parser("self-test",help="Test without loading Qwen; optionally decode a real video")
    t.add_argument("--video",type=Path)
    args=p.parse_args()
    try:
        if args.command=="self-test":self_test(args.video)
        elif args.command=="analyze":analyze(args.video)
        elif args.command=="approve":approve(args.video)
        elif args.command=="status":status()
        elif args.command=="rebuild-master":
            print("Rebuilt",rebuild_master(),"approved actions; no Qwen inference")
    except (RuntimeError,ValueError,FileNotFoundError,PermissionError,zipfile.BadZipFile) as exc:
        print("ERROR:",exc,file=sys.stderr)
        return 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
