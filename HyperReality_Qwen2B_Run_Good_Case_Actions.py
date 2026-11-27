r"""HyperReality — Qwen3-VL-2B Good Case batch runner; original inference untouched.

Place beside the UNMODIFIED HyperReality_Final_Action_Review.py.
Use the original core's predicted ACTION-START / ACTION-END, NOT the internal 5-sec window.

PowerShell from the HyperReality project directory:
  python .\HyperReality_Qwen2B_Run_Good_Case_Actions.py --self-test
  python .\HyperReality_Qwen2B_Run_Good_Case_Actions.py

Default video folder: videos/Good Case (all nested videos included).
Outputs: one workbook for all videos and another per video. 'All Actions' shows
ONLY Qwen's action proposals, with original per-action timestamps. 'Raw Audit'
keeps every internal window: candidates, NONE, malformed output, inference errors.
No YES approval, no master changes, no prompt/model/decoder/parser changes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CORE_NAME = "HyperReality_Final_Action_Review.py"
EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".wmv", ".webm"}

# Original action export columns unchanged, with four read-only audit fields appended.
ACTION_COLUMNS = [
    ("Timestamp", "timestamp"), ("Action Type", "action_type"),
    ("Action", "action"), ("Part Name", "part"), ("Tool", "tool"),
    ("Tool Setting", "tool_setting"), ("Fixture", "fixture"),
    ("Instructions", "instructions"), ("Context / Evidence", "context"),
    ("Before", "before"), ("After", "after"), ("Action Image", "during"),
    ("Qwen Status", "status"), ("Exact Raw Qwen Response", "raw"),
    ("Processing (s)", "seconds"), ("Review / Error Note", "note")
]
RAW_COLUMNS = [
    ("Analysis Window (internal)", "timeframe"), ("Qwen Status", "status"),
    ("Parsed Suggestion", "summary"), ("Exact Raw Qwen Response", "raw"),
    ("Processing (s)", "seconds"), ("Parser / Error Note", "reason"),
]


def original():
    src = SCRIPT_DIR / CORE_NAME
    if not src.is_file():
        raise FileNotFoundError(f"Missing original {CORE_NAME} beside the batch runner")
    spec = importlib.util.spec_from_file_location("hyperreality_2b_UNCHANGED", src)
    if not spec or not spec.loader:
        raise RuntimeError("Could not import original Qwen 2B script")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if mod.MODEL != "Qwen/Qwen3-VL-2B-Instruct" or mod.VERSION != "10.0-review-first":
        raise RuntimeError("Wrong original script version/model; refusing to substitute logic")
    return mod


def sort_key(path: Path):
    return [(1, int(x)) if x.isdigit() else (0, x.casefold()) for x in re.split(r"(\d+)", str(path))]


def discover(folder: Path):
    if not folder.is_dir():
        raise FileNotFoundError(f"Video folder not found: {folder}")
    videos = [p.resolve() for p in folder.rglob("*")
              if p.is_file() and p.suffix.lower() in EXTENSIONS]
    return sorted(videos, key=lambda p: sort_key(p.relative_to(folder)))


def read_one(core, video: Path, do_infer: bool, review_root: Path):
    """Return original per-action candidates plus the complete unfiltered raw audit."""
    folder = core.folder_for(video, root=review_root)
    src_path = folder / "review_source.json"
    saved = core.read_json(src_path) if src_path.exists() else None
    expected = len(core.windows(core.video_meta(video)["duration"]))
    valid = (saved is not None and saved.get("version") == core.VERSION
             and saved.get("video_id") == core.video_identity(video)
             and len(saved.get("audit", [])) == expected)
    if not valid:
        if not do_infer:
            raise RuntimeError("No complete saved original Qwen result for --rebuild-only")
        review = folder / ("Review_" + core.safe_filename(video.stem) + ".xlsx")
        if review.exists():
            raise RuntimeError(
                f"Existing review file without complete matching source; preserving {review}"
            )
        # No modifications to original model, prompt, sampling, inference, parser, timestamps.
        core.analyze(video, root=review_root)
        saved = core.read_json(src_path)
        if not saved or len(saved.get("audit", [])) != expected:
            raise RuntimeError("Original action analysis did not complete all windows")
    else:
        print("  Reusing original Qwen 2B review cache", flush=True)

    audit = [dict(x) for x in saved["audit"]]
    actions = []
    for item in saved.get("actions", []):
        # Event id e0001 indicates which original 5-second inference generated it.
        n = int(item["id"][1:]) - 1
        log = audit[n] if 0 <= n < len(audit) else {}
        result = {key: item.get(key, "") for _, key in ACTION_COLUMNS}
        # KEY RULE: never replace the action's own timestamp with the 5-second audit range.
        result["timestamp"] = item.get("timestamp", "")
        result["status"] = log.get("status", "CANDIDATE")
        result["raw"] = log.get("raw", item.get("raw", ""))
        result["seconds"] = log.get("seconds", item.get("processing_seconds", ""))
        result["note"] = log.get("reason", item.get("status", ""))
        actions.append(result)
    return actions, audit


def make_excel(entries, target: Path, thumbnails=True):
    import xlsxwriter
    from PIL import Image

    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name("._" + target.stem + "_writing.xlsx")
    if temp.exists():
        temp.unlink()
    wb = xlsxwriter.Workbook(str(temp), {"strings_to_formulas": False, "strings_to_urls": False})
    navy, blue = "#142B46", "#246E9F"
    title = wb.add_format({"font_size":16, "bold":True,"font_color":"#FFFFFF","bg_color":navy,"valign":"vcenter"})
    section = wb.add_format({"font_size":11,"bold":True,"font_color":"#FFFFFF","bg_color":navy,"valign":"vcenter"})
    header = wb.add_format({"font_color":"#FFFFFF","bg_color":blue,"bold":True,"text_wrap":True,"valign":"vcenter"})
    info = wb.add_format({"bg_color":"#E8F0F7","font_color":"#35536B","text_wrap":True,"valign":"vcenter"})
    even = wb.add_format({"font_color":"#263E54","font_size":10,"text_wrap":True,"valign":"top"})
    odd = wb.add_format({"font_color":"#263E54","font_size":10,"bg_color":"#F3F7FA","text_wrap":True,"valign":"top"})
    muted = wb.add_format({"font_color":"#566C80","font_size":10,"bg_color":"#E9EFF4","text_wrap":True})
    link = wb.add_format({"font_color":blue,"underline":1})
    warn = wb.add_format({"font_color":"#985418","bg_color":"#FFF2DB","text_wrap":True})
    idx = wb.add_worksheet("File Index")
    acts = wb.add_worksheet("All Actions")
    audit_ws = wb.add_worksheet("Raw Audit")
    idx.hide_gridlines(2); acts.hide_gridlines(2); audit_ws.hide_gridlines(2)

    idx.set_column("A:A",8); idx.set_column("B:B",53); idx.set_column("C:F",18)
    idx.set_column("G:G",72); idx.set_column("H:I",24)
    idx.merge_range("A1:I2", "HYPERREALITY  |  QWEN 2B GOOD CASE", title)
    idx.merge_range("A3:I3", "Per-action timestamps in All Actions; every internal window response kept in Raw Audit. Neither is verified ground truth.", info)
    idx_heads = ["#", "Filename / subfolder", "Action proposals", "Raw windows", "NONE", "Review / errors", "Source video", "Go to actions", "Go to raw"]
    for c, name in enumerate(idx_heads): idx.write(4, c, name, header)
    idx.freeze_panes(5, 2)

    widths = [22,18,32,24,22,20,22,34,60,23,23,23,18,80,17,42]
    for c,w in enumerate(widths): acts.set_column(c,c,w)
    acts.merge_range(0,0,1,15,"HYPERREALITY  |  DETECTED ACTIONS — ACTUAL PROPOSED TIMES",title)
    acts.merge_range(2,0,2,15,"Each row = one Qwen action proposal, with its own estimated start/end. The model's 5-second analysis windows are NOT substituted as timestamps.",info)
    acts.freeze_panes(4,1)

    raw_widths = [26,17,35,90,18,70]
    for c,w in enumerate(raw_widths): audit_ws.set_column(c,c,w)
    audit_ws.merge_range(0,0,1,5,"HYPERREALITY  |  COMPLETE RAW MODEL AUDIT",title)
    audit_ws.merge_range(2,0,2,5,"Internal analysis windows ONLY for traceability. Includes NONE, rejected responses, and inference errors. No actions invented.",info)
    audit_ws.freeze_panes(4,1)
    action_cursor, raw_cursor = 4, 4
    for n, entry in enumerate(entries, 1):
        name = entry["label"]
        actions = entry["actions"]
        raw = entry["audit"]
        action_link = action_cursor+1
        raw_link = raw_cursor+1
        acts.merge_range(action_cursor,0,action_cursor,15, f"VIDEO {n:02d}  |  {name}", section)
        acts.set_row(action_cursor,31)
        audit_ws.merge_range(raw_cursor,0,raw_cursor,5,f"VIDEO {n:02d}  |  {name}",section)
        audit_ws.set_row(raw_cursor,31)
        for c,(label, _) in enumerate(ACTION_COLUMNS):acts.write(action_cursor+1,c,label,header)
        acts.set_row(action_cursor+1,38)
        for c,(label, _) in enumerate(RAW_COLUMNS):audit_ws.write(raw_cursor+1,c,label,header)
        audit_ws.set_row(raw_cursor+1,35)
        for j, item in enumerate(actions):
            row = action_cursor+2+j
            acts.set_row(row,94 if thumbnails else 56)
            fmt = odd if j%2 else even
            for c, (_, key) in enumerate(ACTION_COLUMNS):
                if key in {"before","after","during"}:
                    continue
                value=item.get(key, "")
                if c==14 and isinstance(value,(int,float)):
                    acts.write_number(row,c,float(value),fmt)
                else:
                    s=str(value if value is not None else "")
                    if len(s)>32000:s=s[:31980]+" [TRUNCATED]"
                    acts.write_string(row,c,s,fmt)
            if thumbnails:
                for col,key in ((9,"before"),(10,"after"),(11,"during")):
                    pic=Path(str(item.get(key) or ""))
                    if not pic.is_file(): continue
                    try:
                        with Image.open(pic) as im: width,height=im.size
                        scale=min(148/max(1,width),105/max(1,height),1.0)
                        acts.insert_image(row,col,str(pic),{"x_scale":scale,"y_scale":scale,
                                         "x_offset":3,"y_offset":2,"object_position":1})
                    except (OSError,ValueError):
                        pass
        if not actions:
            acts.merge_range(action_cursor+2,0,action_cursor+2,15,
                "No parsed action proposal from this video. Every Qwen output is still available on Raw Audit.",muted)
        for j, item in enumerate(raw):
            row=raw_cursor+2+j
            audit_ws.set_row(row,60)
            fmt=odd if j%2 else even
            for col, (_, key) in enumerate(RAW_COLUMNS):
                value=item.get(key, "")
                if col==4 and isinstance(value,(int,float)):
                    audit_ws.write_number(row,col,float(value),fmt)
                else:
                    s=str(value if value is not None else "")
                    if len(s)>32000:s=s[:31980]+" [TRUNCATED]"
                    f=warn if col==1 and str(item.get("status","")) in {"REVIEW","FILE ERROR"} else fmt
                    audit_ws.write_string(row,col,s,f)
        if not raw:
            audit_ws.merge_range(raw_cursor+2,0,raw_cursor+2,5,"No raw model responses saved for this file.",muted)
        indexrow=n+4
        idx.write_number(indexrow,0,n)
        idx.write_url(indexrow,1,f"internal:'All Actions'!A{action_link}",link,name)
        idx.write_number(indexrow,2,len(actions))
        idx.write_number(indexrow,3,len(raw))
        idx.write_number(indexrow,4,sum(str(x.get("status",""))=="NO ACTION" for x in raw))
        idx.write_number(indexrow,5,sum(str(x.get("status","")) in {"REVIEW","FILE ERROR"} for x in raw))
        idx.write_string(indexrow,6,str(entry["video"]),even)
        idx.write_url(indexrow,7,f"internal:'All Actions'!A{action_link}",link,"Action timeline")
        idx.write_url(indexrow,8,f"internal:'Raw Audit'!A{raw_link}",link,"All raw responses")
        action_cursor+=max(len(actions),1)+4
        raw_cursor+=max(len(raw),1)+4
    if entries:idx.autofilter(4,0,4+len(entries),8)
    wb.close()
    try:temp.replace(target)
    except PermissionError as exc:raise PermissionError(f"Close {target.name} in Excel, then rerun") from exc
    return target


def self_test(core):
    core.self_test()
    test_data=[
        {"video":Path("Good Case/IMG_01.MOV"), "label":"IMG_01.MOV", "actions":[
            {"timestamp":"0:05 - 0:08","action_type":"move","action":"Moved the trolley",
             "part":"trolley","status":"CANDIDATE","raw":"move|trolley|||0|6|Trolley moved beside mechanic."}],
         "audit":[{"timeframe":"0:05 - 0:10","status":"CANDIDATE","raw":"move|trolley|||0|6|Trolley moved beside mechanic."},
                  {"timeframe":"0:10 - 0:15","status":"NO ACTION","raw":"NONE"},
                  {"timeframe":"0:15 - 0:20","status":"REVIEW","raw":"invalid|format","reason":"invalid format"}]},
        {"video":Path("Good Case/IMG_02.MOV"),"label":"IMG_02.MOV","actions":[],
         "audit":[{"status":"FILE ERROR","raw":"[FILE_ERROR] simulated"}]},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        output=Path(tmp)/"combined.xlsx"
        make_excel(test_data,output,thumbnails=False)
        per=Path(tmp)/"individual.xlsx"
        make_excel([test_data[0]],per,thumbnails=False)
        with zipfile.ZipFile(output) as z:
            assert z.testzip() is None
            xml=" ".join(z.read(x).decode("utf-8") for x in z.namelist() if x.startswith("xl/worksheets/sheet") and x.endswith(".xml"))
            strings=z.read("xl/sharedStrings.xml").decode("utf-8")
            assert "0:05 - 0:08" in strings and "0:05 - 0:10" in strings
            assert "Moved the trolley" in strings and "NONE" in strings and "invalid|format" in strings
            assert "IMG_01.MOV" in strings and "IMG_02.MOV" in strings
            assert per.is_file()
    print("PASS: unmodified original parser, true per-action timestamps, filename sections, raw NONE/rejected/error, combined + per-video Excel")
    print("Software-only self-test; live Qwen inference must run locally")


def main():
    p=argparse.ArgumentParser(description="Qwen3-VL-2B — every Good Case video, action timestamps in one Excel and individual Excel")
    p.add_argument("--videos-dir",type=Path,default=Path("videos")/"Good Case",help="Run all videos under this directory recursively (default: videos/Good Case)")
    p.add_argument("--rebuild-only",action="store_true",help="Export existing original Qwen results without loading model")
    p.add_argument("--no-thumbnails",action="store_true",help="Skip thumbnail insertion in batch exports")
    p.add_argument("--self-test",action="store_true",help="Test parsing and Excel output without running Qwen")
    args=p.parse_args()
    os.chdir(SCRIPT_DIR)
    try:
        core=original()
        if args.self_test:
            self_test(core); return 0
        directory=args.videos_dir.expanduser().resolve()
        video_paths=discover(directory)
        if not video_paths:
            raise RuntimeError(f"No video files found under {directory}")
        output=SCRIPT_DIR/"outputs"/"HyperReality_Qwen2B_Good_Case_Actions"
        individual=output/"Per_Video"
        combined=output/"HyperReality_Good_Case_All_Actions.xlsx"
        original_root=SCRIPT_DIR/"outputs"/"HyperReality_Final"
        # Memory optimization ONLY: reuse exactly the same original model instance.
        original_loader=core.load_qwen
        loaded=[None]
        def load_once():
            if loaded[0] is None: loaded[0]=original_loader()
            return loaded[0]
        core.load_qwen=load_once
        entries=[]
        print(f"FOUND {len(video_paths)} VIDEO(S) IN {directory}",flush=True)
        print("QWEN3-VL-2B ORIGINAL MODEL, PROMPT, ACTION LOCALIZATION, PARSER: UNCHANGED",flush=True)
        print("EXCEL ALL ACTIONS = QWEN'S SHORT ACTION TIMESTAMPS; RAW AUDIT = INTERNAL WINDOWS",flush=True)
        for i,video in enumerate(video_paths,1):
            label=video.relative_to(directory).as_posix()
            print(f"\n[{i}/{len(video_paths)}] {label}",flush=True)
            try:
                actions,audit=read_one(core,video,not args.rebuild_only,original_root)
            except Exception as ex:
                detail=f"{type(ex).__name__}: {ex}"
                print("WARNING:",detail,flush=True)
                actions=[];audit=[{"timeframe":"","status":"FILE ERROR", "summary":"",
                                   "raw":"[FILE_ERROR] "+detail,"seconds":"","reason":detail}]
            entry={"video":video,"label":label,"actions":actions,"audit":audit}
            entries.append(entry)
            short=hashlib.sha256(str(video).encode()).hexdigest()[:8]
            name=core.safe_filename(video.stem)
            file=individual/f"Actions_{name}_{short}.xlsx"
            make_excel([entry],file,thumbnails=not args.no_thumbnails)
            make_excel(entries,combined,thumbnails=not args.no_thumbnails)
            print(f"  ACTION PROPOSALS: {len(actions)} | RAW OUTPUTS: {len(audit)}",flush=True)
            print("  PER-VIDEO:",file,flush=True)
            print("  COMBINED:",combined,flush=True)
        print(f"\nDONE: {len(entries)} video(s), {sum(len(e['actions']) for e in entries)} action proposals, "
              f"{sum(len(e['audit']) for e in entries)} original raw responses",flush=True)
        print("Master approvals untouched. Qwen proposals NOT independently verified.",flush=True)
        return 0
    except (RuntimeError,ValueError,FileNotFoundError,PermissionError) as ex:
        print("ERROR:",ex,file=sys.stderr);return 1


if __name__=="__main__":
    raise SystemExit(main())
