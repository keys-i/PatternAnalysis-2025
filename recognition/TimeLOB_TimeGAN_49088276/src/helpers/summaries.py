from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd
from tabulate import tabulate

from .textui import C, render_card, kv_table, set_table_style, term_width, bold_white_borders, TABLE_FMT


def first_last_time(msg_df: pd.DataFrame) -> tuple[str, str]:
    if "time" not in msg_df.columns:
        return ("", "")
    try:
        t = pd.to_datetime(msg_df["time"], errors="coerce", unit=None)
        return (str(t.min()), str(t.max()))
    except Exception:
        return ("", "")


def summarize_df(df: pd.DataFrame, name: str, peek: int, c: C) -> List[str]:
    lines: List[str] = []
    title = f"{c.BOLD}{name}{c.RESET}" if c.enabled else name
    lines.append(title)
    lines.append(f"shape: {df.shape[0]} rows × {df.shape[1]} cols")
    cols = list(df.columns)
    col_str = ", ".join(cols)
    lines.append("columns: " + col_str if len(col_str) < 160 else "columns: " + ", ".join(cols[:12]) + ", …")
    dtypes = df.dtypes.astype(str).to_dict()
    na_counts = {k: int(v) for k, v in df.isna().sum().items() if int(v) > 0}
    lines.append("dtypes: " + ", ".join([f"{k}:{v}" for k, v in dtypes.items()]))
    lines.append("na_counts: " + (str(na_counts) if na_counts else "{}"))
    for col in ("type", "direction"):
        if col in df.columns:
            try:
                vc = df[col].value_counts(dropna=False).to_dict()
                lines.append(f"value_counts[{col}]: {vc}")
            except Exception:
                pass
    if "time" in df.columns:
        try:
            t = pd.to_datetime(df["time"], errors="coerce", unit=None)
            lines.append(f"time: min={t.min()} max={t.max()}")
            if t.notna().all():
                is_mono = bool((t.diff().dropna() >= pd.Timedelta(0)).all())
                lines.append(f"time monotonic nondecreasing: {is_mono}")
        except Exception:
            pass

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        sample_cols = num_cols[: min(8, len(num_cols))]
        desc_df = df[sample_cols].describe().round(6)
        lines.append(f"{c.BOLD}describe(sample numeric cols):{c.RESET}" if c.enabled else "describe(sample numeric cols):")
        lines.extend(tabulate(desc_df, headers="keys", tablefmt=TABLE_FMT).splitlines())

    if peek > 0:
        lines.append(f"{c.BOLD}head:{c.RESET}" if c.enabled else "head:")
        head_tbl = tabulate(df.head(peek), headers="keys", tablefmt=TABLE_FMT, showindex=False)
        lines.extend(head_tbl.splitlines())
        lines.append(f"{c.BOLD}tail:{c.RESET}" if c.enabled else "tail:")
        tail_tbl = tabulate(df.tail(peek), headers="keys", tablefmt=TABLE_FMT, showindex=False)
        lines.extend(tail_tbl.splitlines())

    return lines


def print_dir_listing(path: str, c: C, style: str) -> str:
    import os
    if os.path.isdir(path):
        files = sorted(os.listdir(path))
        body = [f"path: {path}", f"files: {len(files)}"]
        body += [f"• {f}" for f in files[:10]]
        if len(files) > 10:
            body.append(f"• (+{len(files)-10} more)")
    else:
        body = [f"path: {path}", f"{'files: (missing)'}"]
    return render_card("Data directory", body, c, style=style, align="left")


def print_summary(lines: list[str], c: C, style: str) -> str:
    if "" in lines:
        idx = lines.index("")
        msg_part = lines[:idx]
        ob_part  = lines[idx+1:]
    else:
        msg_part, ob_part = lines, []

    def split_title(block: list[str]) -> tuple[str, list[str]]:
        if not block:
            return ("", [])
        title, body = block[0], block[1:]
        return (title, body)

    out = []
    t1, b1 = split_title(msg_part)
    if t1:
        out.append(render_card(t1, b1, c, style=style, align="left"))
    t2, b2 = split_title(ob_part)
    if t2:
        out.append(render_card(t2, b2, c, style=style, align="left"))
    return "\n".join(out)


def _fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0; f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024.0; i += 1
    return f"{f:.2f} {units[i]}"


def print_report(W_train, W_val, W_test, meta: dict, c: C, style: str, *,
                  verbose: bool = False,
                  scaler_obj = None,
                  clip_bounds = None,
                  time_coverage: tuple[str, str] = ("","")) -> str:
    block1 = [
        ("train windows", "×".join(map(str, W_train.shape))),
        ("val windows",   "×".join(map(str, W_val.shape))),
        ("test windows",  "×".join(map(str, W_test.shape))),
        ("seq_len",       str(meta.get("seq_len"))),
        ("stride",        str(meta.get("stride"))),
        ("feature_set",   str(meta.get("feature_set"))),
        ("#features",     str(len(meta.get("feature_names", [])))),
        ("scaler",        str(meta.get("scaler"))),
        ("sorted_by_time",str(meta.get("sorted_by_time"))),
        ("every",         str(meta.get("every"))),
    ]
    lines1 = kv_table(block1, c)
    out = [render_card("Preprocessing report", lines1, c, style=style, align="right")]

    rc = meta.get("row_counts", {})
    if rc:
        block2 = [(k, str(v)) for k, v in rc.items()]
        lines2 = kv_table(block2, c)
        out.append(render_card("Row counts", lines2, c, style=style, align="right"))

    if getattr(W_train, "size", 0):
        win = W_train[0]
        block3 = [
            ("window[0] mean", f"{float(win.mean()):.6f}"),
            ("window[0] std",  f"{float(win.std()):.6f}"),
            ("features", ", ".join(meta.get("feature_names", [])[:8]) + ("…" if len(meta.get("feature_names", []))>8 else "")),
        ]
        lines3 = kv_table(block3, c)
        out.append(render_card("Sample window", lines3, c, style=style, align="right"))

    if not verbose:
        return "\n".join(out)

    vlines: list[str] = []
    total_bytes = (getattr(W_train, "nbytes", 0) + getattr(W_val, "nbytes", 0) + getattr(W_test, "nbytes", 0))
    vlines.append(f"memory total: {_fmt_bytes(total_bytes)}")
    vlines.append(f"train bytes: {_fmt_bytes(getattr(W_train, 'nbytes', 0))}")
    vlines.append(f"val bytes:   {_fmt_bytes(getattr(W_val, 'nbytes', 0))}")
    vlines.append(f"test bytes:  {_fmt_bytes(getattr(W_test, 'nbytes', 0))}")

    tmin, tmax = time_coverage
    if tmin or tmax:
        vlines.append(f"time coverage: {tmin}  →  {tmax}")

    out.append(render_card("Resources & coverage", vlines, c, style=style, align="right"))

    if scaler_obj is not None:
        s_rows = []
        if hasattr(scaler_obj, "mean_") and hasattr(scaler_obj, "scale_"):
            s_rows = [
                ("type", "StandardScaler"),
                ("mean[0:8]",  np.array2string(scaler_obj.mean_[:8],  precision=4, separator=", ")),
                ("scale[0:8]", np.array2string(scaler_obj.scale_[:8], precision=4, separator=", ")),
            ]
        elif hasattr(scaler_obj, "data_min_") and hasattr(scaler_obj, "data_max_"):
            s_rows = [
                ("type", "MinMaxScaler"),
                ("data_min[0:8]", np.array2string(scaler_obj.data_min_[:8], precision=4, separator=", ")),
                ("data_max[0:8]", np.array2string(scaler_obj.data_max_[:8], precision=4, separator=", ")),
                ("feature_range", str(getattr(scaler_obj, "feature_range", None))),
            ]
        if s_rows:
            out.append(render_card("Scaler parameters", kv_table(s_rows, c), c, style=style, align="right"))

    if clip_bounds is not None:
        lo, hi = clip_bounds
        cb_rows = [
            ("q-lo[0:8]", np.array2string(lo[:8], precision=4, separator=", ")),
            ("q-hi[0:8]", np.array2string(hi[:8], precision=4, separator=", ")),
        ]
        out.append(render_card("Clip bounds (preview)", kv_table(cb_rows, c), c, style=style, align="right"))

    def _count_windows(n_rows: int, seq_len: int, stride: int) -> int:
        if n_rows < seq_len:
            return 0
        return 1 + (n_rows - seq_len) // stride

    rc_train = rc.get("train", 0); rc_val = rc.get("val", 0); rc_test = rc.get("test", 0)
    overlap = 1.0 - (meta.get("stride", 1) / max(1, meta.get("seq_len", 1)))
    perf_rows = [
        ("expected train windows", str(_count_windows(rc_train, meta.get("seq_len", 0), meta.get("stride", 1)))),
        ("expected val windows",   str(_count_windows(rc_val,   meta.get("seq_len", 0), meta.get("stride", 1)))),
        ("expected test windows",  str(_count_windows(rc_test,  meta.get("seq_len", 0), meta.get("stride", 1)))),
        ("overlap ratio",          f"{overlap:.3f}"),
    ]
    out.append(render_card("Windowing details", kv_table(perf_rows, c), c, style=style, align="right"))

    return "\n".join(out)


def print_dataset_info(loader, c: C, style: str, peek: int = 5) -> str:
    meta = loader.get_meta()
    feature_set = meta.get("feature_set")
    feats = meta.get("feature_names") or []

    if not feats:
        if feature_set == "core":
            feats = ["mid_price","spread","mid_log_return","queue_imbalance_l1","depth_imbalance_l10"]
        elif feature_set == "raw10":
            feats = ([f"ask_price_{i}" for i in range(1,11)] +
                     [f"ask_size_{i}" for i in range(1,11)] +
                     [f"bid_price_{i}" for i in range(1,11)] +
                     [f"bid_size_{i}" for i in range(1,11)])

    intro = [
        f"Feature set: {c.BOLD}{feature_set}{c.RESET}" if c.enabled else f"Feature set: {feature_set}",
        f"Total features: {len(feats)}",
        ""
    ]

    try:
        W_train, W_val, W_test = loader.load_arrays()
        if W_train.size + W_val.size + W_test.size == 0:
            raise ValueError("No windows produced; lower seq_len or stride.")
        blocks = [W.reshape(-1, W.shape[-1]) for W in (W_train, W_val, W_test) if getattr(W,"size",0)]
        all_data = np.concatenate(blocks, axis=0)
        df = pd.DataFrame(all_data, columns=feats)

        intro.append(f"{c.BOLD}Statistical summary (aggregated across splits):{c.RESET}" if c.enabled else "Statistical summary (aggregated across splits):")
        desc_df = df.describe().round(6)
        intro.extend(tabulate(desc_df, headers="keys", tablefmt=TABLE_FMT).splitlines())
        intro.append("")

        means = df.mean().sort_values(ascending=False).head(5)
        stds  = df.std().sort_values(ascending=False).head(5)

        intro.append(f"{c.BOLD}Highest-mean features:{c.RESET}" if c.enabled else "Highest-mean features:")
        intro.extend(tabulate(list(means.items()), headers=[f"{c.MAGENTA}feature{c.RESET}" if c.enabled else "feature", "mean"], tablefmt=TABLE_FMT).splitlines())
        intro.append("")

        intro.append(f"{c.BOLD}Most-variable features (by std):{c.RESET}" if c.enabled else "Most-variable features (by std):")
        intro.extend(tabulate(list(stds.items()), headers=[f"{c.MAGENTA}feature{c.RESET}" if c.enabled else "feature", "std"], tablefmt=TABLE_FMT).splitlines())
        intro.append("")

        intro.append(f"{c.BOLD}Example rows (first few timesteps):{c.RESET}" if c.enabled else "Example rows (first few timesteps):")
        ex_tbl = tabulate(df.head(peek).round(6), headers="keys", tablefmt=TABLE_FMT, showindex=True)
        intro.extend(ex_tbl.splitlines())

    except Exception as e:
        intro.append(f"{c.RED}(Could not compute stats: {e}){c.RESET}" if c.enabled else f"(Could not compute stats: {e})")

    return render_card("Dataset summary", intro, c, style=style, align="left")