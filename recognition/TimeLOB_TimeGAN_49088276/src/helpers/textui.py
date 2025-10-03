import os
import re
import shutil
from datetime import datetime
from typing import List, Tuple, Sequence
from tabulate import tabulate

# Try Colorama on Windows (optional)
try:
    import colorama  # type: ignore
    colorama.just_fix_windows_console()
except Exception:
    pass

# ---------------- defaults ----------------
DEFAULT_STYLE = "box"   # default to box panels
TABLE_FMT = "github"    # tabulate format; switch with set_table_style()

# ------------- terminal capabilities & colors -------------
def supports_color(no_color_flag: bool) -> bool:
    if no_color_flag or os.environ.get("NO_COLOR"):
        return False
    try:
        # If stdout is a TTY, assume color; terminals and most IDE consoles support it.
        return os.isatty(1)
    except Exception:
        return False

class C:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.RESET   = "\033[0m"  if enabled else ""
        self.DIM     = "\033[2m"  if enabled else ""
        self.BOLD    = "\033[1m"  if enabled else ""
        self.CYAN    = "\033[36m" if enabled else ""
        self.YELLOW  = "\033[33m" if enabled else ""
        self.GREEN   = "\033[32m" if enabled else ""
        self.MAGENTA = "\033[35m" if enabled else ""
        self.BLUE    = "\033[34m" if enabled else ""
        self.RED     = "\033[31m" if enabled else ""
        self.WHITE   = "\033[37m" if enabled else ""

# ------------- ANSI helpers -------------
_ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

def visible_len(s: str) -> int:
    """Printable width (strip ANSI first)."""
    return len(_ANSI_RE.sub("", s))

def strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)

def truncate_visible(s: str, max_cols: int) -> str:
    """
    Truncate to max_cols printable columns without breaking ANSI sequences.
    """
    if max_cols <= 0:
        return ""
    out, cols = [], 0
    i, n = 0, len(s)
    while i < n and cols < max_cols:
        m = _ANSI_RE.match(s, i)
        if m:
            out.append(m.group(0))
            i = m.end()
            continue
        ch = s[i]
        out.append(ch)
        cols += 1
        i += 1
    # ensure we don't end inside an ANSI state (we don't maintain state machine,
    # but common sequences are self-contained; still append reset for safety)
    if cols >= max_cols:
        out.append("\033[0m")
    return "".join(out)

def ljust_visible(s: str, width: int) -> str:
    pad = max(0, width - visible_len(s))
    return s + (" " * pad)

# ------------- layout helpers -------------
def set_table_style(name: str) -> None:
    """Set tabulate tablefmt. Small whitelist, but allow custom strings."""
    global TABLE_FMT
    allowed = {
        "github", "grid", "fancy_grid", "heavy_grid", "simple", "outline",
        "rounded_grid", "double_grid", "pipe", "orgtbl", "jira", "psql"
    }
    TABLE_FMT = name if name in allowed else name  # pass-through (tabulate will raise if invalid)

def term_width(default: int = 100) -> int:
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except Exception:
        return default

def wrap_text(s: str, width: int) -> List[str]:
    """
    ANSI-aware word wrap by visible width.
    """
    if visible_len(s) <= width:
        return [s]
    parts = s.split(" ")
    out, cur = [], ""
    for tok in parts:
        if not cur:
            cur = tok
        elif visible_len(cur) + 1 + visible_len(tok) <= width:
            cur += " " + tok
        else:
            out.append(cur)
            cur = tok
    if cur:
        out.append(cur)
    return out

def is_table_line(s: str) -> bool:
    """
    Heuristic: lines that look like tables (markdown pipes or box-drawing).
    """
    t = strip_ansi(s).strip()
    if not t:
        return False
    if t.startswith("|") and "|" in t[1:]:
        return True
    if t.startswith("+") and t.endswith("+"):
        return True
    # box drawing / markdown borders
    if set(t) <= set("-:|+ ─═│║┼┬┴├┤┌┐└┘╭╮╯╰╪╫╠╬╣╦╩╔╗╚╝"):
        return True
    return False

# ------------- table/border styling -------------
def bold_white_borders(table: str, c: C) -> str:
    """
    Paint table border glyphs in bold white without touching cell content.
    Works for markdown pipes and Unicode box drawing.
    """
    if not getattr(c, "enabled", False):
        return table

    bold, white, reset = c.BOLD, c.WHITE, c.RESET
    border_chars = set("│║|┼┬┴├┤┌┐└┘─═╭╮╯╰╪╫╠╬╣╦╩╔╗╚╝+-:")
    horiz_set = set("─═-")
    vert_set = set("│║|:")

    def paint(ch: str) -> str:
        return f"{bold}{white}{ch}{reset}"

    painted_lines = []
    for raw in table.splitlines():
        line = raw
        # operate on non-ANSI plane but keep indexes by iterating char-by-char
        out_chars = []
        for ch in line:
            if ch in border_chars:
                out_chars.append(paint(ch))
            else:
                out_chars.append(ch)
        painted_lines.append("".join(out_chars))
    return "\n".join(painted_lines)

def kv_table(
    rows: List[Tuple[str, str]],
    c: C,
    headers: Tuple[str, str] = ("key", "value"),
) -> List[str]:
    if not rows:
        return []

    if c.enabled:
        h_key = f"{c.BOLD}{c.MAGENTA}{headers[0]}{c.RESET}"
        h_val = f"{c.BOLD}{c.MAGENTA}{headers[1]}{c.RESET}"
        tinted = [(f"{c.CYAN}{k}{c.RESET}", v) for k, v in rows]
    else:
        h_key, h_val = headers
        tinted = rows

    table_txt = tabulate(
        tinted,
        headers=[h_key, h_val],
        tablefmt=TABLE_FMT,
        stralign="left",
        disable_numparse=True,
    )
    table_txt = bold_white_borders(table_txt, c)
    return table_txt.splitlines()

# -------------------- NEW: generic table renderer --------------------
def table(
    rows: Sequence[Sequence[str]],
    headers: Sequence[str],
    c: C,
    *,
    tint_header: bool = True,
    tint_first_col: bool = True,
) -> List[str]:
    """
    Render a 2D table (rows + headers) with optional header-row tint
    and first-column tint, plus bold white borders.
    """
    rows_list = [list(map(str, r)) for r in rows]
    if c.enabled and tint_first_col and rows_list:
        for i, r in enumerate(rows_list):
            if r:
                r[0] = f"{c.YELLOW}{r[0]}{c.RESET}"

    if c.enabled and tint_header:
        hdr = [f"{c.BOLD}{c.MAGENTA}{h}{c.RESET}" for h in headers]
    else:
        hdr = list(map(str, headers))

    tbl = tabulate(
        rows_list,
        headers=hdr,
        tablefmt=TABLE_FMT,
        stralign="left",
        disable_numparse=True,
        showindex=False,
    )
    tbl = bold_white_borders(tbl, c)
    return tbl.splitlines()

# ------------- message bubbles & panels -------------
def _bubble(title: str, body_lines: List[str], c: C, align: str = "left", width: int | None = None) -> str:
    termw = term_width()
    width = min(termw, width or termw)
    base_inner = max(24, width - 10)

    widest_tbl = 0
    for ln in body_lines:
        if is_table_line(ln):
            widest_tbl = max(widest_tbl, visible_len(ln))

    max_inner = min(max(base_inner, widest_tbl), width - 10)
    indent = 2 if align == "left" else max(2, width - (max_inner + 8))
    pad = " " * indent

    ts = datetime.now().strftime("%H:%M")
    title_colored = f"{c.BOLD}{c.BLUE}{title}{c.RESET}" if c.enabled else title
    head = f"{title_colored}  {c.DIM}{ts}{c.RESET}"
    head_lines = wrap_text(head, max_inner)

    lines = [pad + " " + head_lines[0]]
    for hl in head_lines[1:]:
        lines.append(pad + " " + hl)

    lines.append(pad + "  " + ("╭" + "─" * (max_inner + 2) + "╮"))

    for ln in body_lines:
        if is_table_line(ln):
            width_ok = max_inner
            body = ljust_visible(ln, width_ok)
            body = truncate_visible(body, width_ok)
            lines.append(pad + "  " + "│ " + body + " │")
        else:
            for wln in wrap_text(ln, max_inner):
                lines.append(pad + "  " + "│ " + ljust_visible(wln, max_inner) + " │")

    tail_left  = pad + "  " + "╰" + "─" * (max_inner + 2) + "╯" + "⟋"
    tail_right = pad + " "  + "⟍" + "╰" + "─" * (max_inner + 2) + "╯"
    lines.append(tail_left if align == "left" else tail_right)
    return "\n".join(lines)

def _panel(title: str, body_lines: List[str], c: C, width: int | None = None) -> str:
    termw = term_width()
    width = width or termw
    inner = width - 4

    widest_tbl = 0
    for ln in body_lines:
        if is_table_line(ln):
            widest_tbl = max(widest_tbl, visible_len(ln))
    inner = min(max(inner, widest_tbl + 2), termw - 4)
    width = inner + 4

    border = "─" * (width - 2)
    title_colored = f"{c.BOLD}{c.BLUE}{title}{c.RESET}" if c.enabled else title
    out = [f"{c.CYAN}┌{border}┐{c.RESET}"]
    title_line = f" {title_colored} "
    pad_space = max(0, width - 2 - visible_len(title_line))
    out.append(f"{c.CYAN}│{c.RESET}{title_line}{' '*pad_space}{c.CYAN}│{c.RESET}")
    out.append(f"{c.CYAN}├{border}┤{c.RESET}")

    content_width = inner - 2
    for ln in body_lines:
        if is_table_line(ln):
            body = ljust_visible(ln, content_width)
            body = truncate_visible(body, content_width)
            out.append(f"{c.CYAN}│{c.RESET} {body} {c.CYAN}│{c.RESET}")
        else:
            for sub in wrap_text(ln, content_width):
                out.append(f"{c.CYAN}│{c.RESET} {ljust_visible(sub, content_width)} {c.CYAN}│{c.RESET}")

    out.append(f"{c.CYAN}└{border}┘{c.RESET}")
    return "\n".join(out)

def render_card(title: str, body_lines: List[str], c: C, style: str = DEFAULT_STYLE, align: str = "left") -> str:
    return _bubble(title, body_lines, c, align=align) if style == "chat" else _panel(title, body_lines, c)

# Convenience sugar for quick key→value panels
def render_kv_panel(title: str, rows: List[Tuple[str, str]], c: C, style: str = DEFAULT_STYLE, align: str = "right") -> str:
    return render_card(title, kv_table(rows, c), c, style=style, align=align)
