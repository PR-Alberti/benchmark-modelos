"""Formatacao de numeros, duracoes e imagens para a pagina e o markdown."""

import base64
import html
import io
import subprocess

from .config import REPO


def jpeg(t, lado=224, qualidade=80):
    from PIL import Image
    a = (t.float().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    im = Image.fromarray(a)
    if im.size[0] != lado:
        im = im.resize((lado, lado), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=qualidade, optimize=True, progressive=True)
    return buf.getvalue()


def data_uri(b):
    return "data:image/jpeg;base64," + base64.b64encode(b).decode()


def tempo_fmt(seg):
    if seg is None:
        return None
    h, m = divmod(int(round(seg / 60)), 60)
    return f"{h} h {m:02d}" if h else f"{m} min"


def duracao(seg):
    """Como tempo_fmt, mas com segundos para o que dura pouco (o FRR de 1 sessao leva ~10 s)."""
    if seg is None:
        return None
    return f"{int(round(seg))} s" if seg < 90 else tempo_fmt(seg)


def commit():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "?"


def fmt(v, formato):
    if v is None:
        return None
    if formato == "pct":
        return f"{v * 100:.1f}%".replace(".", ",")
    return f"{v:.3f}".replace(".", ",")


def mi(n):
    return f"{n / 1e6:,.1f} M".replace(",", "X").replace(".", ",").replace("X", ".")


def bi(n):
    return f"{n / 1e9:.2f} bi".replace(".", ",")


def parte(trein, total):
    return f"{trein / total * 100:.2f}".replace(".", ",") + "%"


def esc(s):
    return html.escape(str(s), quote=True)


def virg(x, casas=2):
    return f"{x:.{casas}f}".replace(".", ",")


def frac_txt(x):
    """Fracao do FRR sem zeros sobrando: 0,001  0,05  0,1  1."""
    return f"{x:.3f}".rstrip("0").rstrip(".").replace(".", ",")


def milhar(n):
    return f"{n:,}".replace(",", ".")
