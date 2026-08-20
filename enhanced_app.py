"""Runtime wrapper that adds the BaseLoop audio/UX upgrade layer.

The original Flask application remains the source of all routes and business logic.
This module injects versioned, additive frontend assets so the large legacy template
can stay stable while the audio engine and interaction layer evolve independently.
"""

from __future__ import annotations

from flask import request

from app import app

ASSET_VERSION = "20260820.3"
_MARKER = 'data-baseloop-enhanced="1"'
_HEAD_ASSETS = f"""
    <link rel="preload" href="/static/baseloop-enhanced.css?v={ASSET_VERSION}" as="style">
    <link rel="stylesheet" href="/static/baseloop-enhanced.css?v={ASSET_VERSION}" {_MARKER}>
""".rstrip()
_BODY_ASSETS = "\n".join(
    (
        f'<script defer src="/static/baseloop-enhanced-core.js?v={ASSET_VERSION}" {_MARKER}></script>',
        f'<script defer src="/static/baseloop-enhanced-audio.js?v={ASSET_VERSION}" {_MARKER}></script>',
        f'<script defer src="/static/baseloop-mix-balance.js?v={ASSET_VERSION}" {_MARKER}></script>',
    )
)


@app.after_request
def inject_baseloop_enhancements(response):
    """Inject the additive audio/UX layer into successful HTML page responses."""
    if request.method != "GET" or response.status_code != 200:
        return response
    if response.mimetype != "text/html" or response.direct_passthrough:
        return response

    html = response.get_data(as_text=True)
    if not html or _MARKER in html:
        return response

    if "</head>" in html:
        html = html.replace("</head>", f"{_HEAD_ASSETS}\n</head>", 1)
    if "</body>" in html:
        html = html.replace("</body>", f"{_BODY_ASSETS}\n</body>", 1)

    response.set_data(html)
    response.headers["X-BaseLoop-Enhancement"] = ASSET_VERSION
    return response


__all__ = ["app"]
