#!/usr/bin/env python3
"""Capture Streamlit dashboard screenshots via a running Chrome CDP session.

Usage:
    streamlit run app_solar_monitoring_enhanced.py --server.headless true --server.port 8520
    /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --headless=new --remote-debugging-port=9222 about:blank
    python capture_dashboard_screenshots.py http://localhost:8520
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import requests
import websocket


CDP_PORT = int(os.getenv('CHROME_CDP_PORT', '9222'))
OUT_DIR = Path('docs/screenshots')


class CDPClient:
    def __init__(self, ws_url: str):
        self.ws = websocket.create_connection(ws_url, timeout=10)
        self.next_id = 1

    def call(self, method: str, params: dict | None = None):
        msg_id = self.next_id
        self.next_id += 1
        self.ws.send(json.dumps({'id': msg_id, 'method': method, 'params': params or {}}))
        while True:
            message = json.loads(self.ws.recv())
            if message.get('id') == msg_id:
                if 'error' in message:
                    raise RuntimeError(message['error'])
                return message.get('result', {})

    def evaluate(self, expression: str):
        return self.call('Runtime.evaluate', {
            'expression': expression,
            'awaitPromise': True,
            'returnByValue': True,
        })


def get_tab_ws_url(url: str) -> str:
    requests.put(f'http://localhost:{CDP_PORT}/json/new?{url}', timeout=10)
    targets = json.load(urlopen(f'http://localhost:{CDP_PORT}/json/list', timeout=10))
    for target in targets:
        if target.get('type') == 'page' and target.get('url', '').startswith(url):
            return target['webSocketDebuggerUrl']
    for target in targets:
        if target.get('type') == 'page':
            return target['webSocketDebuggerUrl']
    raise RuntimeError('No Chrome page target found')


def wait_for_app(cdp: CDPClient, timeout: int = 90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            result = cdp.evaluate("document.body && document.body.innerText")
            text = result.get('result', {}).get('value', '') or ''
            if 'Solar Forecasting & Anomaly Dashboard' in text and 'Fleet status overview' in text:
                return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError('Streamlit app did not finish rendering in time')


def click_text(cdp: CDPClient, selector: str, text: str):
    expression = f"""
    (() => {{
      const items = Array.from(document.querySelectorAll({selector!r}));
      const el = items.find(e => (e.innerText || e.textContent || '').includes({text!r}));
      if (!el) return false;
      el.click();
      return true;
    }})()
    """
    result = cdp.evaluate(expression)
    if not result.get('result', {}).get('value'):
        raise RuntimeError(f'Could not click text: {text}')
    time.sleep(5)


def screenshot(cdp: CDPClient, filename: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cdp.call('Emulation.setDeviceMetricsOverride', {
        'width': 1440,
        'height': 1400,
        'deviceScaleFactor': 1,
        'mobile': False,
    })
    data = cdp.call('Page.captureScreenshot', {
        'format': 'png',
        'captureBeyondViewport': False,
    })['data']
    path = OUT_DIR / filename
    path.write_bytes(base64.b64decode(data))
    print(path)


def main():
    app_url = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:8520'
    cdp = CDPClient(get_tab_ws_url(app_url))
    cdp.call('Page.enable')
    cdp.call('Runtime.enable')
    wait_for_app(cdp)

    screenshot(cdp, 'executive-summary.png')

    click_text(cdp, '[role="tab"]', 'Data Sources')
    screenshot(cdp, 'data-sources-quality.png')

    click_text(cdp, 'button', 'Plant E')
    time.sleep(8)
    screenshot(cdp, 'plant-e-data-quality.png')

    click_text(cdp, '[role="tab"]', 'Model')
    screenshot(cdp, 'modeling-notes.png')


if __name__ == '__main__':
    main()
