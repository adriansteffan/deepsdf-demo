# DeepSDF slide deck

Static export of the presentation. To view, serve `dist/` over any local HTTP server
and open it in a browser.

```bash
cd dist
python3 -m http.server 8000
```

Then open <http://localhost:8000>.

Or, if you have Node:

```bash
cd dist
npx serve
```

Open the URL it prints (usually <http://localhost:3000>).

You can't open `dist/index.html` directly, as it uses ES module
imports that browsers block over `file://`. A local server is required.

Use Space or → to advance and Shift+Space or ← to go back.
