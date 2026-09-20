"""Pin Gradio 3's built-in controls to English for this application page."""

from jinja2 import ChoiceLoader, DictLoader


# Gradio 3.39 initializes its translation store from navigator.language and
# has no public locale option. Set page-local values before its module loads;
# this does not change browser preferences or modify installed Gradio files.
ENGLISH_BOOTSTRAP = """<script id="app-english-locale">
Object.defineProperty(window.navigator, "language", {
    configurable: true, get: () => "en-US"
});
Object.defineProperty(window.navigator, "languages", {
    configurable: true, get: () => ["en-US", "en"]
});
document.documentElement.lang = "en";
</script>"""


def configure_english_ui():
    """Install an in-memory template override before Gradio serves its page."""
    from gradio.routes import templates

    env = templates.env
    overrides = {}
    for name in ("frontend/index.html", "frontend/share.html"):
        source, _, _ = env.loader.get_source(env, name)
        if 'id="app-english-locale"' in source:
            continue
        if "<head>" not in source:
            raise RuntimeError("Unsupported Gradio template; expected Gradio 3.39.0.")
        overrides[name] = source.replace("<head>", "<head>\n" + ENGLISH_BOOTSTRAP, 1)
    if overrides:
        env.loader = ChoiceLoader([DictLoader(overrides), env.loader])
        env.cache.clear()
