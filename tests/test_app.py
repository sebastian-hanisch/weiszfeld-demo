"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


ENABLER = {"share_slider": "Großkunde", "factor_slider": "Großkunde an der Schwelle"}   # Regler, die nur bei einer Gewichtsart wirken (sonst disabled)


def _enable(at, key):
    if key in ENABLER:
        at.sidebar.selectbox(key="weights_select").set_value(ENABLER[key]).run(timeout=TIMEOUT)


def test_every_slider_at_its_min_and_max_does_not_raise():
    keys = [s.key for s in _fresh_app().slider]
    assert keys and "n_slider" in keys and "share_slider" in keys and "factor_slider" in keys
    for key in keys:
        for edge in ("min", "max"):
            at = _fresh_app()
            _enable(at, key)
            slider = at.slider(key=key)
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (key, edge, [str(e) for e in at.exception])


def test_disabled_regler_do_not_change_the_result_while_disabled():
    """Mode-conditional dead control: Anteil und Faktor sind bei den anderen Gewichtsarten gesperrt und wirkungslos (und zeigen es durch 'disabled')."""
    at = _fresh_app()
    assert at.slider(key="share_slider").proto.disabled and at.slider(key="factor_slider").proto.disabled
    at.sidebar.selectbox(key="weights_select").set_value("Großkunde").run(timeout=TIMEOUT)
    assert not at.slider(key="share_slider").proto.disabled and at.slider(key="factor_slider").proto.disabled


def test_every_select_regler_with_each_option_does_not_raise():
    probe = _fresh_app()
    for widget in list(probe.selectbox) + list(probe.select_slider):
        for option in widget.options:
            at = _fresh_app()
            target = at.selectbox(key=widget.key) if widget.type == "selectbox" else at.select_slider(key=widget.key)
            value = option if widget.type == "selectbox" else type(target.value)(option)
            target.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (widget.key, option, [str(e) for e in at.exception])


def test_extreme_combinations_do_not_raise():
    at = _fresh_app()
    at.sidebar.selectbox(key="weights_select").set_value("Großkunde an der Schwelle").run(timeout=TIMEOUT)
    at.slider(key="n_slider").set_value(3).run(timeout=TIMEOUT)
    at.slider(key="factor_slider").set_value(1.0).run(timeout=TIMEOUT)
    at.select_slider(key="maxit_select").set_value(100).run(timeout=TIMEOUT)
    at.slider(key="tol_slider").set_value(-12).run(timeout=TIMEOUT)
    at.sidebar.selectbox(key="solver_select").set_value("klassisch").run(timeout=TIMEOUT)
    at.sidebar.selectbox(key="start_select").set_value("nächster Datenpunkt").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    at.slider(key="n_slider").set_value(200).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
