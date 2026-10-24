"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass

import streamlit as st

import wsz_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None      # bei Auswahlreglern: die erlaubten Werte


SETTING_SPECS = {
    "layout_select": SettingSpec("layout", str, C._BASE["layout"], choices=C.LAYOUTS),
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "weights_select": SettingSpec("w", str, C._BASE["weights"], choices=C.WEIGHT_KINDS),
    "share_slider": SettingSpec("share", int, C.DEFAULT_SHARE, C.SHARE_MIN, C.SHARE_MAX),
    "factor_slider": SettingSpec("f", float, C.DEFAULT_FACTOR, C.FACTOR_MIN, C.FACTOR_MAX),
    "s_slider": SettingSpec("s", float, C.DEFAULT_S, C.S_MIN, C.S_MAX),
    "angle_slider": SettingSpec("angle", int, C.DEFAULT_ANGLE, C.ANGLE_MIN, C.ANGLE_MAX),
    "start_select": SettingSpec("start", str, C._BASE["start"], choices=C.STARTS),
    "solver_select": SettingSpec("solver", str, C._BASE["solver"], choices=C.SOLVERS),
    "tol_slider": SettingSpec("tol", int, C.DEFAULT_TOL_EXP, C.TOL_EXP_MIN, C.TOL_EXP_MAX),
    "maxit_select": SettingSpec("maxit", int, C.DEFAULT_MAXIT, choices=C.MAXIT_OPTIONS),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "customer_slider": SettingSpec("kunde", int, C.DEFAULT_CUSTOMER, C.CUSTOMER_MIN, C.N_MAX),
}
PRESET_KEYS = {
    "layout": "layout_select", "n": "n_slider", "weights": "weights_select", "share": "share_slider", "factor": "factor_slider", "s": "s_slider",
    "angle": "angle_slider", "start": "start_select", "solver": "solver_select", "tol_exp": "tol_slider", "maxit": "maxit_select", "seed": "seed_input",
    "customer": "customer_slider",
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf erlaubte Werte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if spec.choices is not None:
                if value not in spec.choices:
                    continue
            else:
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
            st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
