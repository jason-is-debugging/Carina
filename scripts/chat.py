"""Streamlit chat UI for Carina, with edit-to-learn feedback.

Run with::

    uv run streamlit run scripts/chat.py

Layout
------
* Main area: the conversation. Each assistant reply carries a text area
  where the user can correct it; saving the correction records it as
  feedback.
* Sidebar: checkpoint picker, sampling parameters, and the learning
  controls (learning rate, anchor weight, epochs) plus a button that
  runs the online update over the collected feedback.

The learning step is deliberately a *separate, explicit action*: the
generation path stays inference-only, and pressing "learn" performs a
bounded number of gradient steps and then reloads the updated weights.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from carina.config import (  # noqa: E402
    FeedbackConfig,
    ServeConfig,
)
from carina.serve.engine import (  # noqa: E402
    ChatEngine,
    ChatTurn,
    find_checkpoint,
)
from carina.serve.feedback import FeedbackRecord, FeedbackStore  # noqa: E402
from carina.serve.learning import OnlineLearner  # noqa: E402


def _init_state() -> None:
    """Seed the session state on first run."""
    if "turns" not in st.session_state:
        st.session_state.turns = []
    if "pending" not in st.session_state:
        st.session_state.pending = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:8]
    if "checkpoint" not in st.session_state:
        st.session_state.checkpoint = ""
    if "last_stats" not in st.session_state:
        st.session_state.last_stats = None


def _load_engine(path: str) -> ChatEngine | None:
    """Load (or reload) the engine for the given checkpoint."""
    key = f"engine::{path}"
    if st.session_state.get(key) is None:
        try:
            st.session_state[key] = ChatEngine(
                path,
                tokenizer_dir="",
                turn_separator=ServeConfig.turn_separator,
                max_context_chars=ServeConfig.max_context_chars,
                lora_reusage=FeedbackConfig.lora_reusage,
                max_loop_times=FeedbackConfig.max_loop_times,
                tokenizer_dirs=(
                    ServeConfig.tokenizer_dir,
                    ServeConfig.tokenizer_dir_qwen,
                ),
            )
        except Exception as err:  # noqa: BLE001
            st.error(f"Failed to load checkpoint: {type(err).__name__}: {err}")
            return None
    return st.session_state[key]


def _sidebar() -> tuple[str, dict[str, Any]]:
    """Render the sidebar and return (checkpoint path, params)."""
    st.sidebar.title("Carina")
    st.sidebar.caption("Local AI assistant with edit-to-learn feedback")

    saved_root = Path(ServeConfig.saved_root)
    candidates = sorted(
        saved_root.glob("*/best.pt"), key=lambda p: p.stat().st_mtime,
        reverse=True)
    options = [str(p) for p in candidates]
    if not options:
        options = [str(p) for p in sorted(
            saved_root.glob("*/latest.pt"),
            key=lambda p: p.stat().st_mtime, reverse=True)]
    if not options:
        st.sidebar.error(f"No checkpoint found under {saved_root}")
        return "", {}

    current = st.session_state.get("checkpoint", "")
    index = options.index(current) if current in options else 0
    chosen = st.sidebar.selectbox("Checkpoint", options, index=index)
    st.session_state.checkpoint = chosen
    st.sidebar.caption(chosen)

    with st.sidebar.expander("Sampling", expanded=False):
        temperature = st.slider("temperature", 0.0, 1.5, ServeConfig.temperature,
                                0.05)
        top_k = st.slider("top_k", 0, 200, ServeConfig.top_k, 1)
        max_new = st.slider("max_new_tokens", 32, 512, ServeConfig.max_new_tokens,
                            16)

    with st.sidebar.expander("Learning", expanded=False):
        learning_rate = st.number_input(
            "learning_rate", min_value=1e-7, max_value=1e-3,
            value=float(FeedbackConfig.learning_rate), format="%.1e")
        epochs = st.slider("epochs", 1, 20, FeedbackConfig.epochs, 1)
        anchor_weight = st.slider(
            "anchor_weight", 0.0, 5.0, FeedbackConfig.anchor_weight, 0.1)
        freeze_embed = st.checkbox(
            "Freeze embedding", value=FeedbackConfig.freeze_embedding,
            help="The tied embedding has a dense gradient over the whole "
                 "vocabulary (151665x2048 means 594MB in bf16 alone), and "
                 "changing it perturbs the output distribution of every "
                 "token at once.")

    params = {
        "temperature": temperature,
        "top_k": top_k,
        "max_new_tokens": max_new,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "anchor_weight": anchor_weight,
        "freeze_embedding": freeze_embed,
    }
    return chosen, params


def _render_conversation(engine: ChatEngine, store: FeedbackStore) -> None:
    """Render the chat history with per-reply edit boxes.

    The correction is committed with an explicit button rather than on
    blur: a keyed text area keeps its widget state across reruns, so
    committing on every rerun would append the same record repeatedly.
    """
    turns: list[ChatTurn] = st.session_state.turns
    for index, turn in enumerate(turns):
        with st.chat_message("user"):
            st.markdown(turn.user)
        with st.chat_message("assistant"):
            st.markdown(turn.assistant)
            draft_key = f"draft_{index}"
            draft = st.text_area(
                "Correct the reply", key=draft_key, height=90, value=turn.edited)
            cols = st.columns([1, 3])
            if cols[0].button("Save correction", key=f"save_{index}"):
                edited = draft.strip()
                if edited and edited != turn.assistant.strip():
                    turn.edited = edited
                    record = FeedbackRecord(
                        context=turn.context,
                        model_reply=turn.assistant,
                        edited_reply=edited,
                        session=st.session_state.session_id,
                    )
                    store.append(record)
                    st.success("Recorded as a learning sample")
                    st.rerun()
                elif not edited:
                    st.info("The correction is empty, nothing recorded")


def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(page_title="Carina", layout="centered")
    _init_state()
    checkpoint, params = _sidebar()
    if not checkpoint:
        return
    engine = _load_engine(checkpoint)
    if engine is None:
        return
    store = FeedbackStore(ServeConfig.feedback_file)

    st.sidebar.metric("Feedback collected", store.count())
    if st.sidebar.button("Start learning (backprop)", type="primary"):
        _run_learning(engine, store, params)
        st.rerun()

    _render_conversation(engine, store)

    prompt = st.chat_input("Say something...")
    if not prompt:
        return
    turns: list[ChatTurn] = st.session_state.turns
    with st.chat_message("user"):
        st.markdown(prompt)
    context = engine.build_prompt(turns, prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = engine.generate(
                context,
                max_new_tokens=params["max_new_tokens"],
                temperature=params["temperature"],
                top_k=params["top_k"],
            )
        st.markdown(reply)
    turns.append(ChatTurn(user=prompt, assistant=reply, context=context))
    st.session_state.turns = turns
    # Rerun once so this turn's edit box gets rendered (the edit boxes in
    # _render_conversation are generated from the history list one by one,
    # and this turn is only appended after that).
    st.rerun()


def _run_learning(
    engine: ChatEngine,
    store: FeedbackStore,
    params: dict[str, Any],
) -> None:
    """Run the online update over the collected feedback."""
    usable = store.usable()
    if not usable:
        st.sidebar.warning("No usable feedback samples yet")
        return
    learner = OnlineLearner(
        engine.model,
        engine.tokenizer,
        engine.device,
        max_length=ServeConfig.max_length,
        learning_rate=params["learning_rate"],
        weight_decay=FeedbackConfig.weight_decay,
        grad_clip=FeedbackConfig.grad_clip,
        use_master=FeedbackConfig.use_master,
        n_feedback=FeedbackConfig.n_feedback,
        n_anchor=FeedbackConfig.n_anchor,
        anchor_weight=params["anchor_weight"],
        feedback_weight=FeedbackConfig.feedback_weight,
        replay_ratio=FeedbackConfig.replay_ratio,
        replay_capacity=FeedbackConfig.replay_capacity,
        anchor_corpus=FeedbackConfig.anchor_corpus,
        freeze_embedding=params["freeze_embedding"],
    )
    with st.spinner(f"Running {params['epochs']} update rounds over "
                    f"{len(usable)} feedback samples..."):
        stats = learner.learn(usable, epochs=params["epochs"])
    out_dir = Path(ServeConfig.learned_model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / FeedbackConfig.output_name
    learner.save(target)
    st.session_state.last_stats = stats
    # Make the following conversation use the updated weights: rebuild the
    # engine from the new checkpoint.
    st.session_state.checkpoint = str(target)
    st.session_state.pop(f"engine::{engine.checkpoint}", None)
    st.sidebar.success(
        f"Learning done, loss {stats['loss_before']:.3f} -> {stats['loss_after']:.3f}")
    st.sidebar.caption(f"New weights: {target}")


if __name__ == "__main__":
    main()
