from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from purchase_intention.inference import predict_purchase
from purchase_intention.modeling import FEATURE_IMPORTANCE_PATH


# Use common dataset values for technical fields so the demo stays focused on behavior.
DEFAULT_TECHNICAL_CONTEXT = {
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
}


def clean_feature_name(feature: str) -> str:
    return (
        feature.replace("num__", "")
        .replace("cat__", "")
        .replace("_", " ")
        .replace("ProductRelated", "Product pages")
        .replace("Administrative", "Account/admin")
        .replace("Informational", "Info/help")
    )


def load_feature_importance(top_n: int = 8) -> pd.DataFrame:
    if not FEATURE_IMPORTANCE_PATH.exists():
        return pd.DataFrame(
            [{"Signal": "Train the model first", "Relative influence": "Not available"}]
        )

    importance = pd.read_csv(FEATURE_IMPORTANCE_PATH).head(top_n).copy()
    importance["Signal"] = importance["feature"].map(clean_feature_name)
    importance["Relative influence"] = importance["importance"].map(lambda value: f"{value:.1%}")
    return importance[["Signal", "Relative influence"]]


def build_probability_panel(probability: float, intent_level: str) -> str:
    color = {
        "High intent": "#15803d",
        "Medium intent": "#b45309",
        "Low intent": "#b91c1c",
    }[intent_level]
    percentage = probability * 100
    return f"""
    <div class="intent-card">
        <div class="intent-header">
            <span class="intent-label">{intent_level}</span>
            <span class="intent-percent">{percentage:.1f}%</span>
        </div>
        <div class="intent-track">
            <div class="intent-fill" style="width: {percentage:.1f}%; background: {color};"></div>
        </div>
    </div>
    """


def build_business_summary(probability: float, intent_level: str) -> str:
    if intent_level == "High intent":
        return (
            "This visitor looks close to converting. Keep the experience smooth and avoid adding friction."
        )
    if intent_level == "Medium intent":
        return (
            "This session shows meaningful interest, but the user may still need reassurance or a nudge."
        )
    return (
        "This session currently looks unlikely to convert without a strong change in behavior."
    )


def build_recommendation(intent_level: str) -> str:
    if intent_level == "High intent":
        return "Recommended action: highlight checkout, shipping clarity, and trust signals instead of extra promotions."
    if intent_level == "Medium intent":
        return "Recommended action: show social proof, a limited-time offer, or product comparisons to help the user decide."
    return "Recommended action: retarget later, simplify discovery, or surface more relevant products before pushing checkout."


def build_session_explanation(
    visitor_type: str,
    product_related: int,
    product_related_duration: float,
    bounce_rates: float,
    exit_rates: float,
    page_values: float,
    weekend: bool,
) -> str:
    signals: list[str] = []

    if visitor_type == "Returning_Visitor":
        signals.append("returning visitor behavior usually signals stronger shopping intent")
    elif visitor_type == "New_Visitor":
        signals.append("new visitors often need more reassurance before buying")

    if product_related >= 30:
        signals.append("the shopper explored many product pages")
    elif product_related <= 5:
        signals.append("the shopper viewed very few product pages")

    if product_related_duration >= 900:
        signals.append("time spent on product pages was very high")
    elif product_related_duration <= 120:
        signals.append("time spent on product pages was brief")

    if page_values >= 20:
        signals.append("high page value is a strong conversion signal")
    elif page_values == 0:
        signals.append("page value is low, which weakens purchase intent")

    if bounce_rates <= 0.02 and exit_rates <= 0.05:
        signals.append("low bounce and exit behavior suggests the visitor stayed engaged")
    elif bounce_rates >= 0.08 or exit_rates >= 0.12:
        signals.append("high bounce or exit behavior suggests friction or weak interest")

    if weekend:
        signals.append("the session happened on a weekend, which can shift shopping patterns")

    if not signals:
        signals.append("the session shows a balanced mix of browsing and engagement signals")

    bullets = "\n".join(f"- {signal.capitalize()}." for signal in signals[:4])
    return f"### Why the model responded this way\n{bullets}"


def build_payload(
    Administrative: int,
    Administrative_Duration: float,
    Informational: int,
    Informational_Duration: float,
    ProductRelated: int,
    ProductRelated_Duration: float,
    BounceRates: float,
    ExitRates: float,
    PageValues: float,
    Month: str,
    VisitorType: str,
    Weekend: bool,
) -> dict[str, object]:
    return {
        "Administrative": Administrative,
        "Administrative_Duration": Administrative_Duration,
        "Informational": Informational,
        "Informational_Duration": Informational_Duration,
        "ProductRelated": ProductRelated,
        "ProductRelated_Duration": ProductRelated_Duration,
        "BounceRates": BounceRates,
        "ExitRates": ExitRates,
        "PageValues": PageValues,
        "SpecialDay": 0.0,
        "Month": Month,
        "VisitorType": VisitorType,
        "Weekend": Weekend,
        **DEFAULT_TECHNICAL_CONTEXT,
    }


def build_improvement_payload(payload: dict[str, object]) -> dict[str, object]:
    improved = payload.copy()
    improved["ProductRelated"] = min(int(improved["ProductRelated"]) + 12, 100)
    improved["ProductRelated_Duration"] = min(
        float(improved["ProductRelated_Duration"]) + 420,
        5000,
    )
    improved["BounceRates"] = max(float(improved["BounceRates"]) * 0.6, 0.0)
    improved["ExitRates"] = max(float(improved["ExitRates"]) * 0.7, 0.0)
    improved["PageValues"] = min(float(improved["PageValues"]) + 12, 400)
    return improved


def build_scenario_comparison(payload: dict[str, object], result: dict[str, object]) -> str:
    improved_payload = build_improvement_payload(payload)
    improved_result = predict_purchase(improved_payload)
    current_probability = float(result["purchase_probability"])
    improved_probability = float(improved_result["purchase_probability"])
    lift = improved_probability - current_probability

    return (
        "### Scenario comparison\n"
        "| Session | Purchase probability | Intent segment |\n"
        "| --- | ---: | --- |\n"
        f"| Current behavior | {current_probability:.1%} | {result['intent_level']} |\n"
        f"| Stronger engagement scenario | {improved_probability:.1%} | {improved_result['intent_level']} |\n\n"
        f"Estimated lift: **{lift:+.1%}**"
    )


def run_prediction(
    Administrative: int,
    Administrative_Duration: float,
    Informational: int,
    Informational_Duration: float,
    ProductRelated: int,
    ProductRelated_Duration: float,
    BounceRates: float,
    ExitRates: float,
    PageValues: float,
    Month: str,
    VisitorType: str,
    Weekend: bool,
) -> tuple[str, str, str, str, str, str, pd.DataFrame]:
    payload = build_payload(
        Administrative,
        Administrative_Duration,
        Informational,
        Informational_Duration,
        ProductRelated,
        ProductRelated_Duration,
        BounceRates,
        ExitRates,
        PageValues,
        Month,
        VisitorType,
        Weekend,
    )
    result = predict_purchase(payload)

    probability = result["purchase_probability"]
    intent_level = result["intent_level"]
    purchase_call = "Likely to purchase" if result["predicted_purchase"] else "Unlikely to purchase"

    return (
        build_probability_panel(probability, intent_level),
        build_business_summary(probability, intent_level),
        f"{purchase_call}\nIntent segment: {intent_level}",
        build_recommendation(intent_level),
        build_session_explanation(
            visitor_type=VisitorType,
            product_related=ProductRelated,
            product_related_duration=ProductRelated_Duration,
            bounce_rates=BounceRates,
            exit_rates=ExitRates,
            page_values=PageValues,
            weekend=Weekend,
        ),
        build_scenario_comparison(payload, result),
        load_feature_importance(),
    )


APP_CSS = """
.intent-card {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 18px;
    background: #ffffff;
}
.intent-header {
    align-items: baseline;
    display: flex;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 14px;
}
.intent-label {
    font-size: 18px;
    font-weight: 700;
}
.intent-percent {
    font-size: 34px;
    font-weight: 800;
}
.intent-track {
    background: #e5e7eb;
    border-radius: 999px;
    height: 14px;
    overflow: hidden;
}
.intent-fill {
    border-radius: 999px;
    height: 100%;
}
"""


with gr.Blocks(title="Purchase Intention Demo") as demo:
    gr.Markdown(
        """
        # Purchase Intention Demo
        Estimate whether a shopping session looks ready to convert based on browsing behavior.

        This version hides the technical dataset fields and focuses on signals a person can actually understand.
        """
    )

    with gr.Row():
        with gr.Column():
            visitor_type = gr.Dropdown(
                choices=["Returning_Visitor", "New_Visitor", "Other"],
                value="Returning_Visitor",
                label="Visitor Type",
                info="Returning visitors usually have stronger buying intent.",
            )
            month = gr.Dropdown(
                choices=["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                value="Nov",
                label="Month",
            )
            weekend = gr.Checkbox(label="Weekend Session", value=False)

            product_related = gr.Slider(
                minimum=1,
                maximum=100,
                value=20,
                step=1,
                label="Product Pages Viewed",
            )
            product_related_duration = gr.Slider(
                minimum=0,
                maximum=5000,
                value=500,
                step=10,
                label="Time on Product Pages (seconds)",
            )
            informational = gr.Slider(
                minimum=0,
                maximum=25,
                value=2,
                step=1,
                label="Info or Help Pages Viewed",
            )
            informational_duration = gr.Slider(
                minimum=0,
                maximum=2000,
                value=60,
                step=10,
                label="Time on Info or Help Pages (seconds)",
            )
            administrative = gr.Slider(
                minimum=0,
                maximum=30,
                value=2,
                step=1,
                label="Account / Admin Pages Viewed",
            )
            administrative_duration = gr.Slider(
                minimum=0,
                maximum=2000,
                value=45,
                step=10,
                label="Time on Account / Admin Pages (seconds)",
            )
            bounce_rates = gr.Slider(
                minimum=0,
                maximum=0.2,
                value=0.02,
                step=0.005,
                label="Bounce Rate",
                info="Higher values suggest the visitor leaves quickly.",
            )
            exit_rates = gr.Slider(
                minimum=0,
                maximum=0.2,
                value=0.04,
                step=0.005,
                label="Exit Rate",
                info="Higher values suggest the session is losing momentum.",
            )
            page_values = gr.Slider(
                minimum=0,
                maximum=400,
                value=5,
                step=1,
                label="Page Value",
                info="A higher value suggests stronger commercial intent.",
            )

            submit = gr.Button("Analyze Session", variant="primary")

        with gr.Column():
            probability_panel = gr.HTML()
            summary = gr.Markdown()
            decision = gr.Textbox(label="Model Decision", lines=2)
            recommendation = gr.Textbox(label="Business Recommendation", lines=2)
            explanation = gr.Markdown()
            scenario_comparison = gr.Markdown()
            feature_importance = gr.Dataframe(
                value=load_feature_importance(),
                headers=["Signal", "Relative influence"],
                datatype=["str", "str"],
                label="Top Model Signals",
                interactive=False,
                wrap=True,
            )

    gr.Examples(
        examples=[
            [
                2,
                45,
                1,
                40,
                6,
                110,
                0.08,
                0.12,
                0,
                "Feb",
                "New_Visitor",
                False,
            ],
            [
                3,
                120,
                2,
                100,
                45,
                1800,
                0.01,
                0.03,
                35,
                "Nov",
                "Returning_Visitor",
                False,
            ],
            [
                1,
                20,
                1,
                30,
                18,
                650,
                0.03,
                0.06,
                12,
                "May",
                "Returning_Visitor",
                True,
            ],
        ],
        inputs=[
            administrative,
            administrative_duration,
            informational,
            informational_duration,
            product_related,
            product_related_duration,
            bounce_rates,
            exit_rates,
            page_values,
            month,
            visitor_type,
            weekend,
        ],
        label="Try sample shopper sessions",
    )

    submit.click(
        fn=run_prediction,
        inputs=[
            administrative,
            administrative_duration,
            informational,
            informational_duration,
            product_related,
            product_related_duration,
            bounce_rates,
            exit_rates,
            page_values,
            month,
            visitor_type,
            weekend,
        ],
        outputs=[
            probability_panel,
            summary,
            decision,
            recommendation,
            explanation,
            scenario_comparison,
            feature_importance,
        ],
    )


if __name__ == "__main__":
    demo.launch(css=APP_CSS, server_port=int(os.getenv("GRADIO_SERVER_PORT", "18060")))
