from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from purchase_intention.inference import predict_purchase


# Use common dataset values for technical fields so the demo stays focused on behavior.
DEFAULT_TECHNICAL_CONTEXT = {
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
}


def build_business_summary(probability: float, intent_level: str) -> str:
    if intent_level == "High intent":
        return (
            f"### {probability:.1%} purchase probability\n"
            "This visitor looks close to converting. Keep the experience smooth and avoid adding friction."
        )
    if intent_level == "Medium intent":
        return (
            f"### {probability:.1%} purchase probability\n"
            "This session shows meaningful interest, but the user may still need reassurance or a nudge."
        )
    return (
        f"### {probability:.1%} purchase probability\n"
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
) -> tuple[str, str, str, str]:
    payload = {
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
    result = predict_purchase(payload)

    probability = result["purchase_probability"]
    intent_level = result["intent_level"]
    purchase_call = "Likely to purchase" if result["predicted_purchase"] else "Unlikely to purchase"

    return (
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
    )


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
            summary = gr.Markdown()
            decision = gr.Textbox(label="Model Decision", lines=2)
            recommendation = gr.Textbox(label="Business Recommendation", lines=2)
            explanation = gr.Markdown()

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
        outputs=[summary, decision, recommendation, explanation],
    )


if __name__ == "__main__":
    demo.launch()
