from pathlib import Path

from lib.smart_beta_llm_wrapper import (
    extract_factor_from_pdf
)


PAPER = Path(
    "Research/Oil_Factor_Mock_Research_Paper.pdf"
)

AVAILABLE_FACTORS = {
    "MKTB": "Equity market beta",
    "CCDB": "Credit spread sensitivity",
    "OILP": "Sensitivity of equity returns to oil-price changes",
    "CONS": "Consumer conditions sensitivity",
    "EMPR": "Employment conditions sensitivity",
}


def main():

    print("Testing LLM research extraction...")

    interpretation = extract_factor_from_pdf(
        pdf_path=PAPER,
        available_factors=AVAILABLE_FACTORS,
        model="gpt-5.1"
    )

    print("\nLLM OUTPUT:")
    print(interpretation.model_dump_json(indent=2))

    assert interpretation.selected_factor == "OILP", (
        "Expected the agent to identify OILP."
    )

    assert interpretation.can_implement, (
        "The agent considered the paper incompatible."
    )

    print("\nSUCCESS: Correct factor identified!")


if __name__ == "__main__":
    main()