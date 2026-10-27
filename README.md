# LLM Integration — Proof of Concept and Future Applications

## Overview

This project explores how large language models (LLMs) can automate the translation of financial research into implementable investment strategies and evaluate their economic value to an asset manager.

The broader objective is to develop an end-to-end research pipeline:

##Research Paper → LLM Interpretation → Portfolio Construction → Backtesting → Strategy Economics

Rather than simply summarizing research, the LLM serves as an interface between unstructured investment research and a quantitative investment management framework.

## Current Implementation: A Simplified Proof of Concept

The current implementation is deliberately a toy example designed to demonstrate the feasibility of this workflow.

We begin with an existing database of security-level exposures to several macroeconomic factors, including equity market, credit, oil, consumption, and employment sensitivities.

Rather than asking the LLM to independently implement a new factor model, the current workflow is:

A user uploads a research paper describing an investment factor.

The LLM interprets the paper and identifies the corresponding factor in our existing database.

The portfolio optimizer constructs three portfolios with positive, neutral, and negative exposure to the selected factor.

A simulated forecasting signal determines which portfolio to hold at each rebalance date.

The backtesting and Strategy Economics engines evaluate portfolio performance and the incremental economic value of predictive information.

For example, a paper discussing oil-price sensitivity is mapped to the existing OILP factor. The optimizer then constructs portfolios with different target oil exposures, allowing us to investigate the economic value of forecasting oil-price movements.

Importantly, the LLM does not currently generate new factor definitions, estimate factor exposures, or modify portfolio optimization code. It maps the research to an existing, supported factor. This restriction is intentional: it greatly simplifies the demonstration and allows us to isolate and validate the LLM's role without introducing unnecessary implementation complexity.

## Intended Full-Scale Implementation

In a more complete implementation, we envision removing the requirement that a factor already exist in our database.

The LLM would instead interpret the research methodology, identify the required data, translate factor definitions into executable calculations, and construct portfolios consistent with the paper's specifications.

The expanded workflow would be:

Research Paper → LLM-Generated Factor Specification → Data Acquisition → Factor Estimation → Portfolio Construction → Backtesting → Economic Valuation

This would support research involving previously unseen factors, different predictive models, and more sophisticated portfolio construction methodologies.

Such a system would require additional safeguards, including code validation, point-in-time data controls, reproducibility checks, and human approval of generated implementations.

## Applications Beyond Smart Beta

Our current proof of concept demonstrates just one application: a smart-beta strategy that dynamically tilts exposure to a single factor according to a predictive signal.

The underlying concept is substantially more general.

Different funds and portfolio managers employ different investment philosophies, risk constraints, and portfolio construction processes. A more sophisticated LLM integration could adapt the research-to-implementation workflow to each manager's existing infrastructure.

Potential applications include:

- Equity factor investing: Translating new factor research into systematic stock-selection and weighting methodologies.

- Global macro: Implementing economic forecasting models and translating their predictions into cross-asset positioning.

- Statistical arbitrage: Extracting predictive relationships and integrating them into existing signal-generation and portfolio optimization frameworks.

- Fixed income: Translating term-structure, relative-value, and credit research into portfolio positioning rules.

- Portfolio risk management: Incorporating research-driven constraints, hedging methodologies, and dynamic exposure adjustments.

In these applications, the LLM could extend beyond interpreting research to proposing and modifying components of the portfolio construction code itself, subject to deterministic testing and human oversight.

## Long-Term Vision

The ultimate goal is not to replace quantitative investment models with an LLM, but to automate the process of translating investment research into testable, economically evaluable strategies.

Portfolio managers would retain control over their investment objectives, risk constraints, and execution infrastructure. The LLM would reduce the manual effort required to interpret research, develop implementations, and integrate new ideas into existing systems.

Our Strategy Economics framework provides an additional layer of evaluation: rather than judging a signal solely by statistical significance or investment performance, we can estimate whether the expected incremental value of the strategy justifies its research and implementation costs.

The current demonstration is intentionally narrow. The broader opportunity is an extensible research-to-portfolio platform that can integrate with different investment strategies, optimization frameworks, and asset management workflows.