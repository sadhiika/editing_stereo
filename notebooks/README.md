# StereoWipe Analysis Notebooks

This directory contains comprehensive Jupyter notebooks for analyzing the StereoWipe benchmark results and generating publication-ready figures and tables.

## Notebooks Overview

### 1. Data Exploration (`01_data_exploration.ipynb`)
**Purpose**: Comprehensive exploration and quality assessment of the StereoWipe dataset

**Key Features**:
- Data loading and structure analysis
- Stereotype rate visualization across categories
- Statistical distributions and patterns
- Data quality checks and validation
- Export processed data for subsequent analysis

**Outputs**:
- `data/processed_evaluation_data.csv`: Clean dataset for analysis
- `data/data_exploration_summary.json`: Comprehensive summary report
- Various visualizations and statistical summaries

### 2. Human-LLM Agreement Analysis (`02_human_llm_agreement.ipynb`)
**Purpose**: Analyze agreement between human annotations and LLM judge evaluations

**Key Features**:
- Agreement metrics (Cohen's kappa, correlation coefficients)
- Disagreement pattern analysis
- Category-wise agreement variation
- Statistical significance testing
- Practical recommendations for system improvement

**Outputs**:
- `data/human_llm_comparison_data.csv`: Detailed comparison data
- `data/human_llm_agreement_analysis.json`: Statistical analysis results
- `data/human_llm_agreement_summary.json`: Executive summary

### 3. Arena Analysis (`03_arena_analysis.ipynb`)
**Purpose**: Arena-style model comparison using Elo ratings and pairwise battles

**Key Features**:
- Pairwise model comparisons
- Elo rating calculations and rankings
- Human vs. objective preference analysis
- Statistical significance testing
- Ranking stability analysis

**Outputs**:
- `data/arena_battles.csv`: Individual battle results
- `data/human_preference_battles.csv`: Human preference battle data
- `data/arena_analysis_summary.json`: Complete tournament analysis

### 4. Bias Categories Deep Dive (`04_bias_categories.ipynb`)
**Purpose**: In-depth analysis of different bias categories and their interactions

**Key Features**:
- Category-wise performance metrics
- CSSS and WOSI metric calculations
- Cross-category statistical comparisons
- Category weight optimization
- Intersectional bias analysis

**Outputs**:
- `data/category_analysis_data.csv`: Category-specific performance data
- `data/category_analysis_report.json`: Comprehensive category analysis
- `data/weight_optimization_results.json`: Weight optimization findings

### 5. Publication Figures (`05_paper_figures.ipynb`)
**Purpose**: Generate high-quality, publication-ready figures and tables

**Key Features**:
- Professional visualization standards (300 DPI, colorblind-friendly)
- Main results figures for research papers
- Statistical summary tables
- Appendix materials and supplementary analyses
- LaTeX integration support

**Outputs**:
- `figures/`: Directory containing all publication figures (PNG and PDF)
- `figures/publication_summary.json`: Complete figure and table catalog
- `figures/latex_figure_references.tex`: Ready-to-use LaTeX references
- Multiple CSV tables for publication use

## Usage Instructions

### Prerequisites
```bash
pip install jupyter pandas numpy matplotlib seaborn scipy scikit-learn networkx
```

### Running the Analysis
1. **Sequential Execution**: Run notebooks in order (01 → 02 → 03 → 04 → 05) for complete analysis
2. **Individual Analysis**: Each notebook can be run independently for specific insights
3. **Data Dependencies**: Later notebooks use outputs from earlier ones when available

### Execution Order
```
01_data_exploration.ipynb          # Foundation analysis
    ↓
02_human_llm_agreement.ipynb       # Agreement analysis
    ↓
03_arena_analysis.ipynb            # Model comparison
    ↓
04_bias_categories.ipynb           # Category deep dive
    ↓
05_paper_figures.ipynb             # Publication materials
```

## Key Metrics Explained

### Stereotype Rate (SR)
- **Definition**: Percentage of responses flagged as stereotypical
- **Range**: [0, 1], where 0 = no stereotypes, 1 = all stereotypical
- **Interpretation**: Lower values indicate better performance

### Stereotype Severity Score (SSS)
- **Definition**: Average severity rating across all responses
- **Range**: [0, 5], where 0 = no stereotypes, 5 = maximum severity
- **Interpretation**: Lower values indicate less severe stereotypical content

### Conditional Stereotype Severity Score (CSSS)
- **Definition**: Average severity rating for stereotypical responses only
- **Range**: [3, 5], conditional on stereotypical classification
- **Interpretation**: Measures severity given that stereotyping occurred

### Weighted Overall Stereotyping Index (WOSI)
- **Definition**: Category-weighted composite score combining SR and SSS
- **Formula**: WOSI = Σ(w_i × (α × SR_i + β × SSS_i)) / Σ(w_i)
- **Parameters**: α = 0.6 (SR weight), β = 0.4 (SSS weight)
- **Interpretation**: Comprehensive performance measure, lower is better

## Data Directory Structure

```
data/
├── processed_evaluation_data.csv           # Clean dataset
├── data_exploration_summary.json           # Data quality report
├── human_llm_comparison_data.csv          # Agreement analysis data
├── human_llm_agreement_analysis.json      # Statistical tests
├── arena_battles.csv                      # Battle results
├── arena_analysis_summary.json            # Tournament summary
├── category_analysis_data.csv             # Category performance
├── category_analysis_report.json          # Category insights
└── weight_optimization_results.json       # Weight optimization
```

## Figures Directory Structure

```
figures/
├── figure1_framework_overview.png          # Main framework figure
├── figure2_main_results.png               # Results comparison
├── figure3_human_llm_agreement.png        # Agreement analysis
├── figure4_category_analysis.png          # Category deep dive
├── appendix_a1_methodology.png            # Methodology details
├── appendix_a2_statistical_analysis.png   # Statistical tests
├── appendix_a3_error_analysis.png         # Error patterns
├── table1_overall_performance.csv         # Performance summary
├── table2_category_performance.csv        # Category breakdown
├── table3_statistical_tests.csv           # Significance tests
├── table4_human_agreement.csv             # Agreement metrics
├── publication_summary.json               # Complete catalog
└── latex_figure_references.tex            # LaTeX integration
```

## Technical Specifications

### Figure Quality Standards
- **Resolution**: 300 DPI minimum for publication
- **Formats**: PNG (for presentations) and PDF (for print)
- **Color Palette**: Colorblind-friendly throughout
- **Typography**: Times New Roman (serif) for academic publications
- **Sizing**: Standard 10x8" for single figures, 14x10" for complex layouts

### Statistical Rigor
- **Significance Testing**: Multiple comparison corrections applied
- **Effect Sizes**: Cohen's d and other effect size measures included
- **Confidence Intervals**: Bootstrap and parametric CIs provided
- **Reproducibility**: All random seeds set for consistent results

## Troubleshooting

### Common Issues
1. **Missing Dependencies**: Install all required packages from prerequisites
2. **Data Not Found**: Run notebooks in sequential order to generate required data
3. **Memory Issues**: Large datasets may require increased memory allocation
4. **LaTeX Errors**: Set `text.usetex = False` in matplotlib if LaTeX unavailable

### Performance Optimization
- **Parallel Processing**: Some analyses can be parallelized for speed
- **Memory Management**: Clear large variables between analyses
- **Caching**: Results are cached where possible to avoid recomputation

## Contributing

When adding new analyses:
1. Follow the existing notebook structure and naming conventions
2. Include comprehensive documentation and explanations
3. Generate appropriate outputs in the `data/` directory
4. Add publication-quality figures to the `figures/` directory
5. Update this README with new notebook descriptions

## Citation

If you use these notebooks in your research, please cite:
```bibtex
@misc{stereowipe2024,
  title={StereoWipe: A Comprehensive Framework for Stereotype Evaluation in Large Language Models},
  author={[Authors]},
  year={2024},
  note={Analysis notebooks and evaluation framework}
}
```