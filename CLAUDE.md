# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Spaceship Titanic** machine learning project for binary classification. The goal is to predict whether passengers were transported to another dimension during an incident on the Spaceship Titanic.

- **Problem Type:** Binary classification
- **Target Variable:** `Transported` (boolean)
- **Dataset:** 8,693 training samples, 4,277 test samples
- **Class Balance:** Perfectly balanced (50.36% vs 49.64%)

## Common Commands

### Data Analysis & Processing Pipeline

```bash
# Run comprehensive exploratory data analysis
# Generates statistics and 10 visualizations in plots/
python eda_analysis.py

# Clustering: segmentación automática de grupos de edad (Age) usando KMeans 1D
# Input: train9.csv → Output: train9_with_age_clusters.csv + age_cluster_summary.csv + plots/age_clustering_*.png
python cluster_age_groups.py --input train9.csv --max-k 10

# Feature engineering: consolidate 5 expense columns into TotalExpenses + HasExpenses
# Input: train.csv → Output: train_with_expense_features.csv
python create_expenses_features.py

# Split Cabin column (format "D/123/S") into Deck, Num, Side
# Input: train1.csv → Output: train2.csv
python split_cabin_column.py

# Move Transported column to final position
# Input: train2.csv → Output: train3.csv
python move_transported_to_end.py

# Convert Num column to Int64 (integer)
# Input: train3.csv → Output: train4.csv
python convert_num_to_int.py

# Convert Num, Age, and TotalExpenses to Int64 (integers)
# Input: train4.csv → Output: train5.csv
python convert_age_to_int.py

# Split PassengerId into Group and NumInGroup
# Input: train5.csv → Output: train6.csv
python split_passenger_id.py

# Add GroupSize column (number of people in each group)
# Input: train6.csv → Output: train7.csv
python add_group_size.py

# Analyze if group members have the same Transported value
# Generates: group_transported_analysis.csv
python analyze_group_transported.py
```

### Run Complete Pipeline

```bash
# Execute all transformations in sequence (after initial EDA)
python create_expenses_features.py && \
python split_cabin_column.py && \
python move_transported_to_end.py && \
python convert_num_to_int.py && \
python convert_age_to_int.py && \
python split_passenger_id.py && \
python add_group_size.py
```

### Environment Setup

```bash
# Install dependencies with conda
conda install -y seaborn matplotlib pandas numpy
```

## High-Level Architecture

### Data Transformation Pipeline

The project follows an 8-stage data pipeline:

```
train.csv (original)
    ↓
[EDA Analysis] → plots/ + EDA_REPORT.md
    ↓
[Expense Consolidation] → train1.csv
    ↓
[Cabin Split] → train2.csv
    ↓
[Column Reorder] → train3.csv
    ↓
[Type Conversion: Num] → train4.csv
    ↓
[Type Conversion: Num, Age, TotalExpenses] → train5.csv
    ↓
[PassengerId Split] → train6.csv
    ↓
[Group Size Feature] → train7.csv (FINAL)
```

**Stage 1 - EDA (`eda_analysis.py`):**
- Loads `train.csv` (8,693 rows × 14 columns)
- Generates descriptive statistics, correlation matrices, missing value analysis
- Creates 10 visualizations saved to `plots/`
- Outputs comprehensive analysis report

**Stage 2 - Expense Feature Engineering (`create_expenses_features.py`):**
- Consolidates 5 expense columns: `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`
- Creates:
  - `TotalExpenses`: Sum of all expenses
  - `HasExpenses`: Binary flag (0 = no spending, 1 = any spending)
- Removes individual expense columns
- Output: 11 columns (14 - 5 + 2)

**Stage 3 - Cabin Decomposition (`split_cabin_column.py`):**
- Splits `Cabin` column (format: "Deck/Number/Side", e.g., "F/123/S")
- Creates three features:
  - `Deck`: Letter A-G, T (8 unique values)
  - `Num`: Numeric cabin number (0-1894)
  - `Side`: P (Port) or S (Starboard)
- Removes original `Cabin` column
- Output: 13 columns (11 - 1 + 3)

**Stage 4 - Column Reordering (`move_transported_to_end.py`):**
- Moves target variable `Transported` from middle to final column
- Better data layout for ML frameworks expecting target at end
- Output: 13 columns

**Stage 5 - Type Conversion: Num (`convert_num_to_int.py`):**
- Converts `Num` from float64 to Int64 (nullable integer)
- Preserves 199 null values as `<NA>`
- Output: 13 columns

**Stage 6 - Type Conversion: Multiple Columns (`convert_age_to_int.py`):**
- Converts `Num`, `Age`, and `TotalExpenses` from float64 to Int64
- Ensures all numeric columns are properly typed as integers
- Preserves null values appropriately
- Output: 13 columns

**Stage 7 - PassengerId Decomposition (`split_passenger_id.py`):**
- Splits `PassengerId` (format: "0028_01") into:
  - `Group`: Group number (1-9280)
  - `NumInGroup`: Position within group (1-8)
- Removes original `PassengerId`
- Output: 14 columns (13 - 1 + 2)

**Stage 8 - Group Size Feature (`add_group_size.py`):**
- Adds `GroupSize`: Number of people in each group
- Position: After `Group`, before `NumInGroup`
- Enables analysis of family/group travel patterns
- Output: 15 columns (14 + 1) - **FINAL DATASET**

### Key Datasets

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| `train.csv` | 8,693 | 14 | Original training data |
| `train1.csv` | 8,693 | 11 | After expense consolidation |
| `train2.csv` | 8,693 | 13 | After cabin split |
| `train3.csv` | 8,693 | 13 | After column reordering |
| `train4.csv` | 8,693 | 13 | After Num type conversion |
| `train5.csv` | 8,693 | 13 | After Num/Age/TotalExpenses type conversion |
| `train6.csv` | 8,693 | 14 | After PassengerId split |
| `train7.csv` | 8,693 | 15 | After GroupSize added |
| `train8.csv` | 8,693 | 15 | After Name → Surname extraction |
| `train9.csv` | 8,693 | 15 | **FINAL** - Surname in Surname_Group format |
| `test.csv` | 4,277 | 13 | Test data (no target) |

### Feature Schema (train9.csv - FINAL)

```
1.  Group           (int64)    - Group number (1-9280), 6217 unique groups
2.  NumInGroup      (int64)    - Position within group (1-8)
3.  GroupSize       (int64)    - Size of group (1-8 people), mean: 2.04
4.  HomePlanet      (object)   - Earth, Europa, Mars
5.  CryoSleep       (object)   - True/False (strongest predictor: 76% vs 38%)
6.  Deck            (object)   - A-G, T (8 unique values)
7.  Num             (Int64)    - Cabin number 0-1894 (nullable integer)
8.  Side            (object)   - P (Port) or S (Starboard)
9.  Destination     (object)   - TRAPPIST-1e, 55 Cancri e, PSO J318.5-22
10. Age             (Int64)    - 0-79 years (nullable integer)
11. VIP             (object)   - True/False (only 2.3% are VIP)
12. Surname         (object)   - Surname_Group format (e.g., "Upead_16"), 6387 unique
13. TotalExpenses   (Int64)    - Sum of all expenses $0-$35,987 (nullable integer)
14. HasExpenses     (int64)    - 0 or 1 (42% spent nothing)
15. Transported     (bool)     - TARGET VARIABLE
```

## Important Findings from EDA

**Strongest Predictors:**
- **CryoSleep:** Passengers in CryoSleep have 76% transport rate vs 38% awake
- **Age Group:** Children (0-12) have 66.9% transport rate vs ~48% for adults
- **HomePlanet:** Europa passengers have slightly higher transport rate (~56%)

**Data Quality:**
- Missing values: < 3% in all columns (excellent quality)
- No significant outliers requiring removal
- Balanced classes (no resampling needed)

**Spending Patterns:**
- 42% of passengers spent nothing (HasExpenses = 0)
- Europa passengers spend 5x more than Earth ($3,452 vs $673 average)
- Total expenses median: $716, mean: $1,441

**Group Travel Patterns:**
- 6,217 unique groups (55% solo travelers, 45% in groups)
- Group sizes range from 1-8 people, mean: 2.04
- Distribution: 55% solo, 19% pairs, 12% triplets, 14% groups of 4+

**Critical Finding: Transport is INDIVIDUAL, Not Group or Family-Based**

Three comprehensive analyses confirm transport is an individual phenomenon:

1. **Groups Analysis** (same Group):
   - Only **43.56%** have same outcome (2+ people)
   - Larger groups: 8 people = 0% consistency (always mixed)

2. **Surnames Analysis** (same Surname):
   - Only **23.43%** have same outcome (2+ people)
   - Families travel in separate groups (91% distributed)
   - Families of 10+: 0% consistency

3. **Nuclear Families Analysis** (same Surname + Group):
   - Only **46.25%** have same outcome (2+ people)
   - Marginal improvement over groups alone (+2.69%)
   - Families of 8: 0% consistency (always mixed)

**Key Implications:**
- **Group ID is NOT a predictor** of Transported status
- **Surname is NOT a predictor** (very low consistency)
- **Surname_Group is NOT a predictor** (barely better than groups)
- **GroupSize may be useful** (clear trend: smaller = more consistent)
- **Transport depends on INDIVIDUAL characteristics**, not group/family membership

## Project Structure

```
.
├── eda_analysis.py                    # Stage 1: EDA with visualizations
├── create_expenses_features.py        # Stage 2: Expense consolidation
├── split_cabin_column.py              # Stage 3: Cabin decomposition
├── move_transported_to_end.py         # Stage 4: Column reordering
├── convert_num_to_int.py              # Stage 5: Num type conversion
├── convert_age_to_int.py              # Stage 6: Num/Age/TotalExpenses type conversion
├── split_passenger_id.py              # Stage 7: PassengerId decomposition
├── add_group_size.py                  # Stage 8: Add GroupSize feature
├── extract_surname.py                 # Stage 9: Extract surname from Name
├── add_group_to_surname.py            # Stage 10: Add Group to Surname (Surname_Group)
├── analyze_group_transported.py       # Analysis: Group vs Transported relationship
├── analyze_surname_transported.py     # Analysis: Surname vs Transported relationship
├── analyze_family_group_transported.py # Analysis: Nuclear families vs Transported
├── train.csv                          # Original training data
├── train1.csv                         # After expense features
├── train2.csv                         # After cabin split
├── train3.csv                         # After column reordering
├── train4.csv                         # After Num type conversion
├── train5.csv                         # After Num/Age/TotalExpenses type conversion
├── train6.csv                         # After PassengerId split
├── train7.csv                         # After GroupSize added
├── train8.csv                         # After Name → Surname extraction
├── train9.csv                         # FINAL: Surname_Group format (Apellido_Grupo)
├── test.csv                           # Test data (no target)
├── sample_submission.csv              # Submission template
├── EDA_REPORT.md                      # Comprehensive EDA report (Spanish)
├── group_transported_analysis.csv     # Group vs Transported analysis results
├── surname_transported_analysis.csv   # Surname vs Transported analysis results
├── family_nuclear_transported_analysis.csv # Nuclear families analysis results
└── plots/                             # 10 visualization PNGs
    ├── 01_transported_distribution.png
    ├── 02_missing_values.png
    ├── 03_categorical_distributions.png
    ├── 04_age_distribution.png
    ├── 05_expenses_distribution.png
    ├── 06_correlation_matrix.png
    ├── 07_transported_by_homeplanet.png
    ├── 08_transported_by_cryosleep.png
    ├── 09_age_by_transported.png
    └── 10_total_expenses.png
```

## Technology Stack

- **Language:** Python 3
- **Data Science Libraries:**
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computing
  - `matplotlib` - Visualization
  - `seaborn` - Statistical visualizations
- **Environment:** Conda

## Next Steps (Model Development)

The EDA and data preprocessing are complete. To build predictive models:

1. **Use `train9.csv` as the final training dataset** (15 columns, fully processed with Surname_Group)

2. **Recommended models** based on EDA insights:
   - Random Forest (handles categorical variables well)
   - Gradient Boosting (XGBoost, LightGBM) - likely best performer
   - Neural Networks with embeddings for categorical features
   - Ensemble of multiple models

3. **Feature importance priorities** (based on analysis):
   - **High:** CryoSleep, Age, HomePlanet, GroupSize
   - **Medium:** Deck, Destination, TotalExpenses, HasExpenses, Side
   - **Low:** Group, NumInGroup (groups don't transport together)
   - **Minimal:** VIP, Name (unless extracting surnames)

4. **Handle missing values** (~2% per column):
   - Impute Age based on groups (e.g., median Age by HomePlanet)
   - For expenses: 0 is a valid value, not missing
   - Categorical: Create "Unknown" category or use mode
   - Num/Deck/Side: Impute based on other cabin features

5. **Additional feature engineering** ideas:
   - Extract surnames from Name column (family patterns)
   - Create age bins/categories (children, teens, adults, seniors)
   - Interaction features: CryoSleep × HomePlanet, Age × GroupSize
   - Spending ratio: TotalExpenses / Age (spending per year of life)
   - Binary flags: IsSolo (GroupSize == 1), IsChild (Age < 13)
   - Deck category grouping (luxury vs standard decks)

6. **Cross-validation strategy**:
   - Use stratified K-fold (preserve class balance)
   - Consider GroupKFold to avoid data leakage (keep groups together)
   - 5-10 folds recommended

7. **Evaluation metrics**:
   - Primary: Accuracy (balanced dataset)
   - Secondary: F1-score, AUC-ROC, Precision/Recall

8. **Important considerations**:
   - Groups are NOT transported together → don't use group-level features naively
   - GroupSize may be useful but Group ID is not
   - CryoSleep is the strongest single predictor (38% → 76% difference)
   - Children have much higher transport rates (focus on age interactions)

## Documentation

**`EDA_REPORT.md`** - Comprehensive exploratory data analysis (Spanish):
- Complete statistics for all variables
- Distribution analysis with visualizations
- Cross-tabulations with target variable
- Feature engineering recommendations
- Model selection guidance

**`group_transported_analysis.csv`** - Group analysis results:
- Per-group breakdown of transported vs not transported
- Identifies groups with consistent vs mixed outcomes
- Enables further analysis of group-level patterns

## Key Insights Summary

1. **CryoSleep is the strongest predictor** (76% vs 38% transport rate)
2. **Children are transported at much higher rates** (67% vs 48% for adults)
3. **Transport is INDIVIDUAL, not group/family-based:**
   - Groups (same Group): 43.56% consistency
   - Surnames (same family): 23.43% consistency
   - Nuclear families (same surname + group): 46.25% consistency
   - **Conclusion:** Group/family membership does NOT predict transport
4. **Larger groups/families NEVER transported together** - 8-person groups: 0% consistency
5. **Europa passengers spend 5x more** than Earth passengers ($3,452 vs $673)
6. **42% of passengers spent nothing** on amenities
7. **Dataset is perfectly balanced** (50.36% vs 49.64%) - no resampling needed
8. **Data quality is excellent** - <3% missing values across all columns
9. **GroupSize may be useful** - clear inverse relationship with transport consistency
10. **Surname_Group is NOT a predictor** - barely better than groups alone (+2.69%)
