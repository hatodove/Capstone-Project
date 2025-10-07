Capstone Project - README

Project Overview
This repository contains all files and notebooks used in my Data Analytics Capstone Project. The project moves through stages of data ingestion, database design, analysis, and reporting. Each file has a clear role in preparing, transforming, analyzing, and presenting insights.

Files

Deliverable 1.1.2 Data Source Links.txt
Links to all the original sources of the raw datasets
------------------------------------------------------------------------------------------

Deliverable 1.1.2 Data.ipynb
Purpose:
This notebook handles the data ingestion and database setup. It performs the following tasks:

Reads all raw input data from CSV files

Cleans and organizes the data for consistency

Structures the data into a star schema (fact and dimension tables)

Saves the transformed data into a SQLite database called analytics_panel.sqlite

Why this matters:
The star schema design makes queries faster and easier for downstream analysis. It also provides a reliable single source of truth for all subsequent notebooks in this project.

Output:
analytics_panel.sqlite – A local SQLite database containing the transformed data in star schema format.
------------------------------------------------------------------------------------------
TABLE REVIEW (extra-not a deliverable)

Politics_and_Safety_Analysis.ipynb
GDP_Analysis.ipynb
Marriage_Analysis.ipynb
Employment_Analysis.ipynb

Purpose:
Each of these notebooks analyzes fertility against a different factor (politics and safety, GDP, marriage, or employment). They all follow the same process:

Review dataset coverage

Run between-country and within-country comparisons

Test lag effects, partial correlations, and first differences

Calculate Pearson and Spearman correlations with p-values

Evaluate the null hypothesis (H0) with a short conclusion
------------------------------------------------------------------------------------------

Deliverable 1.6.3 Analysis.ipynb

Purpose:
This notebook compiles and compares the results from all previous analyses. It brings together the findings from the politics and safety, GDP, marriage, and employment notebooks to evaluate fertility patterns more broadly. The goal is to identify consistent relationships, highlight differences across factors, and provide an integrated view of the overall results.
------------------------------------------------------------------------------------------

Deliverable 1.6.1 Technical Report (FINAL REPORT)
