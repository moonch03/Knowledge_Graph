# Graph Analysis Project

This project focuses on graph network analysis and GraphRAG (Retrieval-Augmented Generation) using Neo4j and OpenAI. It includes analysis of Montreal Gangs and Noordin datasets.

## Structure

- `apps/`: Flask applications for interactive visualization and analysis.
  - `montreal/`: Analysis of the Montreal Gangs network.
  - `noordin/`: Analysis of the Noordin Top terrorist network.
- `notebooks/`: Jupyter Notebooks for data processing and metric calculation.
- `scripts/`: Utility scripts for database testing and indexing.
- `images/`: Exported network visualizations and metric plots.
- `lib/`: Local copies of JavaScript/CSS libraries.

## Setup

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your `.env` file with the following variables:
   ```env
   NEO4J_URI=bolt://localhost:7687
   NEO4J_PASSWORD=your_password
   OPENAI_API_KEY=your_api_key
   ```
4. Run the applications:
   ```bash
   python apps/montreal/app.py
   # or
   python apps/noordin/app.py
   ```
