```markdown
# Turkish Text Classification Project

This repository hosts the code and resources for a machine learning project aimed at classifying Turkish texts into categories: not offensive, sexist, racist, profanity, or insult. The classification is multi-labeled and utilizes a dataset of 80,000 hand-labeled rows of Turkish data.

## Project Structure

```plaintext
project/
│
├── data/
│   ├── external/                    # Third-party data sources
│   │   └── turkish-word-list-master.zip
│   ├── processed/                   # Data prepared for analysis
│   └── raw/                         # Original, unprocessed data files
│
├── models/
│   ├── BERTModels/                  # BERT models for text classification
│   │   ├── model1_v1.pth            # BERT model version 1 for classify offensive or not
│   │   └── model2_v1.pth            # BERT model version 1 for classify for profanity, insult, sexist, racist
│   └── zemberek/                    # Turkish NLP tool for typo correction
│
├── notebooks/
│   ├── Lookup.ipynb                 # Initial project walkthrough and setup
│   ├── OffensivePrediction.ipynb    # Predicting text offensiveness
│   ├── TypoCorrection.ipynb         # Text correction tryouts
│   ├── TypoCorrection2.ipynb        # Text correction tryouts 2
│   └── OffNotOffClassification.ipynb # Predicting whether the text is offensive or not
│
├── streamlit_app/
│   └── app.py                       # Streamlit application for model demonstration
│
├── README.md
└── requirements.txt                 # Project dependencies
```

## Download Links

- **Model1_v1**: [Download here](https://drive.google.com/file/d/1rDB-s9XewTmesH5C6wmrdd2Wff7CcO0F/view?usp=sharing)
- **Turkish Word List**: [Download here](https://drive.google.com/file/d/1LNJVF3Dbky3X6VymeLorhM_bHnjAgvvj/view?usp=sharing)
- **Zemberek**: [Download here](https://drive.google.com/file/d/18GPMUXwpBJx2GeyN1ZT7DNc6pbUSB0mL/view?usp=sharing)
- **Model2_v1**: [Download here](https://drive.google.com/file/d/15u5M9V4e8Jplpi2f5rklsYx_FwkWNJAY/view?usp=sharing)

## Getting Started

1. Clone the repository to your local machine.
2. Install the necessary Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start exploring the project with the `Lookup.ipynb` Jupyter notebook for an overview and initial analysis.

## Contributing

Contributions to improve the project are welcome. Please ensure to follow the existing code structure and update the README accordingly if you add or modify functionalities.

## License

This project is open-source and available under the MIT License.

## Contact

For any queries or further details, please email us at kmlshnbusiness@gmail.com.
```


- Make sure there is an empty line before and after each code block.
- Use "```plaintext" to specify non-highlighted text in your directory structure block if you're not highlighting specific syntax, which helps with clarity.
- Ensure list items and code blocks inside them are properly indented and formatted.
