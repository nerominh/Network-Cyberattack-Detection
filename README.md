# Network Cyberattack Detection

## Run the EDA notebook in VS Code

Use Python 3.14 and open the repository folder in VS Code with the **Python**
and **Jupyter** extensions installed.

Create the project environment from the repository root (macOS/Linux):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Open `notebook/eda.ipynb`, click **Select Kernel** in the upper right, choose
**Python Environments**, then select **.venv**. Its interpreter is
`.venv/bin/python` inside this repository. If it is not listed, run
**Python: Select Interpreter** from the Command Palette, choose **Enter interpreter
path**, and select that file; then select it as the notebook kernel.

Run the notebook cells in order. The workspace settings use `notebook/` as the
working directory, matching the notebook's `../data` and `../figures` paths.

Download the CSVs into the ignored `data/` directory from the repository root:

```sh
.venv/bin/python scripts/download_dataset.py
```

The script downloads only the two partition CSVs from
`mrwellsdavid/unsw-nb15` on Kaggle, checks their dimensions and matching schemas,
and assigns names by row count rather than trusting Kaggle's swapped filenames.
Existing complete local files are validated and reused. Then use **Run All** in
the notebook. The resulting files are:

- `UNSW_NB15_training_set.csv`: the official training data, 175,341 rows.
- `UNSW_NB15_testing_set.csv`: the official test data, 82,332 rows.

Check row counts rather than trusting downloaded filenames; the notebook notes
that some downloads have these names swapped. The CSVs and `.venv` are not
committed. `requirements.txt` records the package versions used for this setup.
