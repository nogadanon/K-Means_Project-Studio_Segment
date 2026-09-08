import os
from tkinter import Tk, filedialog

import requests

API_KEY = "fd5674b02970431f9edf3fb75cc1fd2b.AQiHkwqai524PWV5fhQI5mNw"  # put your key here instead of ollama_your_key_here


def get_file_name():
    """Open a file picker restricted to CSV/Excel files and return the filename without its extension."""
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="Select a CSV or Excel file",
        filetypes=[("CSV and Excel files", "*.csv *.xlsx *.xls"), ("All files", "*.*")],
    )
    root.destroy()

    if not file_path:
        raise ValueError("No file selected.")

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in (".csv", ".xlsx", ".xls"):
        raise ValueError(f"Unsupported file type: {ext}. Please choose a CSV or Excel file.")

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    return file_name


''' Prompt 1:
Write a function that accepts a CSV or Excel file from the user and returns the filename without the extension.

Write another function that accepts the values ​​k_min, k_max, and k from the user.
'''

def get_k_values():
    """Prompt the user for k_min, k_max, and k, validate them, and return the three integers."""
    while True:
        try:
            k_min = int(input("Enter k_min (minimum number of clusters, e.g. 2): "))
            k_max = int(input("Enter k_max (maximum number of clusters, e.g. 10): "))
            k = int(input("Enter k (number of clusters to use for the final model): "))
        except ValueError:
            print("Please enter valid integers.\n")
            continue

        if k_min < 2:
            print("k_min must be at least 2.\n")
            continue
        if k_max < k_min:
            print("k_max must be greater than or equal to k_min.\n")
            continue
        if not (k_min <= k <= k_max):
            print(f"k must be between k_min ({k_min}) and k_max ({k_max}).\n")
            continue

        return k_min, k_max, k

response = requests.post(
    "https://ollama.com/api/chat",
    headers={
        "Authorization": f"Bearer {API_KEY}"
    },
    json={
        "model": "gpt-oss:120b",
        "messages": [
            {
                "role": "user",
                "content": "tell me a joke"
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.7
        }
    }
)

data = response.json()

print(data["message"]["content"])

'''
Prompt 2:
REST API backing Studio_Segment_WEB.html.

Upload a file, preview it, compute the K-Means elbow curve, create clusters, generate
cluster names/descriptions via an LLM, and download the clustered result -- mirroring
the logic in the read-only reference notebook:
C:\\Users\\Lior\\Studio_Segment_K-Means_Project.ipynb (never modified by this file).
'''

'''
 Prompt 3:

Open an html file called Studio_Segment_WEB and build a website in it that the user can upload a csv file to. If the user uploads another file, try converting the file to csv.
All operations are performed only according to the existing code in the jupiter file called Studio_Segment_K-Means_Project, located in the users/Lior folder. Never change anything in this file. Confirm that you find the file. Only after you find the file, continue building the website according to the following steps:
The user uploads a file. The file_name variable will change from None to the name of the new file without the extension (.csv), but without changing the original file saved in the users/Lior folder.
Step 1: SCV Table
The table that the user uploaded must be displayed.
See an example image for step 1.

**In Step 1, display the `numeric_features` and `categorical_features` lists based on the code.
Allow the user to move features from the `numeric_features` list to the `categorical_features` list and to delete features from both lists.
Include an "Apply" button to update the table and the lists. Note that subsequent steps must use the updated lists confirmed by the user

Step 2: WCSS (Elbow)
The user will select k_min, k_max for the variables defined k_min, k_max in the code. The system will calculate WCSS for all K values ​​in the required range according to the existing code. The resulting elbow graph should be displayed.
See sample image for step 2 .

Step 3: Choose k and create clusters
The user chooses k for the variable k in the code. The table step_3 should be displayed. See sample image for step 3 .

Step 4: Generate group name + description (LLaMA)
Using LLM in the main file, fill in the columns in DataFrame 'step_3': name, description . For this, the feature names according to the "features" list in the code.
In Step 4 of the website, before the currently displayed table, add two additional tables: `table_1` and `table_2` (corresponding to the dataframes defined by these names in the "Studio_Segment_K-Means_Project" code).
For `table_1`, use the title: "Number of observations per cluster and mean values".
For `table_2`, use the title: "Common values ​​for categorical features per cluster".

Step 5:
Allow the user to download the updated file, according to the code: df.to_csv(file_name_clustered) . Display the new file name (in the file_name_clustered variable).

To receive a file and data from the user, use the functions:
get_file_name(),
get_k_values()

REST api  is located in the file api.py  .
'''

