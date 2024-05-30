## Explanation
I had a directory of JSON files about products. They didn't contain features, but only had a feature-key called "description" with the features explained in a string. I extracted those features from the string using an LLM API Gemini-Pro from the products. I only did it with a given template only for 7 categories out of 70+ categories.
I did it only for those categories because my project was to be able to find products from a natural language question. I did it by using an LLM to create the query. 

## Some Guidance Through the Repo

- **The complete notebook of the app can be found in "products_categories_fixed/0_complete/all_included"**

- **The extraction of features can be found in "products_categories_fixed/<category_name>/<category_name>_new_template", like "products_categories_fixed/cables_and_dividers_new_template"**

- **Some original products for preference to compare to a featured output can be found in "products_categories_fixed/products_categories_for_extraction"**

## Some models used Explained

The original notebook app also uses a translator to translate the question from Macedonian to English to pass it onto the selected AI's API, and a zero-shot classifier to guess the category of products for which the user is asking.
- Zero-Shot classifier model: "facebook/bart-large-mnli"
- Translator: "facebook/nllb-200-distilled-600M"

Models used with APIs:
- Model: "gemini-pro" at "temperature=0.1" through Google's API for feature extraction and SQL query generation
- Model: "mistralai/Mistral-7B-Instruct-v0.3" through HuggingFace's Inference Endpoints API at "temperature=0.1" for SQL query generation
- Model: "claude-3-sonnet-20240229" through Antropic's API at "temperature=0" for SQL query generation. Also tested out feature extraction with the notebook "trying_with_claude.ipynb" in the folder "products_opus_testing" which didn't entirely go bad
