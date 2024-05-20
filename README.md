# LLM Resume Selection Recommendation

The aim of this project is to create a model that can process a number of resumes (CVs), query them, and return the individuals that match specified criteria most closely. To solve this problem, I will be using the BERT model and Vector Querying.

## Data

The project uses a publicly available dataset from Kaggle:

- [Resume Dataset on Kaggle](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)

This dataset contains approximately 150 rows and has two columns:

- **Job Category**: Categories such as Testing, HR, Python Developer.
- **Raw text data**: The resume/CV text.

The job category will be used to judge whether the model is working correctly.

## Approach

I will be using a **BERT tokenizer** to vectorise the text data. The BERT model provides a semantic representation of the text in vector space. This will then be queried, and using **Cosine Similarity**, a value indicating how related the query and the text are will be calculated.

$$
\text{CosineSimilarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}
$$

For more information on how encoder transformers work, please see [here](https://github.com/jotren/Machine-Learning-Teaching-Material/blob/main/4%29%20lessons/Transformer%20Encoder%20Maths.md).

<img src="./images/CosSim.jpeg" alt="alt text" width="400"/> 

## Testing

After several iterations, I found that the initial model was providing very high similarity scores for unrelated jobs simply because the text had fewer words. This was due to:

- Comparing vectors with fewer words/features can sometimes produce higher similarity scores as they have fewer dimensions to normalise over.
- This phenomenon was exacerbated through normalsation.

To correct for this, the BM25 query equation was used:

$$
\text{BM25}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k1 + 1)}{f(q_i, D) + k1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}
$$

$$
\begin{align*}
f(q_i, D) & \text{ is the term frequency of term } q_i \text{ in document } D. \\
|D| & \text{ is the length of document } D. \\
\text{avgdl} & \text{ is the average document length in the corpus.} \\
\text{IDF}(q_i) & \text{ is the inverse document frequency of term } q_i. \\
k1 & \text{ and } b \text{ are parameters for BM25 (commonly set to 1.2 or 2.0 for } k1 \text{ and 0.75 for } b).
\end{align*}
$$

This equation essentially normalises for the number of words in the document. Implementing this led to a massive improvement in results.

## Deployment

The application is deployed using Flask. For production, I may use a callable function to avoid Dockerizing the application, which can take up significant memory, whereas callable functions are free. The Flask application has two main functions:

```python

app = Flask(__name__)

# Global variables to store embeddings and BM25 model
df = pd.DataFrame()
embeddings_matrix = None
bm25 = None

@app.route('/create_embeddings', methods=['POST'])
def create_embeddings():
    global df, embeddings_matrix, bm25

    data = request.get_json()

    # Check if the data is provided
    if not data or 'resumes' not in data:
        return jsonify({"error": "No data provided"}), 400

    # Convert data to DataFrame
    df = pd.DataFrame(data['resumes'])

    # Preprocess the DataFrame
    df['cleaned_resume'] = df['resume'].apply(clean_text)
    df['tokenized_data'] = df['cleaned_resume'].apply(lambda x: tokenize(x))
    df['embeddings'] = df['tokenized_data'].apply(lambda row: create_embeddings(row))

    # Stack embeddings into a matrix
    embeddings_matrix = np.vstack(df['embeddings'].values)

    # Tokenize the resumes for BM25
    df['tokenized_resume'] = df['resume'].apply(preprocess_for_bm25)

    # Create a BM25 object
    bm25 = BM25Okapi(df['tokenized_resume'].tolist())

    return jsonify({"message": "Data uploaded successfully"}), 200

@app.route('/calculate_similarity', methods=['POST'])
def calcualte_similarity_score():
    global df, embeddings_matrix, bm25

    # Check if data has been uploaded
    if df.empty:
        return jsonify({"error": "No data uploaded"}), 400

    query = request.json['query']

    results = get_combined_similarity(df, query, embeddings_matrix, bm25, df['embeddings'])

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Separating functions allows better integration with the backend. When files are uploaded into the system, the API will return embeddings that can be saved in a database. When querying those embeddings, user input will be sent along with the embeddings from the database.

## API Structures
### Create Embeddings

Endpoint: http://localhost:5000/create_embeddings

Request:

```JSON
{
    "resumes": [
        {"resume": "Experienced data scientist with expertise in Python, machine learning, and SQL."},
        {"resume": "Software engineer with a strong background in Java and web development."},
        {"resume": "Data analyst proficient in R and data visualization."},
        {"resume": "Machine learning engineer with deep learning experience."},
        {"resume": "Backend developer skilled in Python and database management."}
    ]
}
```
Response:

```JSON

{
    "embeddings": [
        [
            [
                -0.16804875433444977,
                0.10804363340139389,
                0.5985521078109741,
                -0.1282108575105667,
                0.3644883930683136,
                0.12319885939359665,
                0.09499166905879974,
                -0.013103890232741833,
                -0.2396308034658432
                ...
            ]
        ]
    ]
}
```

### Calculate Similarity

Endpoint: http://localhost:5000/calculate_similarity

Request:

```JSON

{
    "query": "Python machine learning sklearn SQL database data science database coding programming",
    "embeddings": [
        [
            [
                -0.16804875433444977,
                0.10804363340139389,
                0.5985521078109741,
                -0.1282108575105667,
                0.3644883930683136,
                0.12319885939359665,
                0.09499166905879974,
                -0.013103890232741833,
                -0.2396308034658432
                ...
            ]
        ]
    ]
}

```

Response:



```JSON
{
    "similarity_scores": [
        0.3693714805199362,
        1.818327551786447,
        1.2359929124287832,
        -0.7138150048762963,
        2.0062221880332256,
        2.062000668289734,
        1.9300368053852364,
        0.645678423787,
    ]
}
```
### Key Takeaways

Vector querying is prone to provide high cosine similarity scores for text that have a low number of tokens. There is not enough information to provide a meaningful vector. Need to use BM_25 search algorithmn, which is employed by Google, to get it working.





