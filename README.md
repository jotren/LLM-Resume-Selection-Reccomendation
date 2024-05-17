# LLM Resume Selection Reccomendation

Aim of this project is to create a model that can absorb a number of CVs, query the resumes and then return the individuals that match the criteria most closely. To solve this problem I will be using BERT model and Vector Querying. This 

### Data

Will be using the publicly available dataset from kaggle: 

- https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset

This dataset contains ~150 rows and has two columns:

- __Job Category__: Testing, HR, Python Developer
- __Raw text data__: Resume/CV text

Will use the category to make a judgement on whether the model is working.

### Approach

Will be using a __BERT tokeniser__ to vector query the text data. The BERT model will provide a semantic representation in vector space of the text. This will then be queried and using __Cosine Similarity__, a value for how related the query and the text is will be calcualted.

For information on how encoder transformers work, please see [here](https://github.com/jotren/Machine-Learning-Teaching-Material/blob/main/4%29%20lessons/Transformer%20Encoder%20Maths.md).

<img src="./images/CosSim.jpeg" alt="alt text" width="400"/> 

### Result

### Next Steps