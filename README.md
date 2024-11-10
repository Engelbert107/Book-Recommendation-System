# Book Recommendation System

We used the [Goodreads-books](https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks) dataset from [kaggle](https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks) and developp a book recommendation system that suggests relevant books to users based on a text query. Leveraging natural language processing and machine learning, we analyzed key book features and created a recommendation model, which we deployed as an interactive web application using Streamlit. Users can search by entering a title, description, or keywords, and the system will provide tailored book recommendations. Additionally, users can adjust the number of recommendations to fit their needs. The app outputs a list of relevant books with key details like title, authors, and publication date.


## How to run scripts?
1. Download the project repository to explore the code and notebook with documentations.
2. Install packages.
   ```bash
   pip install -r requirements.txt
   ```
4. Run the pipeline for the best model.
   ```bash
   streamlit run src/app.py
   ```
   ![](https://github.com/Engelbert107/Book-Recommendation-System/blob/main/images/streamlit_cmd.png)
5. Open the Streamlit app into your browser and type something.
   ![](https://github.com/Engelbert107/Book-Recommendation-System/blob/main/images/streamlit_browser.png)


## Access to the notebook through the following link:
- Access to the [notebook file](https://github.com/Engelbert107/Book-Recommendation-System/blob/main/notebook/BookRecommenderSystem.ipynb)
- Access to the [app file](https://github.com/Engelbert107/Book-Recommendation-System/blob/main/src/app.py)
- Access to different [images here](https://github.com/Engelbert107/Book-Recommendation-System/tree/main/images) 
