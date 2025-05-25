# Alert System and Job Posting Classification 



This project scrapes job postings from [karkidi.com](https://www.karkidi.com), preprocesses and clusters jobs based on required skills, and sends email alerts to users when matching jobs are found. The system also includes an interactive Streamlit web app to perform scraping, clustering, and visualization.

---

## What I Did in This Project

1. **Web Scraping**  
   - Built a scraper using `requests` and `BeautifulSoup` to extract job postings, including title, company, location, experience, skills, and summary from multiple pages on karkidi.com.
   - Added polite scraping delays (`time.sleep`) to avoid overwhelming the site.

2. **Data Preprocessing**  
   - Cleaned and normalized the skills text by removing special characters and converting to lowercase.
   - Split skills into lists and then joined back into strings suitable for vectorization.

3. **Clustering Jobs**  
   - Used `TfidfVectorizer` to convert job skills into numerical features.
   - Applied `AgglomerativeClustering` to group similar jobs based on required skills.
   - Saved clustering model and vectorizer using `joblib`.

4. **Email Alerts**  
   - Created a system to match new jobs against user preferences (keywords).
   - Set up sending email alerts via Gmail SMTP for matching jobs.

5. **Streamlit Web App**  
   - Developed an interactive Streamlit app to:
     - Take user input for job keyword and number of pages.
     - Scrape and cluster jobs on demand.
     - Display clustered jobs in tables and allow easy browsing.
   - Enabled saving model and clustered data after each run.

6. **Version Control & Deployment**  
   - Managed the project using Git and GitHub.
   - Deployed the Streamlit app to Streamlit Community Cloud for easy access online.

---

## How to Run Locally

1. Clone the repo:

    ```bash
    git clone https://github.com/ankitswamy/Job-Posting
    cd Job-Posting-Classification
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

---

## Live App

Try the app live on Streamlit:  
[https://job-posting-fauppnrnccuqthzsmdpndw.streamlit.app/](https://job-posting-fauppnrnccuqthzsmdpndw.streamlit.app/)

---

## Project Structure

- `app.py` — Streamlit app script for job scraping and clustering  
- `job_posting_classification.py` — Core functions: scraping, preprocessing, clustering, email alerts  
- `model/` — Saved clustering model and vectorizer  
- `clustered_jobs.csv` — CSV file with clustered jobs  
- `requirements.txt` — Python dependencies  

---

## Contact

Jerin John Chacko  
Email: [ankit.ds24@duk.ac.in](mailto:ankit.ds24@duk.ac.in)  

---

*Thank you for checking out the project!*
