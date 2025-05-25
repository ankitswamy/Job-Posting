# --------- Importing libraries ---------
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import os
import smtplib
from email.mime.text import MIMEText
import schedule
import time
# --------- User Preferences ---------
user_preferences = {
    "jerin.ds24@duk.ac.in": ["python", "machine learning", "data science"]
}

# --------- Scraping ---------
def scrape_karkidi_jobs(keyword="", pages=2):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for page in range(1, pages + 1):
        url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
        print(f"Scraping page: {page} | URL: {url}")
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        job_blocks = soup.find_all("div", class_="ads-details")

        for job in job_blocks:
            try:
                title = job.find("h4").get_text(strip=True)
                company = job.find("a", href=lambda x: x and "Employer-Profile" in x).get_text(strip=True)
                location = job.find("p").get_text(strip=True)
                experience = job.find("p", class_="emp-exp").get_text(strip=True)
                key_skills_tag = job.find("span", string="Key Skills")
                skills = key_skills_tag.find_next("p").get_text(strip=True) if key_skills_tag else ""
                summary_tag = job.find("span", string="Summary")
                summary = summary_tag.find_next("p").get_text(strip=True) if summary_tag else ""

                jobs_list.append({
                    "Title": title,
                    "Company": company,
                    "Location": location,
                    "Experience": experience,
                    "Summary": summary,
                    "Skills": skills
                })
            except Exception as e:
                print(f"Error parsing job block: {e}")
                continue
        time.sleep(1)  # polite scraping delay
    return pd.DataFrame(jobs_list)

# --------- Preprocessing ---------
def preprocess_skills(df):
    df = df.copy()
    df['Skills'] = df['Skills'].str.lower().str.replace(r'[^a-zA-Z0-9, ]', '', regex=True)
    df['Skills'] = df['Skills'].str.split(',').apply(lambda x: [skill.strip() for skill in x])
    df['skills_str'] = df['Skills'].apply(lambda x: ' '.join(x))
    return df

# --------- Clustering ---------
def cluster_jobs(df, n_clusters=5):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['skills_str'])
    model = AgglomerativeClustering(n_clusters=n_clusters)
    df['Cluster'] = model.fit_predict(X.toarray())
    return df, model, vectorizer

# --------- Save artifacts ---------
def save_all(df, model, vectorizer):
    os.makedirs('model', exist_ok=True)
    df.to_csv("clustered_jobs.csv", index=False)
    joblib.dump(model, "model/karkidi_model.pkl")
    joblib.dump(vectorizer, "model/karkidi_vectorizer.pkl")
    print("Saved model, vectorizer and clustered job data.")

# --------- Email Alerts ---------
def send_email(to_email, subject, body, from_email, password):
    msg = MIMEText(body, 'plain', 'utf-8')
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(from_email, password)
        server.send_message(msg)
    print(f"Email sent to {to_email}")

def alert_users(df_new_jobs, user_prefs, from_email, email_password):
    # Ensure skills_str is treated as a string (in case of NaNs or numbers)
    df_new_jobs['skills_str'] = df_new_jobs['skills_str'].astype(str)

    for user_email, keywords in user_prefs.items():
        matched_jobs = df_new_jobs[df_new_jobs['skills_str'].apply(
            lambda x: any(keyword.lower() in x.lower() for keyword in keywords)
        )]

        if not matched_jobs.empty:
            jobs_str = matched_jobs[['Title', 'Company', 'Location']].to_string(index=False)
            subject = "New Job Matches Your Preferences"
            body = f"Hello,\n\nNew jobs matching your preferences:\n\n{jobs_str}\n\nBest regards,\nJob Alert System"
            print(f"📧 Sending alert to {user_email}...")
            send_email(user_email, subject, body, from_email, email_password)
        else:
            print(f"❌ No matching jobs for {user_email}")

# --------- Main pipeline ---------
def main():
    # Use your own Gmail credentials here or load from environment variables
    from_email = "jerin.ds24@duk.ac.in"
    email_password = "**** **** **** ****"

    # Scrape jobs (empty keyword for all jobs)
    df_jobs = scrape_karkidi_jobs(keyword="", pages=3)

    # Preprocess and cluster
    df_jobs = preprocess_skills(df_jobs)
    df_jobs, model, vectorizer = cluster_jobs(df_jobs)

    # Save model and data
    save_all(df_jobs, model, vectorizer)

    # Alert users about matching jobs
    alert_users(df_jobs, user_preferences, from_email, email_password)

if __name__ == "__main__":
    main()

    # Schedule daily run at 8:00 AM
    schedule.every().day.at("08:00").do(main)

    print("Scheduler started. Waiting for the scheduled time...")

    while True:
        schedule.run_pending()
        time.sleep(30)