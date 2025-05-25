import streamlit as st
import pandas as pd
import sys
import os
from job_posting import scrape_karkidi_jobs, preprocess_skills, cluster_jobs
import joblib
import os

st.title("🧠 Job Alert System")

keyword = st.text_input("Enter job keyword", "data science")
pages = st.slider("Number of pages to scrape", 1, 5, 2)

if st.button("Scrape Jobs and Cluster"):
    with st.spinner("Scraping and clustering jobs..."):
        df = scrape_karkidi_jobs(keyword, pages)
        df = preprocess_skills(df)
        df, model, vectorizer = cluster_jobs(df, n_clusters=5)

        os.makedirs("model", exist_ok=True)
        df.to_csv("clustered_jobs.csv", index=False)
        joblib.dump(model, "model/karkidi_model.pkl")
        joblib.dump(vectorizer, "model/karkidi_vectorizer.pkl")

    st.success("✅ Scraping and clustering complete!")
    st.dataframe(df[['Title', 'Company', 'Location', 'Cluster']])

    for clust in sorted(df['Cluster'].unique()):
        st.subheader(f"Cluster {clust}")
        st.table(df[df['Cluster'] == clust][['Title', 'Company', 'Location']].head(5))
