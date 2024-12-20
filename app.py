from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "supersecretkey"  # For flashing messages

# File paths for the CSV files
TASKS_FILE = "TaskFlow_MockData.csv"
PROFILES_FILE = "TaskFlow_LabourWant.csv"
MATCHES_FILE = "TaskFlow_Matches.csv"

# Ensure CSV files exist
if not os.path.exists(TASKS_FILE):
    pd.DataFrame(columns=["Firm_ID", "Job_Title", "Job_Type", "Experience_Level", "Location", "Requirement", "Facilities"]).to_csv(TASKS_FILE, index=False)

if not os.path.exists(PROFILES_FILE):
    pd.DataFrame(columns=["Freelancer_ID", "Experience", "Availability", "Qualification", "Domain"]).to_csv(PROFILES_FILE, index=False)

if not os.path.exists(MATCHES_FILE):
    pd.DataFrame(columns=["Task", "Freelancer", "Score"]).to_csv(MATCHES_FILE, index=False)


def read_csv_safe(file_path):
    try:
        return pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="latin1")  # Fallback encoding

# Home Page
@app.route("/")
def index():
    return render_template("index.html")

# Post Task Page
@app.route("/post_task", methods=["GET", "POST"])
def post_task():
    if request.method == "POST":
        # Collect form data
        firm_id = len(read_csv_safe(TASKS_FILE)) + 1
        job_title = request.form.get("job_title")
        job_type = request.form.get("job_type")
        experience_level = request.form.get("experience_level")
        location = request.form.get("location")
        requirement = request.form.get("requirement")
        facilities = request.form.get("facilities")

        # Save to CSV
        new_task = pd.DataFrame([{
            "Firm_ID": firm_id,
            "Job_Title": job_title,
            "Job_Type": job_type,
            "Experience_Level": experience_level,
            "Location": location,
            "Requirement": requirement,
            "Facilities": facilities,
        }])
        tasks_df = read_csv_safe(TASKS_FILE)
        tasks_df = pd.concat([tasks_df, new_task], ignore_index=True)
        tasks_df.to_csv(TASKS_FILE, index=False)

        flash("Task successfully uploaded!")
        return redirect(url_for("post_task"))

    return render_template("post_task.html")

# View Tasks Page
@app.route("/view_tasks")
def view_tasks():
    matches_df = read_csv_safe(MATCHES_FILE)
    tasks_df = read_csv_safe(TASKS_FILE)

    if matches_df.empty or tasks_df.empty:
        flash("No tasks available.")
        return render_template("view_tasks.html", tasks=[])

    # Normalize column names
    matches_df.columns = matches_df.columns.str.strip().str.lower()
    tasks_df.columns = tasks_df.columns.str.strip().str.lower()

    # Sort matches by score in descending order
    matches_df = matches_df.sort_values(by="score", ascending=False)

    # Map task details to matches
    if "task" in matches_df.columns and "job_title" in tasks_df.columns:
        matches_df = matches_df.merge(tasks_df, left_on="task", right_on="job_title", how="left")
    else:
        flash("Column names do not match for merging.")
        return render_template("view_tasks.html", tasks=[])

    # Prepare data for rendering
    tasks = matches_df.to_dict(orient="records")
    return render_template("view_tasks.html", tasks=tasks)

# Profile Page
@app.route("/profile", methods=["GET", "POST"])
def profile():
    if request.method == "POST":
        # Collect form data
        freelancer_id = len(read_csv_safe(PROFILES_FILE)) + 1
        experience = request.form.get("experience")
        availability = request.form.get("availability")
        qualification = request.form.get("qualification")
        domain = request.form.get("domain")

        # Save to CSV
        new_profile = pd.DataFrame([{
            "Freelancer_ID": freelancer_id,
            "Experience": experience,
            "Availability": availability,
            "Qualification": qualification,
            "Domain": domain,
        }])
        profiles_df = read_csv_safe(PROFILES_FILE)
        profiles_df = pd.concat([profiles_df, new_profile], ignore_index=True)
        profiles_df.to_csv(PROFILES_FILE, index=False)

        flash("Profile successfully created!")
        return redirect(url_for("profile"))

    return render_template("profile.html")

# Matchmaking Algorithm
@app.route("/matchmaking")
def matchmaking():
    tasks_df = read_csv_safe(TASKS_FILE)
    profiles_df = read_csv_safe(PROFILES_FILE)

    if tasks_df.empty or profiles_df.empty:
        return jsonify({"error": "No data available for matchmaking"}), 400

    # Matchmaking logic
    tasks_df["Combined"] = (
        tasks_df["Experience_Level"].fillna("") + " " +
        tasks_df["Job_Type"].fillna("") + " " +
        tasks_df["Location"].fillna("") + " " +
        tasks_df["Requirement"].fillna("") + " " +
        tasks_df["Facilities"].fillna("")
    )
    profiles_df["Combined"] = (
        profiles_df["Experience"].fillna("") + " " +
        profiles_df["Availability"].fillna("") + " " +
        profiles_df["Qualification"].fillna("") + " " +
        profiles_df["Domain"].fillna("")
    )

    vectorizer = TfidfVectorizer()
    tasks_vectors = vectorizer.fit_transform(tasks_df["Combined"])
    profiles_vectors = vectorizer.transform(profiles_df["Combined"])

    similarity_matrix = cosine_similarity(tasks_vectors, profiles_vectors)

    matches = []
    for task_idx, task in tasks_df.iterrows():
        best_match_idx = similarity_matrix[task_idx].argmax()
        best_match_score = similarity_matrix[task_idx].max()
        matches.append({
            "Task": task["Job_Title"],
            "Freelancer": profiles_df.iloc[best_match_idx]["Freelancer_ID"],
            "Score": best_match_score
        })

    # Save matches to CSV
    matches_df = pd.DataFrame(matches)
    matches_df.to_csv(MATCHES_FILE, index=False)

    return jsonify(matches)

if __name__ == "__main__":
    app.run(debug=True)
