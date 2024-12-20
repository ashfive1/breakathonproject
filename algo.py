import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data from CSV files
file_path = 'D:/Ashwanth/college/Break-a-thon/first_repo/breakathonproject/TaskFlow_MockData.csv'
file_path_2 = 'D:/Ashwanth/college/Break-a-thon/first_repo/breakathonproject/TaskFlow_LabourWant.csv'

# Handle encoding explicitly
def safe_read_csv(file_path, encoding_options=('utf-8', 'latin1')):
    for encoding in encoding_options:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read the file {file_path} with provided encodings.")

data = safe_read_csv(file_path)  # Mock dataset for firms
data_want = safe_read_csv(file_path_2)

# Labour Want (Firms)
firms_data = {
    "Firm_ID": list(range(1, len(data_want) + 1)),  # Generating Firm IDs
    "Title": data_want["Job Title"].fillna(""),
    "Category_Name": data_want["Job Type"].fillna(""),
    "Experience": data_want["Experience level"].fillna("Not Specified"),
    "Location": data_want["Location"].fillna(""),
    "Description": data_want["Requirment of the company "].fillna(""),
    "Facilities": data_want["Facilities"].fillna(""),
}

# Labour Do (Freelancers)
num_records = len(data_want)
freelancers_data = {
    "Freelancer_ID": list(range(1, num_records + 1)),
    "Experience": ["3 years", "Internships", "6 years", "5 years"] * (num_records // 4 + 1),
    "Availability": ["Full-time", "Part-time", "Remote", "Contract"] * (num_records // 4 + 1),
    "Qualification": ["Bachelor's", "Diploma", "Master's", "PhD"] * (num_records // 4 + 1),
    "Domain": [
        "Data Analysis", "Graphic Design", "Machine Learning", "Web Development"
    ] * (num_records // 4 + 1),
}

# Truncate lists to match length of data_want
freelancers_data = {key: values[:num_records] for key, values in freelancers_data.items()}

firms_df = pd.DataFrame(firms_data)
freelancers_df = pd.DataFrame(freelancers_data)

# Matching Algorithm
def preprocess_and_match(firms_df, freelancers_df):
    # Replace NaN values with empty strings in both DataFrames
    firms_df = firms_df.fillna("")
    freelancers_df = freelancers_df.fillna("")

    # Combine relevant fields into single text for vectorization
    firms_df['Combined'] = (
        firms_df['Experience'] + " " +
        firms_df['Category_Name'] + " " +
        firms_df['Location'] + " " +
        firms_df['Description'] + " " +
        firms_df['Facilities']
    )

    freelancers_df['Combined'] = (
        freelancers_df['Experience'] + " " +
        freelancers_df['Availability'] + " " +
        freelancers_df['Qualification'] + " " +
        freelancers_df['Domain']
    )

    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer()
    firms_vectors = vectorizer.fit_transform(firms_df['Combined'])
    freelancers_vectors = vectorizer.transform(freelancers_df['Combined'])

    # Compute similarity scores
    similarity_matrix = cosine_similarity(firms_vectors, freelancers_vectors)

    # Generate matches
    matches = []
    for firm_idx, firm in firms_df.iterrows():
        best_match_idx = similarity_matrix[firm_idx].argmax()
        best_match_score = similarity_matrix[firm_idx].max()
        matches.append({
            "Firm_ID": firm['Firm_ID'],
            "Freelancer_ID": freelancers_df.iloc[best_match_idx]['Freelancer_ID'],
            "Score": best_match_score,
        })

    return pd.DataFrame(matches)

# Execute Matching
matches_df = preprocess_and_match(firms_df, freelancers_df)

# Save results to a CSV file
output_file = "D:/Ashwanth/college/break-a-thon/first_repo/breakathonproject/TaskFlow_Matches.csv"
matches_df.to_csv(output_file, index=False)

print("Matching completed. Results saved to:", output_file)
