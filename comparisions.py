import json
from typing import List, Tuple
import string
import re

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lower(s))))

def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = set(pred_tokens) & set(truth_tokens)
    num_same = sum(min(pred_tokens.count(token), truth_tokens.count(token)) for token in common)

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)

def evaluate(predictions: List[str], references: List[str]) -> float:
    """Compute average F1 score across all QnA pairs."""
    assert len(predictions) == len(references), "Prediction and reference counts must match."
    scores = [f1_score(p, r) for p, r in zip(predictions, references)]
    return sum(scores) / len(scores)

# Example Usage:
if __name__ == "__main__":
    # Replace these with your own outputs and references
    with open('llama_raw_output.json', 'r') as f:
        data = json.load(f)
    ground_truths = [
        "University of the Witwatersrand.",
        "andrei@cs.unc.edu",
        "Student Photo Directory",
        "Administration",
        "The purpose is to serve as a collaborative co-working and computing hub, driving hands-on learning opportunities for CS majors.",
        "sosoji@cs.unc.edu",
        "To machines not granted access, the port apparently doesn\u2019t exist, so there is nothing there to attack.",
        "University of Maryland, College Park",
        "It allows them to learn about the recruiting process and build relationships with potential employers. This helps in creating connections that could potentially lead to future job opportunities. The informal",
        "2022",
        "Sitterson Hall.  Location: University of North Carolina at Chapel Hill.  (Room 121)  Hours: Monday - Friday, 8am",
        "The international classes offer courses that complement the major and broaden your global perspective.  This is beneficial for a computer science student as it provides additional skills and a broader",
        "044 Sitterson Hall",
        "Kevin Sun's title is Teaching Assistant Professor.",
        "1-1023 are \u201cwell-known\u201d ports.",
        "Assistant Professor",
        "Brain connectivity.  He is a researcher of brain connectivity in the Department of Biomedical Engineering at the University of Iowa.  Munsell's research focuses on",
        "40 undergraduate students.",
        "Quick Links.  It is located in the top right corner of the page.  It seems to be a section for university students with various links for academic resources",
        "to support the students.  The Employer Board Meetings will help guide and support our programs.  This will ultimately support the students.  This will make the programs",
        "andrew@cs.unc.edu or https://andrewkwong.org",
        "Try using a sniffer program to capture packets involved in establishing the connection.",
        "Up to one year.",
        "M.S. 1984",
        "https://help.cs.unc.edu/en/blog/services-we-provide",
        "The department faculty are members of the Association for Computing Machinery (ACM) and the Institute of Electrical and Electronics Engineers (IEEE). Alternatively, other possible answers could",
        "As of Fall 2021.  It has been implemented at that time.  (Context is not clear on whether it was implemented at the beginning or end",
        "Rafael Zaldivar",
        "The purpose of the IT Requests topic is to facilitate communication with the IT department.",
        "External Relations",
        "Yes, a software practicum course is available.",
        "None",
        "Networking events, community service, internships, career fairs, and job shadowing.  These opportunities allow students to engage with industry professionals and organizations.",
        "MIT Dalai Lama Center for Ethics and Transformative Values",
        "Kauffman Entrepreneurial Fellowship",
        "None",
        "None",
        "Director of Career Services.  She oversees all career initiatives, leads a team of Computer Science Career Assistants, and collaborates with faculty and employers to integrate career",
        "Cyber-Physical Systems, Formal Methods, Control Theory, Hybrid Systems, Autonomy, Embedded and Real-Time Systems, and Probabilistic Systems",
        "Concurrent and distributed computing and real-time systems.",
        "None",
        "3.3",
        "The department typically participates in the UNC Science Expo and tries to hold one open house per year that is open to the public.",
        "UNC-Chapel Hill.",
        "Users who read mail from Gmail can set a vacation message using the info at Google Automatic Reply Help.",
        "Research Administration for Scientists.  The course is designed for senior PhD students, post docs and junior faculty. It's intended for students in the physical, computational,",
        "At least 9 hours of courses (3 classes).",
        "J. Anderson",
        "His research interests include computer architecture and systems, brain-computer interfaces, quantum computing, brain-inspired AI hardware, cognitive modeling, formal control, energy and power efficiency, security, machine learning, datacenters and cloud, and compilers.",
        "Swarthmore College.",
    ]
    for temp in [0.0, 0.3, 0.7, 1.0, 1.3]:
        model_outputs = [item['answer'] for item in data if item['temp']==temp]
        avg_f1 = evaluate(model_outputs, ground_truths)
        print(f"Average F1 Score: {avg_f1:.4f} temp: {temp}")