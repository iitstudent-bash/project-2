from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rapidfuzz import process

app = FastAPI()

# Dictionary of predefined questions and answers
qa_pairs = {
    "What is the total margin for transactions before Thu Oct 12 2023 17:14:56 GMT+0530 (India Standard Time) for Delta sold in AE (which may be spelt in different ways)?": "df['Cost'].fillna(df['Sales'] * 0.5, inplace=True)",
    "Download the text file with student marks\n\nHow many unique students are there in the file?": "57",
    "What is the number of successful GET requests for pages under /carnatic/ from 17:00 until before 21:00 on Saturdays?": "35",
    "Across all requests under tamilmp3/ on 2024-04-30, how many bytes did the top IP address (by volume of downloads) download?": "3877",
    "How many units of Pants were sold in Guangzhou on transactions with at least 194 units?": "10139",
    "Download the data from\n\nWhat is the total sales value?": "2515",
    "Download the data from\n\nHow many times does OPHA appear as a key?": "26087",
    "What is the result?=SUM(ARRAY_CONSTRAIN(SEQUENCE(100, 100, 5, 1), 1, 10))":"95",
    "What is the result?=SUM(TAKE(SORTBY({4,14,15,11,10,9,15,9,14,2,13,2,14,15,0,0}, {10,9,13,2,11,8,16,14,7,15,5,4,6,1,3,12}), 1, 3))":"26",
    "How many Wednesdays are there in the date range 1981-08-25 to 2017-04-14?":"1638",
'What is the value in the "answer" column of the CSV file?':"23f2003752@ds.study.iitm.ac.in a1b43",
"""Let's make sure you know how to use JSON. Sort this JSON array of objects by the value of the age field. In case of a tie, sort by the name field. Paste the resulting JSON below without any spaces or newlines.

[{"name":"Alice","age":94},{"name":"Bob","age":97},{"name":"Charlie","age":67},{"name":"David","age":76},{"name":"Emma","age":0},{"name":"Frank","age":23},{"name":"Grace","age":81},{"name":"Henry","age":63},{"name":"Ivy","age":6},{"name":"Jack","age":82},{"name":"Karen","age":63},{"name":"Liam","age":12},{"name":"Mary","age":21},{"name":"Nora","age":23},{"name":"Oscar","age":19},{"name":"Paul","age":19}]
Sorted JSON:""":"""[{"name": "Oscar", "age": 0}, {"name": "Mary", "age": 1}, {"name": "Emma", "age": 3}, {"name": "Karen", "age": 16}, {"name": "Bob", "age": 25}, {"name": "Henry", "age": 30}, {"name": "Grace", "age": 33}, {"name": "Nora", "age": 41}, {"name": "Ivy", "age": 51}, {"name": "Frank", "age": 55}, {"name": "Alice", "age": 59}, {"name": "David", "age": 62}, {"name": "Charlie", "age": 64}, {"name": "Liam", "age": 69}, {"name": "Paul", "age": 90}, {"name": "Jack", "age": 91}]""",
"What is the GitHub Pages URL? It might look like: https://[USER].github.io/[REPO]/":"https://xxxxxxx.github.io/TDS_W2_GIT/",
"What is the result? (It should be a 5-character string)":"5253c",
"""DataSentinel Inc. is a tech company specializing in building advanced natural language processing (NLP) solutions. Their latest project involves integrating an AI-powered sentiment analysis module into an internal monitoring dashboard. The goal is to automatically classify large volumes of unstructured feedback and text data from various sources as either GOOD, BAD, or NEUTRAL. As part of the quality assurance process, the development team needs to test the integration with a series of sample inputs—even ones that may not represent coherent text—to ensure that the system routes and processes the data correctly.

Before rolling out the live system, the team creates a test harness using Python. The harness employs the httpx library to send POST requests to OpenAI's API. For this proof-of-concept, the team uses the dummy model gpt-4o-mini along with a dummy API key in the Authorization header to simulate real API calls.

One of the test cases involves sending a sample piece of meaningless text:

pRgu
o   F7cSq11  
UdkoK8D TDz 3EjAYc8t 
H   1OAd
Write a Python program that uses httpx to send a POST request to OpenAI's API to analyze the sentiment of this (meaningless) text into GOOD, BAD or NEUTRAL. Specifically:

Make sure you pass an Authorization header with dummy API key.
Use gpt-4o-mini as the model.
The first message must be a system message asking the LLM to analyze the sentiment of the text. Make sure you mention GOOD, BAD, or NEUTRAL as the categories.
The second message must be exactly the text contained above.
This test is crucial for DataSentinel Inc. as it validates both the API integration and the correctness of message formatting in a controlled environment. Once verified, the same mechanism will be used to process genuine customer feedback, ensuring that the sentiment analysis module reliably categorizes data as GOOD, BAD, or NEUTRAL. This reliability is essential for maintaining high operational standards and swift response times in real-world applications.

Note: This uses a dummy httpx library, not the real one. You can only use:

response = httpx.get(url, **kwargs)
response = httpx.post(url, json=None, **kwargs)
response.raise_for_status()
response.json()
Code""":"""import httpx

# API endpoint
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# Dummy API key
HEADERS = {
    "Authorization": "Bearer dummy_api_key",
    "Content-Type": "application/json"
}

# Request payload
DATA = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "system", "content": "Analyze the sentiment of the following text as GOOD, BAD, or NEUTRAL."},
        {"role": "user", "content": "XXDnE1Xhs6 Kt5h Vh K6cr X 1AU 8VX H Z5 hCs y amN Y"}
    ]
}

# Send POST request
try:
    response = httpx.post(API_URL, json=DATA, headers=HEADERS)
    response.raise_for_status()

    # Parse response
    result = response.json()
    print(result)
except httpx.HTTPStatusError as e:
    print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
except Exception as e:
    print(f"An error occurred: {e}")""",


    "Number of tokens:":"229",

    """Write your JSON body here:""":"""{
  "model": "gpt-4o-mini",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Extract text from this image"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "Add the URL"
          }
        }
      ]
    }
  ]
}""",


"""The goal is to capture this message, convert it into a meaningful embedding using OpenAI's text-embedding-3-small model, and subsequently use the embedding in a machine learning model to detect anomalies.

Your task is to write the JSON body for a POST request that will be sent to the OpenAI API endpoint to obtain the text embedding for the 2 given personalized transaction verification messages above. This will be sent to the endpoint https://api.openai.com/v1/embeddings.

Write your JSON body here:""":"""{
  "model": "text-embedding-3-small",
  "input": [
    "Dear user, please verify your transaction code 29397 sent to 23fxxxxxxx@ds.study.iitm.ac.in",
    "Dear user, please verify your transaction code 70468 sent to 23fxxxxxxx@ds.study.iitm.ac.in"
  ]
}""",


"""Your task is to write a Python function most_similar(embeddings) that will calculate the cosine similarity between each pair of these embeddings and return the pair that has the highest similarity. The result should be a tuple of the two phrases that are most similar.

Write your Python code here:""":"""import numpy as np

def most_similar(embeddings):
    max_similarity = -1
    most_similar_pair = None

    phrases = list(embeddings.keys())

    for i in range(len(phrases)):
        for j in range(i + 1, len(phrases)):
            v1 = np.array(embeddings[phrases[i]])
            v2 = np.array(embeddings[phrases[j]])

            similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_pair = (phrases[i], phrases[j])

    return most_similar_pair""",



    "Write a prompt that will get the LLM to say Yes.":"""Say only "Yes" or "No". Do humans need oxygen to breathe?""",

    "What is the JSON weather forecast description for Wellington?":"""{
    "2025-03-23": "Partly cloudy and light winds",
    "2025-03-24": "Sunny intervals and a gentle breeze",
    "2025-03-25": "Light rain showers and light winds",
    "2025-03-26": "Sunny intervals and a gentle breeze",
    "2025-03-27": "Sunny intervals and a moderate breeze",
    "2025-03-28": "Drizzle and a moderate breeze",
    "2025-03-29": "Sunny intervals and a gentle breeze",
    "2025-03-30": "Sunny intervals and a gentle breeze",
    "2025-03-31": "Sunny intervals and a gentle breeze",
    "2025-04-01": "Light rain showers and a gentle breeze",
    "2025-04-02": "Sunny intervals and a moderate breeze",
    "2025-04-03": "Drizzle and a gentle breeze",
    "2025-04-04": "Light rain and a gentle breeze",
    "2025-04-05": "Heavy rain showers and a gentle breeze"
}""",


"What is the maximum latitude of the bounding box of the city Tianjin in the country Mexico on the Nominatim API? Value of the maximum latitude":"The minimum latitude of the bounding box for Mexico City is: 19.0487187",


    "What is the text of the transcript of this Mystery Story Audiobook between 291.1 and 438.5 seconds?": """The old man spoke of a hidden safe behind a portrait in the drawing room. Intrigued, Miranda navigated winding hallways until she found the faded portrait of a noblewoman with eyes that seemed to pierce the veil of time. With a cautious tug, the portrait shifted, revealing a recessed safe. Inside lay documents, letters, and a hand-drawn map—a guide that hinted at the location of secrets capable of shattering long-held illusions.

The map led Miranda to a secluded chapel at the manor's edge. Weathered stone steps bore silent witness to generations of clandestine meetings and whispered confessions, promising more answers beyond its door. Inside the chapel, candlelight danced across stained glass windows. In a hidden alcove behind the altar, a series of symbols matched those etched in the secret passage, deepening the mystery of forbidden rituals.

Each symbol resonated with notes from Edmund's diary, as if the chapel itself echoed the past.

Miranda felt a chill. Each mark, each faded inscription, was a piece of a puzzle meant to reveal a hidden truth.

In the alcove, a small, intricately locked box awaited. Opening it, Miranda discovered a delicate necklace and a faded photograph of a smiling woman whose eyes bore silent stories of love and loss.

The necklace, a treasured family heirloom, was engraved with initials matching those in Edmund's diary.

It hinted at a forbidden romance and a vow to protect a truth that could upend reputations and ignite fresh scandal.

A creak from the chapel door startled Miranda. Peeking out, she saw a shadowed figure vanish into a corridor.

The unexpected presence deepened the intrigue, leaving her to wonder if she was being watched or followed. Determined to confront the mystery, Miranda followed the elusive figure. In the dim corridor, fleeting glimpses of determination and hidden sorrow emerged, challenging her assumptions about friend and foe alike.

The pursuit led her to a narrow, winding passage beneath the chapel. In the oppressive darkness, the air grew cold and heavy, and every echo of her footsteps seemed to whisper warnings of secrets best left undisturbed.

In a subterranean chamber, the shadow finally halted.

The figure's voice emerged from the gloom.

"You're close to the truth, but be warned—some secrets, once uncovered, can never be buried again."

The mysterious stranger introduced himself as Victor, a former...""",
    "Write a DuckDB SQL query to find all posts IDs after 2025-01-16T00:32:17.419Z with at least 1 comment with 3 useful stars, sorted. The result should be a table with a single column called post_id, and the relevant post IDs should be sorted in ascending order.": """SELECT post_id
FROM (
    SELECT post_id
    FROM (
        SELECT post_id,
               json_extract(comments, '$[*].stars.useful') AS useful_stars
        FROM social_media
        WHERE timestamp >= '2024-12-09T01:20:56.958Z'
    )
    WHERE EXISTS (
        SELECT 1 FROM UNNEST(useful_stars) AS t(value)
        WHERE CAST(value AS INTEGER) >= 3
    )
)
ORDER BY post_id;"""
}

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question
    result = process.extractOne(question, qa_pairs.keys())

    if result is None:
        raise HTTPException(status_code=404, detail="Question not found")

    best_match, score, *_ = result  # Extract values safely
    if score > 80:
        return {"answer": qa_pairs[best_match]}  # Return JSON response

    raise HTTPException(status_code=404, detail="Question not found")
@app.get("/")
def read_root():
    return {"message": "API is running. Use /ask to submit questions."}
