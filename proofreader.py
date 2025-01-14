from fastapi import HTTPException
import os
import json
import logging
from openai import OpenAI
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import time
import sys 
import urllib.parse 
import hashlib
import re

api_key = os.getenv("OPENAI_API_KEY")
if (api_key is None):
    raise ValueError("OpenAI API key not found. Please ensure it's set in your environment variables.")

client = OpenAI(api_key=api_key)

app = Flask(__name__)
socketio = SocketIO(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add StreamHandler to ensure error messages are shown in the terminal
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.ERROR)
logger.addHandler(stream_handler)

# API endpoint for the localragAPI
LOCALRAG_API_URL = "http://127.0.0.1:8000"

# define abbreviations for sentence splitting
abbreviations = [
    'Abs.', 'etc.', 'Dr.', 'Prof.', 'ff.', 'ua.', 'usw.', 'zB.', 'gem.', 
    'Nr.', 'Art.', 'vgl.', 'II.', 'III.', 'IV.', 'V.', 'VI.', 'VII.', 'VIII.', 
    'IX.', 'X.', 'Abschn.', 'a.A.', 'a.F.', 'a.M.','bzw.', 'lit.', 'Rn.'
]

# load legal texts that are used for detection
with open('legal_texts.json', 'r') as f:
    legal_texts = json.load(f)
legal_texts_abbreviations = [text['abbreviation'] for text in legal_texts]

# Define a global variable for the cache
cache = {}
#solution = None
#description = None

# Load cache from json file
def load_cache():
    global cache
    try:
        with open('proofreader_cache.json', 'r') as file:
            cache = json.load(file)
            logger.info("Loaded cache from proofreader_cache.json")
    except FileNotFoundError:
        cache = {}

# Save cache to json file
def save_cache():
    global cache
    with open('proofreader_cache.json', 'w') as file:
        json.dump(cache, file, indent=4, sort_keys=True)
        logger.info("Saved cache to proofreader_cache.json")

# Function to query Assistants API
def query_assistants_api(query, assistant_id, thread_id=None):
    try:
        # Create or continue a thread
        if thread_id is None:
            thread = client.beta.threads.create()
            thread_id = thread.id
            logger.info(f"Querying Assistants API. Created new thread with ID: {thread_id}")
        else:
            thread = client.beta.threads.retrieve(thread_id=thread_id)
            logger.info(f"Querying Assistants API. Retrieved existing thread with ID: {thread_id}")

        #logger.info(f"Using assistant ID: {assistant_id}")

        # Add the query to the thread
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=query
        )
        #logger.info(f"Added query to thread {thread_id}: {query}")

        # Run the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )
        #logger.info(f"Started assistant run with ID: {run.id}")

        # Polling mechanism to check the run's status
        while run.status in ["queued", "in_progress"]:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            logger.info(f"Run status: {run_status.status}")
            if run_status.status == "completed":
                break
            elif run_status.status == "failed":
                # Log detailed error information
                error_code = run_status.last_error.code if run_status.last_error else "Unknown"
                error_message = run_status.last_error.message if run_status.last_error else "No error message provided."
                logger.error(f"Run failed with error code: {error_code}, message: {error_message}")
                logger.error(f"The query that caused the error: {query}")
                raise Exception(f"Assistant run failed with error code: {error_code}, message: {error_message}")
            time.sleep(2)  # Wait for 2 seconds before checking again

        # Retrieve the assistant's response
        messages_response = client.beta.threads.messages.list(thread_id=thread_id)
        assistant_message = next(
            (msg for msg in messages_response.data if msg.role == "assistant"),
            None
        )
        if assistant_message is None or not assistant_message.content:
            logger.error("No assistant response found or response content is empty.")
            raise HTTPException(status_code=500, detail="No assistant response found or response content is empty.")
        
        # Extract and return the text content
        #logger.info(f"Assistant response: {assistant_message.content[0].text.value}")
        logger.info(f"Assistant responded successfully.")
        return assistant_message.content[0].text.value, thread_id

    except Exception as e:
        logger.error(f"Assistants API error: {e}")
        raise HTTPException(status_code=500, detail="Failed to query Assistants API.")

# function to convert markdown styles to html styles
def apply_html_styles(text):
    """Apply HTML styles to the given text."""
    logger.info("Applying HTML styles to the text.")

    # find all mentioned paragraphs and add a link to them
    links = find_links(extract_legal_texts(text, legal_texts_abbreviations))

    bold_open = True
    while "**" in text:
        if bold_open:
            text = text.replace("**", "<b>", 1)
        else:
            text = text.replace("**", "</b>", 1)
        bold_open = not bold_open

    """
    # increase text size for first order headlines, marked by # in the beginning and a line break at the end
    text = re.sub(r'^#(.*?)\n', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    # increase text size for second order headlines, marked by ## in the beginning and a line break at the end
    text = re.sub(r'^##(.*?)\n', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    # increase text size for third order headlines, marked by ### in the beginning and a line break at the end
    text = re.sub(r'^###(.*?)\n', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    # remove #
    text = text.replace("#", "")
    # remove line breaks after headlines
    text = text.replace("</h1>\n", "</h1>")
    text = text.replace("</h2>\n", "</h2>")
    text = text.replace("</h3>\n", "</h3>")
    """
    # implementation above did not work. make all kinds of headlines bold and add a line break after
    #text = re.sub(r'^(#{1,3})(.*?)$', r'<b>\2</b><br>', text, flags=re.MULTILINE)
    text = re.sub(r'^(#{1,4})(.*?)$', r'<b>\2</b>', text, flags=re.MULTILINE)

    # Use a regular expression to replace newlines followed by a number and a period
    #text = re.sub(r'\n(\d+)\.', r'<br>\1.', text)

    # remove any text between 【 and 】and make sure there are no double spaces left
    text = re.sub(r'【.*?】', '', text)
    text = text.replace("  ", " ")

    # add line breaks in front of bullet points ' - '
    #text = text.replace("- ", "<br>- ")

    # add line breakts after :
    #text = text.replace(":", ":<br>")

    # add links to all mentioned paragraphs
    if links:
        for paragraph, link in links.items():
            text = text.replace(paragraph, f'<a href="{link}" target="_blank">{paragraph}</a>')

    # Replace newlines with <br> tags
    text = text.replace("\n", "<br>")

    return text

def find_links(paragraphs):
    """Use the provided link structure to find links to mentioned Paragraphs.
    Link example: https://www.gesetze-im-internet.de/bgb/__433.html
    (BGB is the law, 433 is the paragraph)
    Create a dictionary with the paragraph number as key and the link as value.
    Input format: ['§ 433 Abs. 2 BGB', '§ 433 BGB']
    Output format: {'433': 'https://www.gesetze-im-internet.de/bgb/__433.html'}
    """
    logger.info("Finding links to mentioned paragraphs.")
    links = {}
    if not paragraphs:
        return None
    for paragraph in paragraphs:
        # Find the paragraph number in the text. It is the first number in the paragraph.
        paragraph_number = next((word for word in paragraph.split() if word.isdigit()), None)
        # Find the law abbreviation in the text by using the listlegal_texts_abbreviations
        law_abbreviation = next((word.rstrip(':').rstrip('.').rstrip(',').rstrip(')') for word in paragraph.split() if word.rstrip(':').rstrip('.').rstrip(',').rstrip(')') in legal_texts_abbreviations), None).lower()
        
        if law_abbreviation is not None:
            # build the link using the paragraph number and the law abbreviation
            link = f"https://www.gesetze-im-internet.de/{law_abbreviation}/__{paragraph_number}.html"
            links[paragraph] = link
    # log paragraphs along with their links
    for paragraph, link in links.items():
        logger.info(f"{paragraph} -> {link}")

    return links

# route for the main page
@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")

# replace text placeholders like {sentence} or {text}
def replace_text_placeholder(prompt, sentence=None, text=None, citations=None, missing_citations=None, solution=None, description=None, wrong_citations=None, correct_citations=None):
    """Replace {text}, {sentence}, and {citations} placeholders in the prompt with the provided text, sentence, and citations."""
    logger.info("Replacing text placeholders in the prompt.")
    if text is not None:
        prompt = prompt.replace("{text}", text)
    if sentence is not None:
        prompt = prompt.replace("{sentence}", sentence)
    if citations is not None:
        prompt = prompt.replace("{citations}", ", ".join(citations))
    if missing_citations is not None:
        prompt = prompt.replace("{missing_citations}", ", ".join(missing_citations))
    if solution is not None:
        prompt = prompt.replace("{case_solution}", solution)
    if description is not None:
        prompt = prompt.replace("{case_description}", description)
    if wrong_citations is not None:
        prompt = prompt.replace("{wrong_citations}", ", ".join(wrong_citations))
    if correct_citations is not None:
        prompt = prompt.replace("{correct_citations}", ", ".join(correct_citations))
    return prompt

# generate cache key
def save_to_cache(promptgroup_prompts_combined, assistant_id, responses, thread_id):
    """Generate a cache key for quick searching, but also save the promptgroup_prompts_combined, assistant_id, and thread_id in the cache."""
    logger.info("Saving promptgroup to cache.")
    global cache
    # Generate a unique key based on the promptgroup_prompts_combined AND assistant_id
    key = hashlib.md5(json.dumps(promptgroup_prompts_combined).encode('utf-8') + assistant_id.encode('utf-8')).hexdigest()
    # Save the key, promptgroup_prompts_combined, assistant_id, responses, and thread_id in the cache
    cache[key] = {
        "promptgroup_prompts_combined": promptgroup_prompts_combined,
        "assistant_id": assistant_id,
        "responses": responses,
        "thread_id": thread_id
    }

def search_cache(promptgroup_prompts_combined, assistant_id):
    """Search the cache for a matching promptgroup_prompts_combined and assistant_id. If nothing is found, return None."""
    logger.info("Searching cache for promptgroup.")
    global cache
    # Generate a unique key based on the promptgroup_prompts_combined AND assistant_id
    key = hashlib.md5(json.dumps(promptgroup_prompts_combined).encode('utf-8') + assistant_id.encode('utf-8')).hexdigest()
    # Check if the key exists in the cache
    if key in cache:
        return cache[key]
    else:
        return None

# generate feedback function (when button is pressed)
@app.route("/generate-feedback", methods=["POST"])
def generate_feedback():
    logger.info("Generating feedback.")
    #global solution, description

    try:
        # Reload cache
        load_cache()
        
        # load proofreader_prompts.json
        with open('proofreader_prompts.json', 'r') as file:
            promptgroups = json.load(file)
        
        text = request.form.get("student_input")
        selected_case = request.form.get("selected_case")
        logger.info(f"Received student input: {text}")
        logger.info(f"Selected case: {selected_case}")

        # load the solution text for the selected case from cases.json
        with open('cases.json', 'r') as file:
            cases = json.load(file)

        # extract legal texts and their paragraphs mentioned in the input text
        citations = extract_legal_texts(text, legal_texts_abbreviations)
        logger.info(f"Detected citations: {citations}")

        # the json is a list of dicts, the keys are "case_id", "title", "description", "solution"
        solution = next((case["solution"] for case in cases if case["title"] == selected_case), None)
        description = next((case["description"] for case in cases if case["title"] == selected_case), None)

        if solution is None:
            missing_citations = None
        else:
            correct_citations = extract_legal_texts(solution, legal_texts_abbreviations)
            logger.info(f"Correct citations: {correct_citations}")
            # calculate the missing citations
            missing_citations = [citation for citation in correct_citations if citation not in citations]

            wrong_citations = [citation for citation in citations if citation not in correct_citations]

            # if list is empty, set it to None
            if not missing_citations:
                missing_citations = None
            if not wrong_citations:
                wrong_citations = None

            logger.info(f"Missing citations: {missing_citations}")

        # Check the size of the student input
        MAX_INPUT_SIZE = 10000  # Set a threshold for the maximum input size (in characters)
        if len(text) > MAX_INPUT_SIZE:
            logger.error(f"Input is too large. Maximum allowed size is {MAX_INPUT_SIZE} characters.")
            return jsonify({"error": f"Input is too large. Maximum allowed size is {MAX_INPUT_SIZE} characters."})

        # call sentence splitting function
        sentences = split_sentences(text)
        logger.info(f"Number of sentences: {len(sentences)}")
        logger.info(f"Sentences: {sentences}")



        # create promptgroups_sentences and promptgroups_text
        promptgroups_sentences = {}
        promptgroups_text = {}
        for promptgroup_name, promptgroup in promptgroups.items():
            if "{sentence}" in promptgroup["prompts"][0]["prompt"]:
                promptgroups_sentences[promptgroup_name] = promptgroup
            else:
                promptgroups_text[promptgroup_name] = promptgroup

        # generate feedback for the whole text if dict is not empty
        text_feedback = []
        if promptgroups_text:
            text_feedback = generate_text_feedback(text=text, promptgroups=promptgroups_text, citations=citations, missing_citations=missing_citations, wrong_citations=wrong_citations, solution=solution, description=description, correct_citations=correct_citations)


        # generate feedback for each sentence if dict is not empty
        sentence_feedback = []
        if promptgroups_sentences:
            sentence_feedback = generate_sentence_feedback(sentences, promptgroups_sentences)

        if not sentence_feedback and not text_feedback:
            logger.error("No feedback generated.")
            return jsonify({"error": "No feedback generated. Please check your input or feedback data."})

        # save cache to json file
        save_cache()

        return jsonify({
            "success": True,
            "message": "Feedback generated successfully."
        })


    except Exception as e:
        logger.error(f"Error generating feedback: {e}", exc_info=True)
        return jsonify({"error": "Unable to generate feedback."})

# generate per sentence feedback for promptgroups that have the sentence placeholder
def generate_sentence_feedback(sentences, promptgroups):
    """Generate feedback for each sentence based on the promptgroups that have the sentence placeholder."""
    logger.info("Generating feedback for each sentence.")
    feedback = []
    try:
        # loop though all sentences
        for sentence in sentences:
            # loop thourgh all promptgroups
            for promptgroup_name, promptgroup in promptgroups.items():
                if not (promptgroup["useprompt"]):
                    continue
                logger.info(f"processing prompt: {promptgroup_name}")
                # loop though all prompts in the promptgroup
                promptgroup_prompts_combined = list()
                for prompt in promptgroup["prompts"]:
                    # replace placeholders in prompt
                    prompt_text = replace_text_placeholder(prompt["prompt"], sentence=sentence)
                    # attach prompt_text to promptgroup_prompts_combined along with promptgroup["assistant_id"]
                    promptgroup_prompts_combined.append({"prompt": prompt_text, "display_answer": prompt["display_answer"]})
                # check if promptgroup_prompts combined exists in cache
                cache_entry = search_cache(promptgroup_prompts_combined, promptgroup["assistant_id"])
                # if cahe entry is None, query the API
                if cache_entry is None:
                    # query the API for each promptgroup_prompts_combined and add the responses to a list
                    responses = []
                    thread_id = None
                    for prompt in promptgroup_prompts_combined:
                        response, thread_id = query_assistants_api(prompt["prompt"], promptgroup["assistant_id"], thread_id=thread_id)
                        if prompt["display_answer"] and "Note: 1" not in response:
                            response = apply_html_styles(response)
                            responses.append(response)
                            # stream responses to the client
                            socketio.emit("feedback_update", {"sentence": sentence, "response": response, "prompt": promptgroup_name, "thread_id": thread_id, "assistant_id": promptgroup["assistant_id"]})
                        else:
                            logger.info(f"Received respone but not displaying it: {response}")
                    # save the responses in the cache
                    save_to_cache(promptgroup_prompts_combined, promptgroup["assistant_id"], responses, thread_id)
                # if cache entry is not None, use the response from cache
                else:
                    responses = cache_entry["responses"]
                    thread_id = cache_entry["thread_id"]
                    # stream cached responses to the client
                    for response in responses:
                        socketio.emit("feedback_update", {"sentence": sentence, "response": response, "prompt": promptgroup_name, "thread_id": thread_id, "assistant_id": promptgroup["assistant_id"]})
                feedback.append({"sentence": sentence, "responses": responses, "thread_id": thread_id})
    except Exception as e:
        logger.error(f"Error in generate_sentence_feedback: {e}", exc_info=True)
    return feedback

# generate feedback for the whole text
def generate_text_feedback(text, promptgroups, citations, missing_citations, wrong_citations, solution, description, correct_citations):
    """Generate feedback for the whole text based on the promptgroups that do not have the sentence placeholder."""
    logger.info("Generating feedback for the whole text.")
    feedback = []
    #global solution, description
    try:
        # loop though all promptgroups
        for promptgroup_name, promptgroup in promptgroups.items():

            if not (promptgroup["useprompt"]):
                continue

            # if promptgroup_name is "Paragraphen", only proceed if missing_citations is not empty or None
            if promptgroup_name == "Fehlende Paragraphen" and missing_citations is None:
                continue
            if promptgroup_name == "Paragraphen" and citations is None:
                continue
            #if promptgroup_name == "Beurteilung der Loesung":

            logger.info(f"processing prompt: {promptgroup_name}")
            # loop though all prompts in the promptgroup
            promptgroup_prompts_combined = list()
            for prompt in promptgroup["prompts"]:
                # replace placeholders in prompt
                prompt_text = replace_text_placeholder(prompt["prompt"], text=text, missing_citations=missing_citations, citations=citations, solution=solution, description=description, wrong_citations=wrong_citations, correct_citations=correct_citations)
                # attach prompt_text to promptgroup_prompts_combined along with promptgroup["assistant_id"]
                promptgroup_prompts_combined.append({"prompt": prompt_text, "display_answer": prompt["display_answer"]})
            # check if promptgroup_prompts combined exists in cache
            logger.info(f"promptgroup_prompts_combined: {promptgroup_prompts_combined}")
            cache_entry = search_cache(promptgroup_prompts_combined, promptgroup["assistant_id"])
            # if cahe entry is None, query the API
            if cache_entry is None:
                # query the API for each promptgroup_prompts_combined and add the responses to a list
                responses = []
                thread_id = None
                for prompt in promptgroup_prompts_combined:
                    response, thread_id = query_assistants_api(prompt["prompt"], promptgroup["assistant_id"], thread_id=thread_id)
                    if prompt["display_answer"]:
                        response = apply_html_styles(response)
                        responses.append(response)
                        socketio.emit("overall_feedback_update", {"response": response, "prompt": promptgroup_name, "thread_id": thread_id, "assistant_id": promptgroup["assistant_id"]})
                    else:
                        logger.info(f"Received respone but not displaying it: {response}")
                # save the responses in the cache
                save_to_cache(promptgroup_prompts_combined, promptgroup["assistant_id"], responses, thread_id)
            # if cache entry is not None, use the response from cache
            else:
                responses = cache_entry["responses"]
                thread_id = cache_entry["thread_id"]
                # stream cached responses to the client
                for response in responses:
                    socketio.emit("overall_feedback_update", {"response": response, "prompt": promptgroup_name, "thread_id": thread_id, "assistant_id": promptgroup["assistant_id"]})
            feedback.append({"responses": responses, "thread_id": thread_id})
    except Exception as e:
        logger.error(f"Error in generate_text_feedback: {e}", exc_info=True)
    return feedback

# handle follow-up questions
@socketio.on('follow_up_question')
def handle_follow_up_question(data):
    try:
        question = data['question']
        thread_id = data['thread_id']
        assistant_id = data['assistant_id']
        logger.info(f"Received follow-up question: {question}, thread_id: {thread_id}, assistant_id: {assistant_id}")

        # Query the Assistants API with the follow-up question
        response, _ = query_assistants_api(question, assistant_id=assistant_id, thread_id=thread_id)
        response = apply_html_styles(response)

        # Send the response back to the frontend
        logger.info(f"Sending follow-up response: {response}")
        emit('follow_up_response', {"response": response, "thread_id": thread_id})
    except Exception as e:
        logger.error(f"Error handling follow-up question: {e}", exc_info=True)
        emit('follow_up_response', {"error": "Unable to process follow-up question."})

# get case titles function
@app.route("/get-case-titles", methods=["GET"])
def get_case_titles():
    """Fetch case titles from cases.json."""
    logger.info("Fetching case titles.")
    with open('cases.json', 'r') as file:
        cases = json.load(file)
    case_titles = [case["title"] for case in cases]
    return jsonify({"case_titles": case_titles})

# get case descriptions function
@app.route("/get-case-description", methods=["GET"])
def get_case_description():
    """Fetch case description based on the selected case title."""
    logger.info("Fetching case description.")
    case_title = request.args.get("title")
    case_title = urllib.parse.unquote(case_title)  # Decode the URL-encoded title
    with open('cases.json', 'r') as file:
        cases = json.load(file)
    for case in cases:
        if case["title"] == case_title:
            return jsonify({"description": case["description"]})
    return jsonify({"description": ""})

# split sentences function
def split_sentences(text):
    logger.info("Splitting sentences.")
    # separate text by "."
    sentences = text.split(".")
    # add the dots back to the end of each sentence
    sentences = [sentence + "." for sentence in sentences]

    # loop through each sentence and check whether it ends with an abbreviation, a number, or a single letter. If it does, merge it with the next sentence
    i = 0
    while i < len(sentences) - 1:
        last_word = sentences[i].split()[-1]
        if last_word in abbreviations or last_word.replace(".","").isdigit() or len(last_word) < 3:
            sentences[i] = sentences[i] + sentences[i + 1]
            sentences.pop(i + 1)
        else:
            i += 1

    # filter out any empty sentences or senteces that only contain a dot
    sentences = [sentence for sentence in sentences if sentence.strip() and sentence.strip() != "."]
    return sentences

def extract_legal_texts(text, legal_texts_abbreviations):
    """
    Extract legal texts and their paragraphs mentioned in the input text.
    The function looks for citations starting with '§' or 'Art.' and ending with
    elements in the legal_texts_abbreviations list.

    Args:
        text (str): The input text containing legal citations.
        legal_texts_abbreviations (list): A list of abbreviations marking the end of legal citations.

    Returns:
        list: A list of unique legal citations found in the text, or None if none are found.
    """
    logger.info("Extracting legal texts.")
    citations = []
    sentences = split_sentences(text)
    
    for sentence in sentences:
        logger.info(f"Processing sentence: {sentence}")
        words = sentence.split()
        i = 0
        while i < len(words):
            if words[i].startswith('§') or words[i].startswith('Art.'):
                citation = words[i]  # Start a new citation
                i += 1
                # Collect subsequent words until a valid ending abbreviation is found
                while i < len(words) and not any(words[i].rstrip('.').rstrip(':').rstrip(',').rstrip(')').endswith(abbrev) for abbrev in legal_texts_abbreviations):
                    citation += ' ' + words[i]
                    i += 1
                # Append the final word that ends the citation
                if i < len(words):
                    citation += ' ' + words[i]
                    citations.append(citation.strip())
            # Move to the next word to continue searching
            i += 1

    # Remove duplicates while preserving order
    citations = list(dict.fromkeys(citations))
    
    # Log detected citations
    if citations:
        logger.info("Detected legal citations in the input:")
        for citation in citations:
            logger.info("- " + citation)
    else:
        citations = None
        logger.info("No legal citations detected in the input.")
    
    return citations

# if __name__ == "__main__":
if __name__ == "__main__":
    try:
        logger.info("Starting the server.")
        socketio.run(app, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("Server stopped by the user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)