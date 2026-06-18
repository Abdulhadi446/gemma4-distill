#!/usr/bin/env python3
"""
Gemma 4 12B Distillation Data Generator
=========================================
Generates high-quality prompt-response pairs across all supported
modalities (text, image, audio, video, mixed) using the Gemma 4 12B
instruction-tuned model with 4-bit quantization.

Output: JSONL files per modality under ./data/
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID = "google/gemma-4-12B-it"
OUTPUT_DIR = Path("./data")
ERROR_LOG = OUTPUT_DIR / "errors.log"
REJECTED_LOG = OUTPUT_DIR / "rejected.jsonl"

SAMPLES_PER_MODALITY = {
    "text": 30,
    "image": 15,
    "audio": 10,
    "video": 10,
    "mixed": 10,
}

GEN_KWARGS = dict(
    max_new_tokens=1024,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    do_sample=True,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("distill")


def log_error(msg: str, exc: bool = True) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ERROR_LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}\n")
        if exc:
            import traceback

            traceback.print_exc(file=f)
    logger.error(msg)


# ---------------------------------------------------------------------------
# Quality scoring heuristic
# ---------------------------------------------------------------------------
def quality_score(response_text: str) -> float:
    """Return a score in [0, 1] based on length, repetition, and structure."""
    if not response_text or len(response_text.strip()) < 20:
        return 0.0

    text = response_text.strip()
    words = text.split()
    sentences = text.replace("!",".").replace("?",".").split(".")

    # Length bonus: prefer 50-2000 chars
    length = len(text)
    if length < 50:
        length_score = 0.2
    elif length < 100:
        length_score = 0.5
    elif length <= 2000:
        length_score = 1.0
    else:
        length_score = 0.8  # very long might be rambling

    # Repetition penalty: repeated n-grams
    trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    if trigrams:
        unique_ratio = len(set(trigrams)) / len(trigrams)
        repetition_penalty = min(1.0, unique_ratio * 1.5)
    else:
        repetition_penalty = 1.0

    # Sentence variety: prefer 2+ sentences
    sentence_count = max(1, len([s for s in sentences if len(s.strip()) > 5]))
    variety_score = min(1.0, sentence_count / 6.0)

    # Reasoning token presence (bonus for structured thinking)
    reasoning_bonus = 0.0
    for marker in ["<|think|>", "<channel|>", "reason", "step", "therefore",
                   "because", "since", "first", "second", "finally"]:
        if marker in text.lower():
            reasoning_bonus = 0.1
            break

    score = 0.35 * length_score + 0.35 * repetition_penalty + 0.20 * variety_score + 0.10 * reasoning_bonus
    return round(min(1.0, max(0.0, score)), 4)


# ---------------------------------------------------------------------------
# Prompt templates (high-quality, diverse)
# ---------------------------------------------------------------------------
TEXT_PROMPTS = [
    # Reasoning
    (
        "Solve the following problem step by step. "
        "A train leaves Station A at 9:00 AM traveling at 80 km/h towards Station B. "
        "Another train leaves Station B at 9:30 AM traveling at 100 km/h towards Station A. "
        "The distance between the stations is 450 km. At what time will the two trains meet, "
        "and how far from Station A will they be?"
    ),
    (
        "A bag contains 5 red marbles, 3 blue marbles, and 2 green marbles. "
        "If you draw two marbles without replacement, what is the probability that "
        "both marbles are the same color? Show your reasoning step by step."
    ),
    (
        "Explain the logical fallacy in the following argument: "
        "'Every person who has ever eaten tomatoes has died eventually. "
        "Therefore, tomatoes are poisonous and should be avoided.' "
        "Identify the fallacy type and construct a corrected version."
    ),
    (
        "You have a 3-gallon jug and a 5-gallon jug. How can you measure exactly 4 gallons "
        "of water? Describe each step clearly."
    ),
    (
        "Compare and contrast the time complexity and space complexity of quicksort "
        "and mergesort. Under what conditions would you prefer one over the other? "
        "Include best, average, and worst-case analysis."
    ),

    # Coding
    (
        "Write a Python function that takes a list of integers and returns the "
        "length of the longest increasing subsequence. Your solution should run in "
        "O(n log n) time. Include a brief explanation of how the algorithm works."
    ),
    (
        "Implement a rate limiter in Python that limits API calls to 100 requests "
        "per minute per user. Use a sliding window approach. Show a complete class "
        "implementation with thread safety."
    ),
    (
        "Write a Python function to serialize and deserialize a binary tree. "
        "Your serialization should use a compact string format, and your "
        "deserialization should reconstruct the exact tree. Include type hints."
    ),
    (
        "Review this code for bugs and style issues:\n\n"
        "def calc(x, y):\n"
        "    if x = 0:\n"
        "        return y + 1\n"
        "    elif x > 0 and y = 0:\n"
        "        return calc(x - 1, 1)\n"
        "    else:\n"
        "        return calc(x - 1, calc(x, y - 1))\n\n"
        "Identify all issues and provide a corrected, well-documented version."
    ),
    (
        "Design a simple in-memory key-value cache with TTL support in Python. "
        "Support get, set, delete, and expire operations. The cache should evict "
        "expired entries lazily. Provide the full implementation."
    ),

    # Math
    (
        "Prove that the square root of 2 is irrational. Provide a clear, "
        "step-by-step proof by contradiction."
    ),
    (
        "Compute the definite integral of x^2 * sin(x) from 0 to pi. "
        "Show all steps including integration by parts."
    ),
    (
        "Find all real solutions to the equation: x^4 - 5x^2 + 4 = 0. "
        "Show your work and explain each transformation."
    ),
    (
        "A sequence is defined by a_1 = 1, a_2 = 1, and a_n = a_{n-1} + a_{n-2} "
        "for n > 2. Prove by induction that a_n = (phi^n - psi^n) / sqrt(5) "
        "where phi = (1 + sqrt(5))/2 and psi = (1 - sqrt(5))/2."
    ),
    (
        "Explain the concept of eigenvalues and eigenvectors geometrically. "
        "Give a real 2x2 matrix example and show how to compute its eigenvalues "
        "and eigenvectors step by step."
    ),

    # Instruction following / creative
    (
        "Write a formal email to a professor requesting a recommendation letter. "
        "You should: (1) remind them who you are and which class you took, "
        "(2) explain why you need the letter, (3) offer to provide your resume "
        "and personal statement, (4) mention the deadline, and (5) express gratitude."
    ),
    (
        "Summarize the key differences between supervised learning, unsupervised learning, "
        "and reinforcement learning. For each paradigm, give two real-world applications "
        "and one common algorithm."
    ),
    (
        "Explain the concept of 'attention is all you need' to a high school student. "
        "Use analogies and avoid jargon. Focus on why attention mechanisms were "
        "revolutionary for sequence modeling."
    ),
    (
        "Write a detailed comparison of HTTP/2 and HTTP/3 protocols. Cover: "
        "multiplexing, header compression, connection establishment, "
        "and use cases where each excels."
    ),
    (
        "Design a microservices architecture for an e-commerce platform. "
        "Describe the service boundaries, inter-service communication patterns, "
        "data management strategy, and how you would handle distributed transactions. "
        "Include a diagram-compatible ASCII description."
    ),
    (
        "Explain the CAP theorem in distributed systems. For each pair of consistency, "
        "availability, and partition tolerance, give a concrete database example that "
        "sacrifices the third property."
    ),
    (
        "Write a Python script that fetches data from a paginated REST API, "
        "handles retries with exponential backoff, and saves results to a SQLite "
        "database. Include proper error handling and logging."
    ),
    (
        "Describe how gradient descent works for training neural networks. "
        "Include the concepts of: loss function, gradient computation, learning rate, "
        "momentum, and adaptive learning rates (Adam). Use mathematical notation where appropriate."
    ),
    (
        "Translate the following English sentence into French, German, Spanish, and Japanese: "
        "'The rapid advancements in artificial intelligence are transforming how we interact "
        "with technology, but they also raise important ethical questions about privacy, bias, "
        "and accountability.' Then explain any culturally-specific translation choices you made."
    ),
    (
        "Write a SQL query to find employees who earn more than their department's average salary. "
        "Then write the same query using a window function. Explain which approach is more efficient and why."
    ),

    # Additional advanced prompts
    (
        "Explain the difference between L1 and L2 regularization. Show how each "
        "affects the weight updates during gradient descent and why L1 leads to "
        "sparse solutions. Include the mathematical formulations."
    ),
    (
        "Given a binary tree where each node contains a digit (0-9), find the sum of all "
        "root-to-leaf numbers. For example, the tree [1,2,3] represents the numbers 12 and 13 "
        "and the sum is 25. Provide a recursive and an iterative solution in Python."
    ),
    (
        "What is the P vs NP problem? Explain it in simple terms, then discuss why it matters "
        "for cryptography, optimization, and machine learning. Provide examples of NP-complete "
        "problems and their real-world significance."
    ),
    (
        "Design a fault-tolerant distributed file system. Discuss: data replication strategies, "
        "consensus algorithms, handling network partitions, and consistency models. "
        "Compare with existing systems like HDFS and Ceph."
    ),
    (
        "Write a Python implementation of the A* pathfinding algorithm for a 2D grid with obstacles. "
        "The implementation should accept a grid, start, and goal, and return the optimal path. "
        "Include Manhattan distance heuristic and visualize the path."
    ),
]

IMAGE_PROMPTS = [
    {
        "prompt": "Transcribe all the text visible in this document image. Preserve the original formatting including line breaks and indentation. If any text is handwritten, mark it with [handwritten] tags.",
        "image_url": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/GoldenGate.png",
    },
    {
        "prompt": "Describe this image in detail. Include: the main subject, setting, colors, composition, lighting, mood, and any text visible. Then explain what story or message the image conveys.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/NASA_earth_observatory_%28astronaut_photograph%29.jpg/800px-NASA_earth_observatory_%28astronaut_photograph%29.jpg",
    },
    {
        "prompt": "Analyze this photograph. What time of day was it taken? What season? What geographic region might this be? Identify any landmarks, vegetation, or architectural styles visible. Justify each inference.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Mountain_View%2C_California_skyline.jpg/800px-Mountain_View%2C_California_skyline.jpg",
    },
    {
        "prompt": "Read and extract all structured information from this image. If there is a table, list each row and column. If there is a chart, describe the axes, data trends, and key values. If there is text, transcribe it faithfully.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Global_Temperature_Anomaly_1880-2020.svg/800px-Global_Temperature_Anomaly_1880-2020.svg.png",
    },
    {
        "prompt": "This appears to be a piece of artwork or an illustration. Describe the artistic style, technique, and influences you observe. What period or movement does it belong to? What symbolism or themes are present?",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF.jpg/800px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF.jpg",
    },
    {
        "prompt": "What is shown in this image? Identify all objects, people, and activities visible. Describe the spatial relationships between elements and provide a comprehensive scene understanding.",
        "image_url": "https://raw.githubusercontent.com/huggingface/datasets/main/docs/source/en/imgs/datasets.jpg",
    },
    {
        "prompt": "Extract the text from this screenshot or UI image. Then describe the layout: what elements are clickable, what information is displayed, and what the user can do on this screen.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Wikipedia_Mobile_Web_2022.png/800px-Wikipedia_Mobile_Web_2022.png",
    },
    {
        "prompt": "Analyze this scientific figure or diagram. What field of science does it relate to? Identify all labeled components, the relationships between them, and summarize what finding or concept the figure communicates.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/DNA_Structure%2BKey%2BLabelled.png/800px-DNA_Structure%2BKey%2BLabelled.png",
    },
    {
        "prompt": "This appears to be a map or satellite image. Identify geographic features, landmarks, and any annotations. Describe what this location might be used for and what nearby points of interest exist.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/World_map_%28orthographic_projection%29.svg/800px-World_map_%28orthographic_projection%29.svg.png",
    },
    {
        "prompt": "Describe the food or objects in this image. Identify ingredients, preparation style, and cultural origin. Assess the visual quality, presentation, and what sensory experience it suggests.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Good_Food_Display_-_NCI_Visuals_Online.jpg/800px-Good_Food_Display_-_NCI_Visuals_Online.jpg",
    },
    {
        "prompt": "Read and transcribe any text in this image with high precision. Then translate the text into English. Finally, describe the visual context: what kind of document or scene is this, and what is its purpose?",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/JPEG_example_flower.jpg/800px-JPEG_example_flower.jpg",
    },
    {
        "prompt": "Analyze the people in this image. Estimate their ages, emotional states, relationships to each other, and the social context. What event or situation is captured? Support your observations with visual evidence.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Banquet_after_wedding_feast_in_Tripoli.jpg/800px-Banquet_after_wedding_feast_in_Tripoli.jpg",
    },
    {
        "prompt": "Examine this image for text content. Read any labels, signs, or written material visible. Then describe the broader scene: location, activity, cultural context, and any story the image tells.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Times_Square_New_York_City_%28HDR%29.jpg/800px-Times_Square_New_York_City_%28HDR%29.jpg",
    },
    {
        "prompt": "Compare and contrast the different visual elements in this composition. Discuss foreground vs background, use of color and contrast, lighting direction, depth of field, and how these technical choices affect the viewer's perception.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/Shaolin_monk_burning_incense_2010.jpg/800px-Shaolin_monk_burning_incense_2010.jpg",
    },
    {
        "prompt": "This image contains a visual representation of data or information. Describe the type of visualization, identify all variables plotted, interpret the key trends or patterns, and state the main conclusion a reader should draw.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Internet_users_per_100_inhabitants_ITU.svg/800px-Internet_users_per_100_inhabitants_ITU.svg.png",
    },
]

AUDIO_PROMPTS = [
    {
        "prompt": "Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.",
        "audio_url": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/journal1.wav",
    },
    {
        "prompt": "Listen to this audio clip and transcribe it. Then answer: What is the speaker's topic? What is their main argument or point? What tone or emotion do they convey? Provide evidence from the audio for each answer.",
        "audio_url": "https://www2.cs.uic.edu/~i101/SoundFiles/preamble10.wav",
    },
    {
        "prompt": "Transcribe the speech in this audio clip. After transcription, analyze the speaker's accent or dialect. What regional or social background might they have? What clues in pronunciation, vocabulary, or rhythm support your analysis?",
        "audio_url": "https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg10.wav",
    },
    {
        "prompt": "Listen to this audio and provide: (1) a verbatim transcription, (2) the language being spoken, (3) the estimated number of speakers, (4) the gender and approximate age of each speaker, and (5) the emotional tone of the conversation.",
        "audio_url": "https://www2.cs.uic.edu/~i101/SoundFiles/CantinaBand3.wav",
    },
    {
        "prompt": "Transcribe the following speech segment. Then summarize the content in 2-3 sentences. Finally, identify any key terms, names, or technical vocabulary used and explain what they refer to.",
        "audio_url": "https://www2.cs.uic.edu/~i101/SoundFiles/StarWars3.wav",
    },
    {
        "prompt": "Audio Speech Recognition: Transcribe this audio in English. Focus on accuracy — capture every word, including hesitations, repetitions, and false starts. Mark unclear segments with [unclear].",
        "audio_url": "https://www2.cs.uic.edu/~i101/SoundFiles/preamble10.wav",
    },
    {
        "prompt": "This audio contains someone reading a famous historical document. Identify which document it is, who wrote it, when it was written, and its historical significance. Then transcribe the portion you hear.",
        "audio_url": "https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg10.wav",
    },
    {
        "prompt": "Analyze the acoustic properties of this audio sample. Describe: the recording quality, background noise level, number of channels, approximate duration, and any artifacts or distortions you detect. Then transcribe any speech content.",
        "audio_url": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/journal1.wav",
    },
    {
        "prompt": "Identify the language(s) spoken in this audio. Provide a transcription in the original script and a translation into English. Then describe the speaker's communicative intent: are they informing, persuading, entertaining, or something else?",
        "audio_url": "https://www2.cs.uic.edu/~i101/SoundFiles/CantinaBand60.wav",
    },
    {
        "prompt": "This is a multi-speaker audio clip. Transcribe each speaker's lines separately, labeling them as Speaker 1, Speaker 2, etc. Describe the relationship between speakers and the nature of their interaction.",
        "audio_url": "https://www2.cs.uic.edu/~i101/SoundFiles/StarWars3.wav",
    },
]

VIDEO_PROMPTS = [
    {
        "prompt": "Describe this video in detail. Cover: the setting, characters or objects visible, actions occurring, the sequence of events, and the overall mood or atmosphere. Use a chronological structure for your description.",
        "video_url": "https://github.com/bebechien/gemma/raw/refs/heads/main/videos/ForBiggerBlazes.mp4",
    },
    {
        "prompt": "Watch this video and answer: What is the main activity or event shown? Where and when does it take place (time of day, indoor/outdoor)? Who or what are the main subjects? Tell a coherent narrative of what happens from start to finish.",
        "video_url": "https://github.com/bebechien/gemma/raw/refs/heads/main/videos/ForBiggerBlazes.mp4",
    },
    {
        "prompt": "Perform temporal reasoning on this video. List the key events in chronological order with approximate timestamps. Identify any cause-effect relationships between events. What happens first, what follows, and what is the final outcome?",
        "video_url": "https://github.com/bebechien/gemma/raw/refs/heads/main/videos/ForBiggerBlazes.mp4",
    },
    {
        "prompt": "Analyze the visual composition of this video. Discuss camera angles, lighting, color palette, motion, and editing style. How do these cinematographic choices support the narrative or message?",
        "video_url": "https://github.com/bebechien/gemma/raw/refs/heads/main/videos/ForBiggerBlazes.mp4",
    },
    {
        "prompt": "Watch this video carefully and answer specific questions: (1) How many distinct scenes or shots are there? (2) What objects appear in more than one scene? (3) Is there any text visible on screen? (4) Does the setting change between scenes?",
        "video_url": "https://github.com/bebechien/gemma/raw/refs/heads/main/videos/ForBiggerBlazes.mp4",
    },
    {
        "prompt": "Describe the people or characters in this video. Estimate their ages, emotions, and relationships. What are they doing and why? Describe their body language, facial expressions, and interactions.",
        "video_url": "https://github.com/bebechien/gemma/raw/refs/heads/main/videos/ForBiggerBlazes.mp4",
    },
    {
        "prompt": "Watch this video and create a detailed scene-by-scene storyboard description. For each scene, note: duration, camera position, main subjects, action, dialogue (if any), and transitions. End with an overall summary.",
        "video_url": "https://github.com/bebechien/gemma/raw/refs/heads/main/videos/ForBiggerBlazes.mp4",
    },
    {
        "prompt": "Analyze the motion dynamics in this video. Describe: direction and speed of movement, any patterns or trajectories, how objects enter and exit the frame, and whether the motion appears natural or staged.",
        "video_url": "https://github.com/bebechien/gemma/raw/refs/heads/main/videos/ForBiggerBlazes.mp4",
    },
    {
        "prompt": "Extract all actionable information from this video. List: visible text (signs, labels), identifiable locations or landmarks, any demonstrated process or procedure, and safety-relevant observations.",
        "video_url": "https://github.com/bebechien/gemma/raw/refs/heads/main/videos/ForBiggerBlazes.mp4",
    },
    {
        "prompt": "This video shows a sequence of events. Identify the beginning, middle, and end. What is the inciting incident? What is the climax or turning point? How does it resolve? Use specific visual evidence from the video to support your analysis.",
        "video_url": "https://github.com/bebechien/gemma/raw/refs/heads/main/videos/ForBiggerBlazes.mp4",
    },
]

MIXED_PROMPTS = [
    # Image + Audio
    {
        "prompt": "Describe how the audio content relates to the image shown. First describe what you see in the image, then summarize what you hear in the audio, and finally explain the connection between them.",
        "image_url": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/GoldenGate.png",
        "audio_url": "https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/apps/sample-data/journal1.wav",
    },
    {
        "prompt": "You are given an image and an audio clip. Does the audio describe or correspond to the image? Explain your reasoning. If they are unrelated, say so and describe each independently.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/NASA_earth_observatory_%28astronaut_photograph%29.jpg/800px-NASA_earth_observatory_%28astronaut_photograph%29.jpg",
        "audio_url": "https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg10.wav",
    },
    # Image + Text (complex reasoning)
    {
        "prompt": "Study this image carefully. Then answer: What type of location is shown? What period or era does it represent? What cultural or historical significance does it have? Provide a detailed analysis referencing specific visual elements.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Times_Square_New_York_City_%28HDR%29.jpg/800px-Times_Square_New_York_City_%28HDR%29.jpg",
    },
    {
        "prompt": "Read the text in this image and cross-reference it with what you see visually. Is the text descriptive, instructive, or unrelated to the image? Provide a thorough analysis of the relationship between text and visual content.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Global_Temperature_Anomaly_1880-2020.svg/800px-Global_Temperature_Anomaly_1880-2020.svg.png",
    },
    # Audio + Image
    {
        "prompt": "A person is describing something in the audio while an image is provided. Does the audio description match the visual content? Point out specific correspondences or discrepancies.",
        "image_url": "https://raw.githubusercontent.com/huggingface/datasets/main/docs/source/en/imgs/datasets.jpg",
        "audio_url": "https://www2.cs.uic.edu/~i101/SoundFiles/preamble10.wav",
    },
    # Image sequence understanding (text + image mixed via interleaved)
    {
        "prompt": "Here are two images. Compare them: what elements are common, what has changed, and what story do they tell together? Consider composition, content, mood, and any narrative arc.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF.jpg/800px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF.jpg",
    },
    {
        "prompt": "Combine what you learn from this image and this audio to answer: what is the broader context? Where might this scene and audio be from? What is happening just before and just after this moment?",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Good_Food_Display_-_NCI_Visuals_Online.jpg/800px-Good_Food_Display_-_NCI_Visuals_Online.jpg",
        "audio_url": "https://www2.cs.uic.edu/~i101/SoundFiles/CantinaBand3.wav",
    },
    {
        "prompt": "This image contains a diagram or schematic. Describe what it represents. Then imagine you are explaining this diagram in a tutorial — what would you say? Write a clear, pedagogical explanation suitable for a beginner.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/DNA_Structure%2BKey%2BLabelled.png/800px-DNA_Structure%2BKey%2BLabelled.png",
    },
    {
        "prompt": "Analyze the cultural or social context of this image. What does it reveal about the society, time period, or community depicted? Reference specific visual details. Then, if the audio provides additional context, incorporate it.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Banquet_after_wedding_feast_in_Tripoli.jpg/800px-Banquet_after_wedding_feast_in_Tripoli.jpg",
        "audio_url": "https://www2.cs.uic.edu/~i101/SoundFiles/CantinaBand60.wav",
    },
    {
        "prompt": "Consider this image and imagine a news story that goes with it. Write a brief news article (3-4 paragraphs) that the image could illustrate. Include a headline, dateline, and quote from someone you imagine might be involved.",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/Shaolin_monk_burning_incense_2010.jpg/800px-Shaolin_monk_burning_incense_2010.jpg",
    },
]


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
def build_messages(content_list: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": "You are a helpful, accurate, and thoughtful assistant."},
        {"role": "user", "content": content_list},
    ]


def generate_sample(
    model: AutoModelForMultimodalLM,
    processor: AutoProcessor,
    messages: list[dict],
) -> str | None:
    """Run generation; return the parsed response text or None on failure."""
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = model.generate(**inputs, **GEN_KWARGS)

        raw_response = processor.decode(
            outputs[0][input_len:], skip_special_tokens=False
        )
        parsed = processor.parse_response(raw_response)

        if isinstance(parsed, dict):
            return parsed.get("response", "") or parsed.get("text", "")
        return str(parsed) if parsed else None

    except torch.cuda.OutOfMemoryError:
        log_error("CUDA OOM — clearing cache and skipping")
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        log_error(f"Generation error: {e}")
        return None


def save_sample(path: Path, sample: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def save_rejected(sample: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REJECTED_LOG, "a") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Modality generators
# ---------------------------------------------------------------------------
def generate_text_samples(model, processor, device, num):
    out_path = OUTPUT_DIR / "text.jsonl"
    samples, rejected = [], 0

    for prompt_text in tqdm(TEXT_PROMPTS[:num], desc="Text", unit="sample"):
        messages = build_messages([{"type": "text", "text": prompt_text}])
        response = generate_sample(model, processor, messages)
        if not response:
            rejected += 1
            save_rejected({"modality": "text", "prompt": prompt_text, "reason": "generation failed"})
            continue

        score = quality_score(response)
        entry = {
            "messages": [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": response},
            ],
            "modality": "text",
            "metadata": {"source": "gemma4-12b-distill", "quality_score": score},
        }

        if score >= 0.4:
            save_sample(out_path, entry)
            samples.append(entry)
        else:
            rejected += 1
            save_rejected(entry)

    return samples, rejected


def generate_image_samples(model, processor, device, num):
    out_path = OUTPUT_DIR / "image.jsonl"
    samples, rejected = [], 0

    for item in tqdm(IMAGE_PROMPTS[:num], desc="Image", unit="sample"):
        content = [
            {"type": "image", "url": item["image_url"]},
            {"type": "text", "text": item["prompt"]},
        ]
        messages = build_messages(content)
        response = generate_sample(model, processor, messages)
        if not response:
            rejected += 1
            save_rejected({"modality": "image", "prompt": item["prompt"], "reason": "generation failed"})
            continue

        score = quality_score(response)
        entry = {
            "messages": [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": response},
            ],
            "modality": "image",
            "metadata": {"source": "gemma4-12b-distill", "quality_score": score},
        }

        if score >= 0.4:
            save_sample(out_path, entry)
            samples.append(entry)
        else:
            rejected += 1
            save_rejected(entry)

    return samples, rejected


def generate_audio_samples(model, processor, device, num):
    out_path = OUTPUT_DIR / "audio.jsonl"
    samples, rejected = [], 0

    for item in tqdm(AUDIO_PROMPTS[:num], desc="Audio", unit="sample"):
        content = [
            {"type": "text", "text": item["prompt"]},
            {"type": "audio", "audio": item["audio_url"]},
        ]
        messages = build_messages(content)
        response = generate_sample(model, processor, messages)
        if not response:
            rejected += 1
            save_rejected({"modality": "audio", "prompt": item["prompt"], "reason": "generation failed"})
            continue

        score = quality_score(response)
        entry = {
            "messages": [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": response},
            ],
            "modality": "audio",
            "metadata": {"source": "gemma4-12b-distill", "quality_score": score},
        }

        if score >= 0.4:
            save_sample(out_path, entry)
            samples.append(entry)
        else:
            rejected += 1
            save_rejected(entry)

    return samples, rejected


def generate_video_samples(model, processor, device, num):
    out_path = OUTPUT_DIR / "video.jsonl"
    samples, rejected = [], 0

    for item in tqdm(VIDEO_PROMPTS[:num], desc="Video", unit="sample"):
        content = [
            {"type": "video", "video": item["video_url"]},
            {"type": "text", "text": item["prompt"]},
        ]
        messages = build_messages(content)
        response = generate_sample(model, processor, messages)
        if not response:
            rejected += 1
            save_rejected({"modality": "video", "prompt": item["prompt"], "reason": "generation failed"})
            continue

        score = quality_score(response)
        entry = {
            "messages": [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": response},
            ],
            "modality": "video",
            "metadata": {"source": "gemma4-12b-distill", "quality_score": score},
        }

        if score >= 0.4:
            save_sample(out_path, entry)
            samples.append(entry)
        else:
            rejected += 1
            save_rejected(entry)

    return samples, rejected


def generate_mixed_samples(model, processor, device, num):
    out_path = OUTPUT_DIR / "mixed.jsonl"
    samples, rejected = [], 0

    for item in tqdm(MIXED_PROMPTS[:num], desc="Mixed", unit="sample"):
        content = []
        if "image_url" in item:
            content.append({"type": "image", "url": item["image_url"]})
        content.append({"type": "text", "text": item["prompt"]})
        if "audio_url" in item:
            content.append({"type": "audio", "audio": item["audio_url"]})

        messages = build_messages(content)
        response = generate_sample(model, processor, messages)
        if not response:
            rejected += 1
            save_rejected({"modality": "mixed", "prompt": item["prompt"], "reason": "generation failed"})
            continue

        score = quality_score(response)
        entry = {
            "messages": [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": response},
            ],
            "modality": "mixed",
            "metadata": {"source": "gemma4-12b-distill", "quality_score": score},
        }

        if score >= 0.4:
            save_sample(out_path, entry)
            samples.append(entry)
        else:
            rejected += 1
            save_rejected(entry)

    return samples, rejected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 64)
    print("  Gemma 4 12B Distillation Data Generator")
    print("=" * 64)

    # --- Device check ---
    if not torch.cuda.is_available():
        print("\n[WARNING] CUDA not detected. This script requires a GPU.")
        print("Proceeding anyway — the model load may fail on CPU.\n")

    # --- Create output dir ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load model with 4-bit quantization ---
    print(f"\nLoading {MODEL_ID} with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    device = model.device
    print(f"  Model device: {device}")
    print(f"  Model dtype:  {model.dtype}")
    print()

    # --- Generate per modality ---
    generators = {
        "text": generate_text_samples,
        "image": generate_image_samples,
        "audio": generate_audio_samples,
        "video": generate_video_samples,
        "mixed": generate_mixed_samples,
    }

    summary = {}
    total_accepted = 0
    total_rejected = 0

    for modality, gen_fn in generators.items():
        target = SAMPLES_PER_MODALITY[modality]
        print(f"\n[{modality.upper()}] Generating up to {target} samples...")
        try:
            accepted, rejected = gen_fn(model, processor, device, target)
            summary[modality] = {
                "target": target,
                "accepted": len(accepted),
                "rejected": rejected,
            }
            total_accepted += len(accepted)
            total_rejected += rejected
        except Exception as e:
            log_error(f"Fatal error in {modality} generation: {e}")
            summary[modality] = {"target": target, "accepted": 0, "rejected": 0}

    # --- Summary ---
    print("\n" + "=" * 64)
    print("  GENERATION SUMMARY")
    print("=" * 64)
    for mod, stats in summary.items():
        rej_rate = (stats["rejected"] / max(1, stats["rejected"] + stats["accepted"])) * 100
        print(f"  {mod:8s} | target={stats['target']:2d} | "
              f"accepted={stats['accepted']:2d} | "
              f"rejected={stats['rejected']:2d} | "
              f"rejection_rate={rej_rate:5.1f}%")

    total_target = sum(SAMPLES_PER_MODALITY.values())
    overall_rej_rate = (total_rejected / max(1, total_rejected + total_accepted)) * 100
    print(f"\n  {'TOTAL':8s} | target={total_target:2d} | "
          f"accepted={total_accepted:2d} | "
          f"rejected={total_rejected:2d} | "
          f"rejection_rate={overall_rej_rate:5.1f}%")

    print(f"\n  Output directory: {OUTPUT_DIR.resolve()}")
    print(f"  Error log:        {ERROR_LOG}")
    print(f"  Rejected log:     {REJECTED_LOG}")
    print("=" * 64)

    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
