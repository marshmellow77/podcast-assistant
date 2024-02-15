import streamlit as st
import feedparser
import os
import time
import requests
from whisper_module.whisper import transcribe
from mutagen.mp3 import MP3
import threading
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_google_vertexai import VertexAI
import datetime
import hashlib


if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""

# --- Utility Functions ---


def search_podcasts(search_query="science and technology"):  # Use a default query
    """Searches the Podcast Index API for podcasts matching a query.

    Args:
        search_query: The query to search for.

    Returns:
        A list of dictionaries containing podcast titles and URLs.
    """

    base_url = "https://api.podcastindex.org/api/1.0/search/byterm?q="

    load_dotenv()
    api_key = os.getenv("PCI_API_KEY")
    api_secret = os.getenv("PCI_API_SECRET")

    # Construct authentication headers
    epoch_time = int(datetime.datetime.now().timestamp())
    data_to_hash = api_key + api_secret + str(epoch_time)
    sha_1 = hashlib.sha1(data_to_hash.encode()).hexdigest()

    headers = {
        "X-Auth-Date": str(epoch_time),
        "X-Auth-Key": api_key,
        "Authorization": sha_1,
        "User-Agent": "postcasting-index-python-cli",
    }

    # Perform the search request
    response = requests.get(base_url + search_query, headers=headers)

    if response.status_code == 200:
        data = response.json()
        results = []
        for feed in data["feeds"]:
            results.append({"title": feed["title"], "url": feed["url"]})
        return results
    else:
        print(f"Error: Received status code {response.status_code}")
        return []


def download_episode(url, download_path):
    """Downloads the episode using requests."""
    with st.spinner("Downloading episode..."):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(download_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            return True
        else:
            return False


def transcribe_episode(audio_path, transcription_path, transcription_complete):
    text = transcribe(audio_path)["text"]

    transcription_complete["status"] = True
    # st.session_state.transcribed_text = text

    os.makedirs("transcripts", exist_ok=True)

    with open(transcription_path, "w") as file:
        file.write(text)


def countdown_timer(duration, transcription_complete):
    countdown_text = st.empty()
    i = duration
    while not transcription_complete["status"]:
        # if i % 10 == 0 or i <= 10 or i == duration:
        countdown_text.text(f"Transcribing... (approx. {i} seconds remaining)")
        time.sleep(1)
        i -= 1
    countdown_text.text("Transcription complete!")


def save_transcription(text, output_file):
    """Saves the transcription to a file."""
    with open(output_file, "w") as file:
        file.write(text)


def init_directories():
    """Initializes required directories."""
    os.makedirs("downloads", exist_ok=True)
    os.makedirs("transcripts", exist_ok=True)


def setup_chatbot(model_id, model_constructor):
    """Sets up and returns the chatbot model."""
    llm = model_constructor(model_name=model_id, temperature=0)
    memory = ConversationBufferMemory()
    return ConversationChain(llm=llm, memory=memory)


# --- Main App Functions ---


def main_podcast_downloader_transcriber():
    st.title("Podcast Downloader & Transcriber")

    search_query = st.text_input("Search for a podcast")

    if search_query:
        search_results = search_podcasts(search_query)
        if search_results:
            # Extract titles for the selectbox
            podcast_titles = [podcast["title"] for podcast in search_results]
            selected_podcast_title = st.selectbox("Select a podcast", podcast_titles)

            # Find the URL of the selected podcast
            selected_podcast_url = next(
                (
                    item["url"]
                    for item in search_results
                    if item["title"] == selected_podcast_title
                ),
                None,
            )

            if selected_podcast_url:
                # Parse the RSS feed of the selected podcast
                feed = feedparser.parse(selected_podcast_url)
                episodes = sorted(
                    feed.entries, key=lambda ep: ep.published_parsed, reverse=True
                )
                episode_titles = [episode.title for episode in episodes]

                if episode_titles:
                    # Step 2: Select an episode from the selected podcast
                    episode_choice = st.selectbox("Select an episode", episode_titles)

                    if st.button("Download and Transcribe"):
                        episode = next(
                            (ep for ep in episodes if ep.title == episode_choice), None
                        )
                        if episode and episode.enclosures:
                            episode_url = episode.enclosures[0].href
                            filename = episode.title.replace("/", "_") + ".mp3"
                            os.makedirs("downloads", exist_ok=True)
                            download_path = os.path.join("downloads", filename)
                            transcript_path = download_path.replace(
                                ".mp3", ".txt"
                            ).replace("downloads", "transcripts")

                            if os.path.exists(download_path):
                                st.warning(
                                    "Episode already downloaded - loading audio from file."
                                )
                            else:
                                if download_episode(episode_url, download_path):
                                    st.success("Episode downloaded!")
                                else:
                                    st.error("Failed to download episode.")

                            if os.path.exists(transcript_path):
                                st.warning(
                                    "Episode already transcribed - loading transcription from file."
                                )
                                f = open(transcript_path)
                                st.session_state.transcribed_text = f.read()
                                # transcription_complete["status"] = True
                                st.session_state.transcription_complete = True
                            else:
                                audio_length = MP3(download_path).info.length
                                st.write(
                                    f"Length of podcast episode: {audio_length/60:.1f} minutes"
                                )

                                duration = int(audio_length / 38)
                                st.write(
                                    f"Estimating ~{duration} seconds (with ~38x speed) for transcribing ..."
                                )

                                transcription_complete = {"status": False}

                                transcription_thread = threading.Thread(
                                    target=transcribe_episode,
                                    args=(
                                        download_path,
                                        transcript_path,
                                        transcription_complete,
                                    ),
                                )
                                transcription_thread.start()

                                countdown_timer(duration, transcription_complete)

                                transcription_thread.join()
                                f = open(transcript_path)
                                st.session_state.transcribed_text = f.read()
                                st.session_state.transcription_complete = True


def main_chatbot():
    # if transcription_complete["status"]:
    st.title("Chatbot for Podcast Discussion")

    load_dotenv()
    # Define the models, their IDs, and their constructors
    models = {
        "Gemini Pro": {"id": "gemini-pro", "constructor": VertexAI},
        # "GPT-4": {"id": "gpt-4-turbo-preview", "constructor": ChatOpenAI},
        "Mixtral 8x7B": {"id": "mistral-small", "constructor": ChatMistralAI},
    }

    # Dropdown in the sidebar
    model_names = [""] + list(models.keys())  # Adding an empty field at the start
    if "model_name" not in st.session_state:
        st.session_state.model_name = ""

    previous_model = st.session_state.model_name
    st.session_state.model_name = st.sidebar.selectbox(
        "Choose a model",
        model_names,
        index=model_names.index(st.session_state.model_name),
    )

    if "chatchain" not in st.session_state:
        st.session_state.chatchain = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model_selected" not in st.session_state:
        st.session_state.model_selected = False

    # Load the model when a valid model is selected
    if st.session_state.model_name and st.session_state.model_name in models:
        if st.session_state.model_name != previous_model:
            st.session_state.model_selected = True
            model_id = models[st.session_state.model_name]["id"]
            model_constructor = models[st.session_state.model_name]["constructor"]

            @st.cache_resource
            def load_chain():
                llm = model_constructor(model_name=model_id, temperature=0)
                memory = ConversationBufferMemory()
                chain = ConversationChain(llm=llm, memory=memory)
                return chain

            st.session_state.chatchain = load_chain()

            # Display the selected model in a disabled dropdown
            st.sidebar.selectbox(
                "Model selected", [st.session_state.model_name], disabled=True
            )

            system_prompt = f"""You are a helpful assistant that excels in analysing transcripts for the user.
Below is the transcript of a podcast episode. Answer all queries from the user truthfully. Rely only on the information given in the transcript. Do not make things up. It is ok to say "There is no information about this in the transcript."

=== BEGIN TRANSCRIPT ===
{st.session_state.transcribed_text}
=== END TRANSCRIPT ===

Please confirm that you have analysed the transcript and you are ready to start chatting by saying "I have analysed the transcript, feel free to ask any questions about it."
"""

            # content = "=== BEGIN FILE ===\n"
            # content += st.session_state.transcribed_text
            # content += "\n=== END FILE ===\nPlease confirm that you have analysed the audio content by saying 'Yes, I have analysed the audio content.'. Use British English spelling."
            output = st.session_state.chatchain(system_prompt)["response"]
            # st.session_state.messages.append(
            #     {
            #         "role": "user",
            #         "content": "I have uploaded an audio file. Please confirm that you have read the transcripts.",
            #     }
            # )
            st.session_state.messages.append({"role": "assistant", "content": output})

    # Chat input and response logic
    if st.session_state.chatchain:
        if prompt := st.chat_input("What is up?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            output = st.session_state.chatchain(prompt)["response"]
            st.session_state.messages.append({"role": "assistant", "content": output})

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Display a placeholder or message if no model is selected yet
    if not st.session_state.model_selected:
        st.sidebar.write("Please select a model to start the chat.")


# --- App Entry Point ---

if __name__ == "__main__":

    if "transcription_complete" not in st.session_state:
        st.session_state.transcription_complete = False

    if "chatbot_started" not in st.session_state:
        st.session_state.chatbot_started = False

    init_directories()  # Make sure required directories exist

    if not st.session_state.chatbot_started:
        main_podcast_downloader_transcriber()
    if st.session_state.transcription_complete:
        if not st.session_state.chatbot_started:
            if st.button("Start Chatbot"):
                st.session_state.chatbot_started = True
                st.rerun()
        else:
            main_chatbot()
