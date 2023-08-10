import streamlit as st
import tiktoken
import os
from dotenv import load_dotenv

load_dotenv()

askai = ""
streaming = True
max_token_value = 16*1024
nr_event_client = None

if os.getenv("OPENAI_API_KEY"):
    import openai
    askai = "OpenAI"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if os.getenv("OPENAI_API_BASE"):
        openai.api_base =  os.getenv("OPENAI_API_BASE")
    if os.getenv("OPENAI_API_TYPE"):
        openai.api_type = os.getenv("OPENAI_API_TYPE")
    if os.getenv("OPENAI_API_VERSION"):
        openai.api_version = os.getenv("OPENAI_API_VERSION")
    if os.getenv("DEPLOYMENT_NAME"):
        deployment_name = os.getenv("DEPLOYMENT_NAME")
    else:
        deployment_name = "gpt-4-32k"
    if os.getenv("NEW_RELIC_LICENSE_KEY"):
        try:
            from nr_openai_observability import monitor
        except ImportError:
            print("Error importing New Relic OpenAI Observability")
            print("Please install the SDK with: pip install nr-openai-observability")
            exit(1)
        try:
            from newrelic_telemetry_sdk import Event, EventClient
            nr_event_client = EventClient(os.getenv("NEW_RELIC_LICENSE_KEY"))
        except ImportError:
            print("Error importing New Relic Python Telemetry SDK")
            print("Please install the SDK with: pip install newrelic-telemetry-sdk")
            exit(1)
        monitor.initialization(application_name=os.getenv("NEW_RELIC_APPLICATION_NAME"))
        streaming = False # NR monitoring does not currently support streaming
elif os.getenv("ANTHROPIC_API_KEY"):
    askai = "Anthropic"
    from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
    anthropic = Anthropic(
        api_key = os.getenv("ANTHROPIC_API_KEY")
    )
    if os.getenv("DEPLOYMENT_NAME"):
        deployment_name = os.getenv("DEPLOYMENT_NAME")
    else:
        deployment_name = "claude-2"
else:
    print("Must set OpenAI or Anthropic API key")
    exit(1)


if os.getenv("MAX_TOKEN_VALUE"):
    max_token_value = int(os.getenv("MAX_TOKEN_VALUE"))

st.set_page_config(
    page_title="ZoomGPT", page_icon="🎙️", layout="wide",
    initial_sidebar_state="expanded"
)

# Get rid of streamlit menus
st.markdown(
    """
        <style>
            div[data-testid="stToolbar"] {
            visibility: hidden;
            height: 0%;
            position: fixed;
            }
            div[data-testid="stDecoration"] {
            visibility: hidden;
            height: 0%;
            position: fixed;
            }
            div[data-testid="stStatusWidget"] {
            visibility: hidden;
            height: 0%;
            position: fixed;
            }
            #MainMenu {
            visibility: hidden;
            height: 0%;
            }
            header {
            visibility: hidden;
            height: 0%;
            }
            footer {
            visibility: hidden;
            height: 0%;
            }
        </style>""",
unsafe_allow_html=True)

with st.sidebar:
    st.header("ZoomGPT: 🎙️ Ask me anything")

# This holds the prompt
if "query" not in st.session_state:
    st.session_state["query"] = []

# Save the dialogue
if "dialogue" not in st.session_state:
    st.session_state["dialogue"] = []

def submit_prompts():
    st.session_state["query"] = st.session_state["prompt_selection"]

def clear_dialogue():
    try:
        st.session_state["dialogue"] = []
    except:
        pass

def positive(nr_event_client):
    if nr_event_client is None:
        return
    # If we have a NR event client, send a positive response
    event = Event(
        "ZoomGPTResponse", {"response": 1}
    )
    try:
        response = nr_event_client.send(event)
        response.raise_for_status()
    except Exception as e:
        print(f"Error submitting NR event: {e}")
    print(response)

def negative(nr_event_client):
    if nr_event_client is None:
        return 
    # If we have a NR event client, send a negative response
    event = Event(
        "ZoomGPTResponse", {"response": 0}
    )
    try:
        response = nr_event_client.send(event)
        response.raise_for_status()
    except Exception as e:
        print(f"Error submitting NR event: {e}")

with st.sidebar:
    uploaded_file = st.file_uploader(
        "Upload a meeting transcript",
        type=["vtt"],
        accept_multiple_files=False
    )

if not uploaded_file:
    st.stop()

if askai == "OpenAI":
    encoder = tiktoken.encoding_for_model(deployment_name)

with st.sidebar:
    c1, c2 = st.columns(2)
    with c1:
        st.slider("Max tokens in response", min_value=64, max_value=max_token_value, value=1024, step=32, key="max_tokens")
    with c2:
        st.slider("Temperature", min_value=0.01, max_value=1.0, value=0.1, step=0.01, key="temperature")

    st.multiselect("Selection of prompts: ",
        [
            "Who attended the meeting?",
            "What was the meeting agenda?",
            "What were the meetings objectives?",
            "What topics were discussed in the meeting?",
            "What were the key outcomes of this meeting?",
            "What were the followup or action items?",
            "What questions did this meeting answer?",
            "What decisions were made in the meeting?",
            "Which topic was discussed the most in the meeting?",
            "Was the meeting's time effectively managed?",
            "Was there conflict during the meeting?",
            "Were there any profanities during the meeting?",
            "Were there any ulterior motives or hidden symbolism during the meeting?",
            "Which points were confusing or needed clarification amongst participants?",
            "Were there any unexpected challenges or roadblocks?",
            "What was the most interesting takeaway from the meeting?",
            "Who spoke the most?",
            "Who spoke the least?",
            "Who did not speak at all?",
            "Who was the most confused during the meeting?",
            "Who contributed the most during the meeting?",
            "What was the tone of the meeting?",
            "Were any deadlines discussed during the meeting?",
            "Were any projects discussed during the meeting?",
            "Do you think the meeting was a good use of peoples time?",
            "Were there any people mentioned in the meeting that did not attend?",
            "How could the meeting be improved?",
            "Summarise the meeting.",
            "Summarise the meeting for a manager.",
            "Summarise the meeting for a engineer.",
            "Summarise the meeting for a developer.",
            "Summarise the meeting for a salesperson.",
            "Summarise the meeting for a CFO.",
            "Summarise the meeting for a CEO.",
            "Turn the meeting into a blog post.",
            "Turn the meeting into a professional journal article."
         ],
         key="prompt_selection",
    )
    c1, c2 = st.columns(2)
    with c1:
        st.button("Submit prompts", on_click=submit_prompts)
    with c2:
        st.button("Clear history", on_click=clear_dialogue)

    
if uploaded_file:
    with st.spinner("Parsing file..."):
        try:
            content = uploaded_file.read().decode("utf-8")
            uploaded_file.seek(0)
            dialogue = ""
            dialogues = []
            prev_user = ""
            for line in repr(content).split("\\r\\n"):
                try:
                    user, message = line.split(':')
                    if user != prev_user:
                        text = f"{user}: {message}"
                        dialogue += f"\n{text}"
                        prev_user = user
                        dialogues.append(f"**{user}**")
                        dialogues.append(f"> {message}")
                    else:
                        dialogue += f" {message}"
                        dialogues[-1] += f" {message}"
                except ValueError:
                    pass
        except Exception as e:
            st.error("Error reading file.")
            st.error(e)
            st.stop()
    
    SYSTEM_PROMPT = f"""Generate an answer to the user's question based on the meeting dialogue.
MEETING: {dialogue}
"""

    with st.sidebar:
        if askai == "OpenAI":
            SYSTEM_PROMPT_TOKENS = len(encoder.encode(SYSTEM_PROMPT))
            st.success(f"Transcript contains {SYSTEM_PROMPT_TOKENS} tokens.")
        with st.expander("Expand to see dialogue", expanded=False):
            st.markdown("\n\n".join(dialogues), unsafe_allow_html=True)

for message in st.session_state.dialogue:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

pq = st.chat_input("Ask something")
if pq or st.session_state["query"]:
    query = st.session_state["query"]
    if pq:
        query = [pq]
        
    st.session_state["query"] = []
    
    with st.chat_message("user"):
        for q in query:
            st.markdown(q)

    # Add query to history
    st.session_state["dialogue"].append({"role": "user", "content": "\n\n".join(query)})

    USER_PROMPT = "\n" + f"""
{" ".join(query)}. Format the response using markdown and include unicode emojis where appropriate."""

    if askai == "OpenAI":
        USER_PROMPT_TOKENS = len(encoder.encode(USER_PROMPT))
        TOTAL_PROMPT_TOKENS = SYSTEM_PROMPT_TOKENS + USER_PROMPT_TOKENS + 12

    with st.spinner(f"Asking {askai}..."):

        NEW_PROMPT = SYSTEM_PROMPT + USER_PROMPT
        max_tokens = st.session_state["max_tokens"]
        if askai == "OpenAI":
            spare_tokens = max_token_value - TOTAL_PROMPT_TOKENS
            if spare_tokens < max_tokens:
                max_tokens = spare_tokens

        temperature = st.session_state["temperature"]

        if askai == "OpenAI":
            response = openai.ChatCompletion.create(
                engine=deployment_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT}
                ],
                max_tokens = max_tokens,
                temperature = temperature,
                stream=streaming
            )
        elif askai == "Anthropic":
            prompt = f"{HUMAN_PROMPT} {SYSTEM_PROMPT} {USER_PROMPT} {AI_PROMPT}"
            response = anthropic.completions.create(
                model=deployment_name,
                max_tokens_to_sample=max_tokens,
                prompt=prompt,
                temperature=temperature,
                stream=streaming
            )

    with st.chat_message("assistant"):
        response_widget = st.empty()
        full_response = ""
        if streaming:
            for completion in response:
                if askai == "OpenAI":
                    partial_response = completion["choices"][0]["delta"].get("content", "")
                elif askai == "Anthropic":
                    partial_response = completion.completion
                full_response += partial_response
                response_widget.markdown(full_response)
        else:
            if askai == "OpenAI":
                full_response = response["choices"][0].message.content
                response_widget.markdown(full_response)
            elif askai == "Anthropic":
                pass
        c1, c2, c3 = st.columns([0.05, 0.05, 0.9])
        with c1:
            st.button("👍", on_click=positive, args=(nr_event_client,))
        with c2:
            st.button("👎", on_click=negative, args=(nr_event_client,))
        st.session_state["dialogue"].append({"role": "assistant", "content": full_response})